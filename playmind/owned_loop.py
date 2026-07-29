"""Owned-game observe→plan→act loop using capture + vision + keyboard.

Supports play+learn: vision-shaped rewards update a tabular Q policy while
the farm heuristic (or learned policy) drives the character.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from playmind.actuators import (
    Actuator,
    DryRunKeyboardActuator,
    OwnedGameKeyboardActuator,
    load_keymap,
)
from playmind.capture import capture_config_from_dict
from playmind.learning import (
    OWNED_ACTIONS,
    ExperienceBuffer,
    OnlinePolicy,
    owned_state_key,
    reward_owned,
)
from playmind.planner import HeuristicPlanner, OllamaPlanner, ollama_available
from playmind.ability_memory import (
    AbilityMemory,
    ensure_ability_seeded,
    parse_dynamic_action,
)
from playmind.council import TeacherBrain
from playmind.directive import (
    Goal,
    directive_reward_bonus,
    goal_action_hint,
    parse_directive,
)
from playmind.screen_llm import ScreenLLMPlanner, enrich_obs_from_screen
from playmind.session import SessionConfig, SessionScheduler
from playmind.stuck import StuckTracker, detect_blocking_modal
from playmind.progress import ProgressTracker, ocr_says_no_target
from playmind.process_memory import ProcessMemory
from playmind.travel import TravelMemory
from playmind.ui_memory import UIMemory, discover_and_remember, ensure_seeded
from playmind.soul import feel_soul
from playmind.life_fsm import LifeFSM
from playmind.vision import (
    detect_death_dialog,
    detect_hostile_nameplate_ocr,
    detect_target_bar,
    frame_motion,
    is_world_mob_label,
    read_frame,
)


@dataclass
class OwnedLoopConfig:
    config_path: Path = Path("config/owned_game.json")
    dry_run: bool = True
    use_ollama: bool = False
    ollama_model: str = "llama3.2"
    vision_model: str = "qwen2.5vl:7b"
    tick_seconds: float = 0.05
    max_ticks: int = 0  # 0 = until session stop
    data_dir: Path = Path("data/playmind/owned")
    learn: bool = True
    # Act from Q-table (with epsilon explore). Required for learning to change behavior.
    use_learned_policy: bool = True
    # Ask vision LLM rarely — most ticks are fast Q-learning updates.
    llm_mix: float = 0.02
    use_screen_llm: bool = True
    # Higher = many more learning steps per minute (VLM is the bottleneck).
    vision_every: int = 24
    epsilon: float = 0.3
    replay_n: int = 4
    save_every: int = 20
    # Full OCR discover is expensive; run every N ticks.
    ocr_every: int = 6
    # Teacher model reviews failures and teaches the Actor's Q-table.
    use_teacher: bool = True
    teacher_model: str = "llama3.2"
    teacher_every: int = 12  # at most every N bad ticks


def load_owned_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Copy config/owned_game.example.json and edit it."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _hp_roi_box(rois: dict[str, Any]) -> tuple[int, int, int, int] | None:
    if "hp_roi" not in rois:
        return None
    r = rois["hp_roi"]
    left, top, a, b = int(r[0]), int(r[1]), int(r[2]), int(r[3])
    if a > left and b > top:
        return (left, top, a, b)
    return (left, top, left + a, top + b)


def vision_obs_from_frame(
    frame_path: Path,
    rois: dict[str, Any],
    *,
    prev_frame: Path | None = None,
    steps: int = 0,
    light: bool = False,
) -> dict[str, Any]:
    hp_roi = _hp_roi_box(rois)
    reading = read_frame(frame_path, hp_roi=hp_roi, do_ocr=False)
    motion = frame_motion(prev_frame, frame_path)
    hp = reading.player_hp if reading.player_hp is not None else 1.0
    if light:
        # Fast aftermath path — targeting/death come from enrich + sticky FSM.
        obs: dict[str, Any] = {
            "player": {"x": 0, "y": 0, "hp": hp},
            "adjacent_enemies": [],
            "npc_here": False,
            "npc_pos": {"x": 0, "y": 0},
            "quest_kills": 0,
            "quest_kills_needed": 0,
            "quest_complete": False,
            "steps": steps,
            "quest_text": None,
            "has_target": False,
            "in_combat": False,
            "hostiles_near": False,
            "is_dead": False,
            "is_ghost": False,
            "target_hp_est": None,
            "motion": motion,
        }
        obs.update(reading.to_obs_patch())
        return obs

    has_target, target_hp = detect_target_bar(frame_path)
    is_dead, is_ghost = detect_death_dialog(frame_path)
    if is_dead:
        hp = 0.0
    # Nearby nameplates ≠ selected target. Selection = red ring / unit frame only.
    ocr_blob = f"{reading.quest_text or ''} {reading.raw_text or ''}"
    if ocr_says_no_target(ocr_blob):
        has_target = False
        target_hp = None
    hostiles_near = detect_hostile_nameplate_ocr(ocr_blob)
    obs = {
        "player": {"x": 0, "y": 0, "hp": hp},
        "adjacent_enemies": [{"name": "target", "hp": target_hp or 1.0}] if has_target else [],
        "npc_here": False,
        "npc_pos": {"x": 0, "y": 0},
        "quest_kills": 0,
        "quest_kills_needed": 0,
        "quest_complete": False,
        "steps": steps,
        "quest_text": reading.quest_text,
        "has_target": has_target and not is_dead,
        "in_combat": has_target and not is_dead,
        "hostiles_near": hostiles_near and not is_dead,
        "is_dead": is_dead,
        "is_ghost": is_ghost,
        "target_hp_est": target_hp,
        "motion": motion,
    }
    obs.update(reading.to_obs_patch())
    if is_dead:
        obs["vision_player_hp"] = 0.0
    return obs


@dataclass
class OwnedGameLoop:
    cfg: OwnedLoopConfig = field(default_factory=OwnedLoopConfig)
    directive: str | None = None
    on_status: Callable[[dict[str, Any]], None] | None = None
    should_stop: Callable[[], bool] | None = None
    _tick: int = 0
    _ui_memory: UIMemory | None = None
    _ability_memory: AbilityMemory | None = None
    _stuck: StuckTracker = field(default_factory=StuckTracker)
    _last_llm_action: str | None = None
    _last_llm_raw: str = ""
    _decision_reason: str = ""
    _goal: Goal | None = None
    _teacher: TeacherBrain | None = None
    _teacher_cooldown: int = 0
    _last_ocr: str = ""
    _life: LifeFSM = field(default_factory=LifeFSM)
    _travel: TravelMemory = field(default_factory=TravelMemory)
    _progress: ProgressTracker = field(default_factory=ProgressTracker)
    _process: ProcessMemory | None = None
    _recent: list[tuple[dict[str, Any], str]] = field(default_factory=list)

    def _action_space(self, policy: OnlinePolicy, extra: str | None = None) -> list[str]:
        acts: list[str] = list(OWNED_ACTIONS)
        for bucket in policy.q.values():
            acts.extend(bucket.keys())
        acts.extend(
            [
                "key:1",
                "key:2",
                "key:3",
                "key:4",
                "key:5",
                "key:tab",
                "key:esc",
                "hold:w:0.6",
                "hold:w:1.1",
                "hold:d:1.1",
                "hold:a:1.1",
                "hold:s:1.1",
                "click_label:Close",
                "click_label:Accept",
            ]
        )
        if extra:
            acts.append(extra)
        # While alive, strip poisoned death-UI click actions from Q choices.
        if self._life.phase == "alive":
            cleaned: list[str] = []
            for a in acts:
                low = a.lower()
                if low.startswith("click:"):
                    continue
                if any(
                    k in low
                    for k in (
                        "closest town",
                        "closest city",
                        "graveyard",
                        "safe zone",
                        "release_spirit",
                        "click_label:yes",
                        "click_label:ok",
                    )
                ):
                    continue
                cleaned.append(a)
            acts = cleaned
        return list(dict.fromkeys(acts))

    def run(self) -> None:
        owned = load_owned_config(self.cfg.config_path)
        if not owned.get("i_own_this_game", False):
            raise SystemExit(
                "Refusing to run: set i_own_this_game=true in owned_game.json "
                "only for games you own / may automate."
            )

        keymap = load_keymap(Path(owned.get("keymap_path", "config/keymap.example.json")))
        capture_cfg = owned.get("capture", {"mode": "monitor", "monitor_index": 1})
        rois = owned.get("rois", {})
        session_cfg = SessionConfig(**owned.get("session", {}))
        scheduler = SessionScheduler(config=session_cfg)

        ui_memory = UIMemory(self.cfg.data_dir / "ui_memory.json")
        ensure_seeded(ui_memory)
        self._ui_memory = ui_memory
        ability_memory = AbilityMemory(self.cfg.data_dir / "ability_memory.json")
        ensure_ability_seeded(ability_memory)
        self._ability_memory = ability_memory
        self._process = ProcessMemory(self.cfg.data_dir / "process_memory.json")
        self._process.load()
        self._process.restore_travel(self._travel)
        self._recent = []

        actuator: Actuator
        if self.cfg.dry_run or not owned.get("enable_keyboard", False):
            actuator = DryRunKeyboardActuator(
                keymap=keymap, log_path=self.cfg.data_dir / "dryrun.jsonl"
            )
            mode = "dry_run"
        else:
            window_title = str(
                owned.get("capture", {}).get("window_title")
                or owned.get("game_name")
                or ""
            )
            from playmind.actuators import assert_can_control_window

            if window_title:
                assert_can_control_window(window_title)
            actuator = OwnedGameKeyboardActuator(
                keymap=keymap,
                enabled=True,
                i_own_this_game=True,
                window_title_substr=window_title,
                ui_memory=ui_memory,
                ability_memory=ability_memory,
                log_path=self.cfg.data_dir / "keys.jsonl",
            )
            mode = "live_keyboard"

        planner: Any
        screen_brain: ScreenLLMPlanner | None = None
        if self.cfg.use_ollama:
            if not ollama_available():
                print(
                    "WARNING: --ollama set but Ollama is not reachable at "
                    "http://127.0.0.1:11434 — falling back to heuristic until it is up."
                )
            text_fallback = OllamaPlanner.for_owned(
                model=self.cfg.ollama_model,
                fallback=_OwnedHeuristicFallback(),
            )
            if self.cfg.use_screen_llm:
                screen_brain = ScreenLLMPlanner(
                    vision_model=self.cfg.vision_model,
                    text_model=self.cfg.ollama_model,
                    fallback=text_fallback,
                    ability_summary=ability_memory.known_summary(),
                )
                planner = screen_brain
                brain = f"screen_llm:{self.cfg.vision_model}|{self.cfg.ollama_model}"
            else:
                planner = text_fallback
                brain = f"ollama:{self.cfg.ollama_model}"
        else:
            planner = HeuristicPlanner()
            brain = "heuristic"

        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        policy = OnlinePolicy(
            epsilon=self.cfg.epsilon,
            alpha=0.45,
            gamma=0.9,
            key_fn=owned_state_key,
        )
        policy_path = self.cfg.data_dir / "policy.json"
        policy.load(policy_path)
        scrubbed = 0
        for bucket in policy.q.values():
            for act in list(bucket.keys()):
                low = str(act).lower()
                if low.startswith("click:") or any(
                    k in low
                    for k in (
                        "closest town",
                        "closest city",
                        "safe zone",
                        "graveyard",
                        "release_spirit",
                        "click_label:yes",
                        "click_label:ok",
                        "click_label:a oe",
                    )
                ):
                    del bucket[act]
                    scrubbed += 1
        if scrubbed:
            print(f"Scrubbed {scrubbed} poisoned death-click actions from Q")
            policy.save(policy_path)
        buffer = ExperienceBuffer(
            self.cfg.data_dir / "experience.jsonl",
            key_fn=owned_state_key,
        )
        prior = buffer.load()

        self._goal = parse_directive(self.directive)
        print(f"Directive goal: {self._goal.summary()} ({self._goal.raw!r})")
        if self.cfg.use_teacher and self.cfg.learn:
            self._teacher = TeacherBrain(
                model=self.cfg.teacher_model,
                path=self.cfg.data_dir / "lessons.jsonl",
            )
            print(f"Teacher brain: {self.cfg.teacher_model} (reviews failures → teaches Actor)")
        else:
            self._teacher = None

        ticks = 0
        total_reward = 0.0
        prev_frame_path: Path | None = None
        print(
            f"Owned loop starting mode={mode} brain={brain} learn={self.cfg.learn} "
            f"use_learned={self.cfg.use_learned_policy} "
            f"vision_every={self.cfg.vision_every} replay={self.cfg.replay_n} "
            f"teacher={bool(self._teacher)} prior_xp={prior} config={self.cfg.config_path}"
        )

        try:
            while True:
                if scheduler.should_stop():
                    print("Max session wall time reached; stopping.")
                    break
                if self.should_stop and self.should_stop():
                    print("Stop requested; stopping.")
                    break
                if self.cfg.max_ticks and ticks >= self.cfg.max_ticks:
                    print("Max ticks reached; stopping.")
                    break

                if scheduler.should_start_break():
                    mins = scheduler.start_break()
                    print(f"Break started (~{mins:.1f} min). Sending logout action.")
                    actuator.send(owned.get("logout_action", "logout"))
                    while not scheduler.break_done():
                        time.sleep(1.0)
                    scheduler.end_break()
                    print("Break over; resuming.")

                frame_path = self.cfg.data_dir / "latest.png"
                try:
                    cap = capture_config_from_dict(capture_cfg, frame_path)
                except Exception as exc:  # noqa: BLE001
                    print(f"capture failed: {exc}")
                    time.sleep(1.0)
                    continue

                # Keep previous frame for motion / reward.
                snap_path = self.cfg.data_dir / "prev.png"
                if frame_path.exists():
                    # Capture already wrote latest; use prior snap if present.
                    pass

                obs = vision_obs_from_frame(
                    cap.path, rois, prev_frame=prev_frame_path, steps=ticks
                )
                # Cheap pixel sensors first; always peek death-dialog OCR (truth).
                obs = enrich_obs_from_screen(cap.path, obs, do_ocr=False)
                # Death UI lives in the dialog band — never skip it when world is grey,
                # and scan it often anyway so we don't "feel alive" on a corpse.
                need_death_ocr = bool(
                    obs.get("desaturated")
                    or obs.get("is_dead")
                    or obs.get("is_ghost")
                    or (self._life.phase != "alive")
                    or (ticks % 2 == 0)  # every other tick: cross-check death ROI
                )
                ocr_mode = "death" if need_death_ocr else "alive"
                run_ocr = True  # always OCR something — lies cost more than ~150ms
                if run_ocr:
                    obs = enrich_obs_from_screen(
                        cap.path, obs, do_ocr=True, ocr_mode=ocr_mode
                    )
                    # Merge a death-band peek into alive OCR so phrases aren't missed.
                    if ocr_mode == "alive":
                        death_peek = enrich_obs_from_screen(
                            cap.path, {}, do_ocr=True, ocr_mode="death"
                        )
                        docr = death_peek.get("screen_ocr") or ""
                        if docr:
                            obs["screen_ocr"] = (
                                f"{obs.get('screen_ocr') or ''} | {docr}"
                            ).strip(" |")
                    if ticks % max(1, self.cfg.ocr_every) == 0 or need_death_ocr:
                        ui_hits = discover_and_remember(
                            cap.path, ui_memory, mode="death" if need_death_ocr else ocr_mode
                        )
                    else:
                        ui_hits = []
                else:
                    if getattr(self, "_last_ocr", None):
                        obs["screen_ocr"] = self._last_ocr
                    ui_hits = []
                if obs.get("screen_ocr"):
                    self._last_ocr = obs["screen_ocr"]
                # Full OCR arrives in enrich — refresh hostiles_near from it.
                ocr_all = f"{obs.get('screen_ocr') or ''} {obs.get('quest_text') or ''}"
                if detect_hostile_nameplate_ocr(ocr_all):
                    obs["hostiles_near"] = True
                # Nameplates are not UI buttons — drop them from click candidates.
                clean_hits = [h.label for h in ui_hits[:12] if not is_world_mob_label(h.label)]
                obs["ui_hits"] = clean_hits if clean_hits else [
                    h for h in (obs.get("ui_hits") or []) if not is_world_mob_label(str(h))
                ]
                obs["ui_known"] = len(ui_memory.known_labels())
                obs["ability_summary"] = ability_memory.known_summary()
                obs["abilities_known"] = ability_memory.known()
                # Modal / stuck signals for the brain
                if detect_blocking_modal(obs):
                    obs["modal_menu"] = True
                if self._stuck.is_stuck():
                    obs["stuck_hint"] = (
                        f"action {self._stuck.last_action!r} failed "
                        f"{self._stuck.same_fail_streak}x with no effect — do NOT repeat"
                    )
                else:
                    obs["stuck_hint"] = "none"
                goal = self._goal or parse_directive(self.directive)
                obs["goal_summary"] = goal.summary()
                obs["goal_rules"] = goal.prompt_rules()
                obs["goal_kind"] = goal.kind
                # Sticky life phase before soul/decision — stops dead↔alive thrash.
                self._life.update(obs)
                obs = self._life.patch_obs(obs)
                # Progressive curriculum: veto false targets, escalate when stagnant.
                obs = self._progress.patch_obs(obs)
                soul = feel_soul(obs, frame_path=cap.path)
                obs.update(soul.to_obs())
                if screen_brain is not None:
                    screen_brain.ability_summary = obs["ability_summary"]
                self._tick = ticks
                action = self._choose_action(policy, planner, obs, frame_path=cap.path)
                action = self._reject_invalid_action(action, policy, obs)
                # Let click_label discover via live OCR on this frame.
                if hasattr(actuator, "last_frame"):
                    actuator.last_frame = cap.path
                llm_raw = getattr(planner, "last_raw", "") or self._last_llm_raw or ""
                brain_mode = getattr(planner, "last_mode", "") or ""
                thinking = self._format_thinking(
                    policy, obs, action, llm_raw, brain_mode
                )
                if screen_brain is not None:
                    obs["screen_see"] = screen_brain.last_see
                    obs["brain_mode"] = screen_brain.last_mode
                actuator.send(action)

                time.sleep(self.cfg.tick_seconds)

                # Observe aftermath for learning.
                next_path = self.cfg.data_dir / "latest_next.png"
                reward = 0.0
                next_obs = obs
                try:
                    cap2 = capture_config_from_dict(capture_cfg, next_path)
                    deathish = bool(
                        obs.get("is_dead")
                        or obs.get("is_ghost")
                        or obs.get("desaturated")
                        or self._life.phase != "alive"
                    )
                    next_obs = vision_obs_from_frame(
                        cap2.path,
                        rois,
                        prev_frame=cap.path,
                        steps=ticks + 1,
                        light=not deathish,
                    )
                    # Re-check target cheaply — do NOT carry sticky false locks.
                    if not deathish:
                        try:
                            ht, thp = detect_target_bar(cap2.path)
                            next_obs["has_target"] = bool(ht)
                            next_obs["in_combat"] = bool(ht)
                            next_obs["target_hp_est"] = thp
                        except Exception:
                            next_obs["has_target"] = False
                            next_obs["in_combat"] = False
                    # Aftermath: full OCR only while dead/ghost — alive uses pixels.
                    next_obs = enrich_obs_from_screen(
                        cap2.path,
                        next_obs,
                        do_ocr=deathish,
                        ocr_mode="death" if deathish else "alive",
                    )
                    if not deathish and obs.get("screen_ocr"):
                        next_obs["screen_ocr"] = obs.get("screen_ocr")
                    ocr_n = f"{next_obs.get('screen_ocr') or ''} {next_obs.get('quest_text') or ''}"
                    if detect_hostile_nameplate_ocr(ocr_n):
                        next_obs["hostiles_near"] = True
                    if detect_blocking_modal(next_obs):
                        next_obs["modal_menu"] = True
                    if self.cfg.learn:
                        # Sticky phase on aftermath so rewards see UI progress.
                        self._life.update(next_obs)
                        next_obs = self._life.patch_obs(next_obs)
                        reward = reward_owned(obs, action, next_obs)
                        goal = self._goal or parse_directive(self.directive)
                        reward += directive_reward_bonus(goal, obs, action, next_obs)
                        reward += self._progress.reward_bonus(obs, action, next_obs)
                        self._progress.note(obs, action, next_obs, reward)
                        next_obs = self._progress.patch_obs(next_obs)
                        # Death cause → prevention; successful death clicks → pipeline memory.
                        if self._process is not None:
                            was_alive = (obs.get("life_phase") or "alive") == "alive" and not obs.get(
                                "is_dead"
                            )
                            now_dead = bool(next_obs.get("is_dead")) or (
                                next_obs.get("life_phase")
                                in {"dead_dialog", "confirm", "rez_picker", "ghost"}
                            )
                            if was_alive and now_dead:
                                cause = self._process.note_death_cause(obs, action)
                                if self.cfg.learn and self.cfg.use_learned_policy:
                                    # Teach flee in the pre-death state.
                                    policy.teach(obs, "hold:s:1.2", boost=0.9)
                                    policy.teach(obs, "hold:w:1.2", boost=0.5)
                                    if action.startswith("key:") or action == "attack":
                                        policy.teach(obs, action, boost=-0.7)
                                next_obs["death_cause"] = cause
                            phase0 = str(obs.get("life_phase") or "")
                            phase1 = str(next_obs.get("life_phase") or "")
                            order = ("dead_dialog", "confirm", "rez_picker", "ghost", "alive")
                            if phase0 in order and phase1 in order:
                                if order.index(phase1) > order.index(phase0) or (
                                    bool(obs.get("is_dead")) and not bool(next_obs.get("is_dead"))
                                ):
                                    self._process.note_pipeline(phase0, action, success=True)
                                elif phase0 == phase1 and phase0 != "alive" and reward < 0.2:
                                    self._process.note_pipeline(phase0, action, success=False)
                            if ticks % 10 == 0:
                                self._process.apply_travel_snapshot(self._travel)
                                self._process.save()
                        self._recent.append((dict(obs), action))
                        self._recent = self._recent[-20:]
                        if (
                            self._stuck.last_action == action
                            and reward <= 0.05
                            and owned_state_key(obs) == owned_state_key(next_obs)
                            and not (obs.get("is_dead") or obs.get("is_ghost"))
                        ):
                            reward -= 0.35
                        # Mark failed death probes so next pick is a new random try.
                        if (obs.get("is_dead") or obs.get("life_phase") in {
                            "dead_dialog",
                            "confirm",
                            "rez_picker",
                        }) and reward < 0.5:
                            self._life.note_try_failed(action)
                        reward = round(reward, 4)
                        total_reward += reward
                        space = self._action_space(policy, action)
                        policy.update(obs, action, reward, next_obs, False, space)
                        buffer.add(obs, action, reward, next_obs, False, source="owned")
                        buffer.append_save()
                        # Replay recent XP so each wall-clock second teaches more.
                        if self.cfg.replay_n > 0:
                            policy.replay(buffer.rows, space, n=self.cfg.replay_n)
                        self._reinforce_ui(action, reward, obs, next_obs)
                        # Teacher reviews failures and boosts better actions into Q.
                        if (
                            self._teacher is not None
                            and self._teacher_cooldown <= 0
                            and (reward <= -0.05 or self._stuck.same_fail_streak >= 1)
                        ):
                            better = self._teacher.maybe_teach(
                                policy=policy,
                                obs=obs,
                                action=action,
                                reward=reward,
                                next_obs=next_obs,
                                goal_summary=(self._goal.summary() if self._goal else ""),
                                actions=space,
                                force=self._stuck.same_fail_streak >= 2,
                            )
                            self._teacher_cooldown = max(1, self.cfg.teacher_every)
                            if better:
                                thinking = (
                                    thinking
                                    + f"\nTEACHER: {action} (r={reward}) → {better}"
                                )
                        elif self._teacher_cooldown > 0:
                            self._teacher_cooldown -= 1
                    self._stuck.note_outcome(
                        action,
                        owned_state_key(next_obs),
                        reward,
                        motion=float(next_obs.get("motion") or 0),
                        modal=bool(obs.get("modal_menu") or next_obs.get("modal_menu")),
                    )
                    self._travel.note(
                        action,
                        motion=float(next_obs.get("motion") or 0),
                        had_target=bool(obs.get("has_target")),
                        reward=reward,
                        tick=ticks,
                    )
                    # Roll frames: latest <- next, prev <- old latest
                    shutil.copy2(cap.path, snap_path)
                    shutil.copy2(cap2.path, frame_path)
                    prev_frame_path = snap_path
                except Exception as exc:  # noqa: BLE001
                    print(f"post-action capture failed: {exc}")
                    self._stuck.note_outcome(
                        action,
                        owned_state_key(obs),
                        -0.2,
                        motion=0.0,
                        modal=bool(obs.get("modal_menu")),
                    )

                ticks += 1
                if self.cfg.learn and ticks % max(1, self.cfg.save_every) == 0:
                    policy.save(policy_path)

                status = {
                    "tick": ticks,
                    "action": action,
                    "reward": reward,
                    "total_reward": round(total_reward, 3),
                    "state": owned_state_key(obs),
                    "has_target": obs.get("has_target"),
                    "capture": cap.backend,
                    "vision_hp": obs.get("vision_player_hp"),
                    "motion": round(float(next_obs.get("motion") or 0), 2),
                    "learn": self.cfg.learn,
                    "use_learned": self.cfg.use_learned_policy,
                    "brain": brain,
                    "brain_mode": self._decision_reason
                    or getattr(planner, "last_mode", "")
                    or obs.get("brain_mode")
                    or "",
                    "decision": self._decision_reason,
                    "is_dead": obs.get("is_dead"),
                    "is_ghost": obs.get("is_ghost"),
                    "screen_ocr": (obs.get("screen_ocr") or "")[:200],
                    "ui_hits": obs.get("ui_hits") or [],
                    "ui_known": obs.get("ui_known"),
                    "abilities": obs.get("abilities_known") or ability_memory.known(),
                    "ability_summary": obs.get("ability_summary") or "",
                    "llm_raw": (llm_raw or getattr(planner, "last_raw", "") or "")[:600],
                    "llm_error": getattr(planner, "last_error", "") or "",
                    "q_top": policy.q_snapshot(obs),
                    "thinking": thinking[:1200],
                    "stuck": self._stuck.same_fail_streak,
                    "stuck_hint": obs.get("stuck_hint") or "",
                    "modal_menu": bool(obs.get("modal_menu")),
                    "goal": obs.get("goal_summary") or "",
                    "directive": self.directive or "",
                    "soul": obs.get("soul_summary") or "",
                    "soul_body": obs.get("soul_body") or "",
                    "life_phase": obs.get("life_phase") or "",
                    "travel_heading": self._travel.heading,
                    "travel_commit": self._travel.commit_left,
                    "still_farm": self._travel.still_farm,
                    "progress_stage": obs.get("progress_stage") or self._progress.stage,
                    "stagnant": obs.get("stagnant") if obs.get("stagnant") is not None else self._progress.stagnant,
                    "no_damage_casts": obs.get("no_damage_casts")
                    if obs.get("no_damage_casts") is not None
                    else self._progress.no_damage_casts,
                    "process_memory": self._process.summary() if self._process else "",
                    "preventions": list(self._process.preventions) if self._process else [],
                    "bar_slots": obs.get("bar_slots_filled"),
                    "teacher_teaches": getattr(self._teacher, "teaches", 0) if self._teacher else 0,
                    "teacher_last": getattr(self._teacher, "last_better", "") if self._teacher else "",
                    "session": scheduler.status(),
                    "mode": mode,
                    "frame": str(cap.path),
                }
                if self.on_status:
                    self.on_status(status)
                else:
                    print(status)
        finally:
            if self._process is not None:
                self._process.apply_travel_snapshot(self._travel)
                self._process.save()
            if self.cfg.learn:
                policy.save(policy_path)
                n = buffer.export_finetune_jsonl(self.cfg.data_dir / "finetune.jsonl")
                print(
                    f"Saved policy={policy_path} experience={buffer.path} "
                    f"finetune_rows={n} total_reward={total_reward:.2f} "
                    f"process={self._process.summary() if self._process else ''}"
                )

    def _reinforce_ui(
        self,
        action: str,
        reward: float,
        obs: dict[str, Any],
        next_obs: dict[str, Any],
    ) -> None:
        """If a dynamic click/ability helped, mark that memory entry a success."""
        mem = getattr(self, "_ui_memory", None)
        abil = getattr(self, "_ability_memory", None)
        dyn = parse_dynamic_action(action)
        if dyn is None and action != "release_spirit":
            return
        hp0 = float(obs.get("vision_player_hp") or 0)
        hp1 = float(next_obs.get("vision_player_hp") or 0)
        left_death = bool(obs.get("is_dead")) and not bool(next_obs.get("is_dead"))
        confirm_cleared = bool(obs.get("confirm_pending")) and not bool(
            next_obs.get("confirm_pending")
        )
        ocr0 = (obs.get("screen_ocr") or "").lower()
        ocr1 = (next_obs.get("screen_ocr") or "").lower()
        sure_cleared = ("are you sure" in ocr0) and ("are you sure" not in ocr1)
        picker_cleared = ("choose where" in ocr0) and ("choose where" not in ocr1)
        phase0 = str(obs.get("life_phase") or "")
        phase1 = str(next_obs.get("life_phase") or "")
        order = ("dead_dialog", "confirm", "rez_picker", "ghost", "alive")
        progressed = (
            phase0 in order and phase1 in order and order.index(phase1) > order.index(phase0)
        )
        dialog_gone = bool(obs.get("ghost_buttons") or obs.get("is_dead")) and not (
            next_obs.get("ghost_buttons") or next_obs.get("is_dead")
        )
        success = (
            reward > 0.5
            or hp1 > hp0 + 0.05
            or dialog_gone
            or left_death
            or confirm_cleared
            or sure_cleared
            or picker_cleared
            or progressed
        )
        if not success:
            return
        if mem is not None:
            if dyn and dyn.get("type") == "click_label":
                hit = mem.lookup(str(dyn["label"]))
                if hit:
                    mem.remember(
                        str(dyn["label"]), hit[0], hit[1], source="reinforce", success=True
                    )
                # Also bind semantic aliases when confirm cleared
                if sure_cleared or confirm_cleared:
                    if hit:
                        mem.remember("yes", hit[0], hit[1], source="reinforce", success=True)
            elif dyn and dyn.get("type") == "click_frac":
                fx, fy = float(dyn["fx"]), float(dyn["fy"])
                mem.remember(
                    f"click {fx:.2f} {fy:.2f}",
                    fx,
                    fy,
                    source="reinforce",
                    success=True,
                )
                # Bind the successful probe to the semantic button we were hunting.
                if sure_cleared or confirm_cleared:
                    label = "yes"
                elif picker_cleared or phase0 == "rez_picker":
                    label = "closest town"
                elif left_death or phase0 == "dead_dialog":
                    label = "return to graveyard"
                else:
                    label = None
                if label:
                    mem.remember(label, fx, fy, source="discover_success", success=True)
                    self._life._tried.clear()
        if abil is not None and dyn:
            if dyn.get("type") == "bind":
                abil.bind(str(dyn["name"]), str(dyn["key"]), source="reinforce", success=True)
            elif dyn.get("type") == "ability":
                row = abil.lookup(str(dyn["name"]))
                if row and row.get("key"):
                    abil.bind(
                        str(dyn["name"]),
                        str(row["key"]),
                        hold=row.get("hold"),
                        source="reinforce",
                        success=True,
                    )
            elif dyn.get("type") == "key":
                abil.bind(f"key {dyn['key']}", str(dyn["key"]), source="reinforce", success=True)

    def _reject_invalid_action(
        self,
        action: str,
        policy: OnlinePolicy,
        obs: dict[str, Any],
    ) -> str:
        """Drop illegal nameplate clicks; re-sample from policy (no forced combat)."""
        a = (action or "").strip()
        low = a.lower()
        phase = str(obs.get("life_phase") or "alive")
        alive = phase == "alive" and not obs.get("is_dead") and not obs.get("is_ghost")
        deathish = (
            low == "release_spirit"
            or "graveyard" in low
            or "closest town" in low
            or "closest city" in low
            or "safe zone" in low
            or low.startswith("click_label:yes")
            or low.startswith("click_label:resurrect")
            or low.startswith("click_label:ok")
            or low.startswith("click_label:accept")
            or low.startswith("click_label:a oe")
            or "sanctuary" in low
        )
        # Poisoned Q keeps death-dialog pixel clicks while alive.
        death_click = False
        if alive and low.startswith("click:"):
            try:
                fx_s, fy_s = low.split(":", 1)[1].split(",")
                fx, fy = float(fx_s), float(fy_s)
                # Top-center rez/confirm band — not world clicks.
                if 0.25 <= fx <= 0.70 and 0.05 <= fy <= 0.35:
                    death_click = True
            except Exception:
                death_click = True
        if alive and (deathish or death_click):
            space = [
                x
                for x in self._action_space(policy)
                if x.lower() != "release_spirit"
                and "graveyard" not in x.lower()
                and "closest" not in x.lower()
                and not x.lower().startswith("click:")
                and not (
                    x.lower().startswith("click_label:")
                    and any(
                        k in x.lower()
                        for k in ("safe", "graveyard", "closest", "yes", "ok", "resurrect")
                    )
                )
            ]
            # Prefer travel over more combat spam when rejecting.
            if self._travel.needs_travel(obs) or random.random() < 0.5:
                alt, reason = self._travel.action(obs)
                self._decision_reason = (
                    (self._decision_reason or "") + f" | reject_death_click→{reason}"
                )
                return alt
            alt = policy.choose(obs, space) if space else "hold:w:1.1"
            if alt.lower().startswith("click_label:") and is_world_mob_label(alt.split(":", 1)[-1]):
                alt = "key:tab"
            self._decision_reason = (
                (self._decision_reason or "") + f" | reject_death_while_alive→{alt}"
            )
            return alt
        if alive and low in {"key:esc", "key:escape"} and not obs.get("modal_menu"):
            # Esc mid-fight opens Options — don't.
            alt = "key:1" if obs.get("has_target") else "hold:w:1.1"
            self._decision_reason = (self._decision_reason or "") + f" | reject_esc→{alt}"
            return alt
        if low.startswith("click_label:"):
            label = a.split(":", 1)[1]
            lab = label.lower()
            if (
                is_world_mob_label(label)
                or "shadowglen" in lab
                or "shadowgle" in lab
                or "aldrassil" in lab
            ):
                alt = "key:1" if obs.get("has_target") else "key:tab"
                self._decision_reason = (
                    (self._decision_reason or "")
                    + f" | reject_mob_click({label})→{alt}"
                )
                return alt
        return a

    def _format_thinking(
        self,
        policy: OnlinePolicy,
        obs: dict[str, Any],
        action: str,
        llm_raw: str,
        brain_mode: str,
    ) -> str:
        lines = [
            f"SOUL: {obs.get('soul_summary') or '(waking…)'}",
            f"GOAL: {obs.get('goal_summary') or self.directive or '(none)'}",
            f"DECISION: {self._decision_reason or brain_mode or '?'}",
            f"ACT: {action}",
            f"LLM SAID: {(llm_raw or '(no vision call this tick)').strip()[:400]}",
            f"Q TOP: {policy.q_snapshot(obs)}",
            f"OCR: {(obs.get('screen_ocr') or '')[:180]}",
            f"STUCK: {obs.get('stuck_hint') or 'none'}",
            f"BODY: {obs.get('soul_body')}  PHASE: {obs.get('life_phase')}  "
            f"TARGET: {bool(obs.get('has_target'))}  "
            f"HOSTILES: {bool(obs.get('hostiles_near'))}  BAR: {obs.get('bar_slots_filled')}",
        ]
        return "\n".join(lines)

    def _choose_action(
        self,
        policy: OnlinePolicy,
        planner,
        obs: dict[str, Any],
        frame_path: Path | None = None,
    ) -> str:
        space = self._action_space(policy)

        # Life FSM owns death/ghost — Q and VLM cannot farm while grey.
        life_act, life_reason = self._life.action(
            obs,
            frame_path=frame_path,
            ui_memory=getattr(self, "_ui_memory", None),
            process_memory=self._process,
        )
        if life_act:
            self._decision_reason = life_reason
            return life_act

        # Learned preventions from past deaths (flee / leave bubble / retarget).
        if self._process is not None:
            prev = self._process.prevention_action(obs)
            if prev:
                act, reason = prev
                self._decision_reason = reason
                return act

        # Progressive learning override — leave the bubble when casts do nothing.
        forced = self._progress.force_action(obs)
        if forced:
            act, reason = forced
            self._decision_reason = reason
            return act

        # Travel memory: leave the spawn bubble — discover heading, remember walls.
        if self._travel.needs_travel(obs):
            suspect = (
                bool(obs.get("target_suspect"))
                or int(obs.get("no_damage_casts") or 0) >= 3
                or obs.get("progress_stage") in {"push", "break_loop"}
                or self._travel.still_farm >= 5
            )
            if (obs.get("has_target") or obs.get("in_combat")) and not suspect:
                self._travel.still_farm = 0
                self._travel.expect_target = False
                self._decision_reason = "travel:engage"
                return "key:1" if random.random() < 0.6 else "attack"
            # Stale / false target — do not mash 1; roam instead.
            if suspect:
                self._travel.still_farm = 0
                self._travel.expect_target = False
            # Tab sensor often misses Ascension red-ring — swing once after Tab.
            if self._travel.expect_target:
                self._travel.expect_target = False
                self._decision_reason = "travel:swing_after_tab"
                return "key:1"
            # Tab every ~3rd roam tick; if Tab keeps missing, turn then resume.
            if self._travel.tab_miss >= 3:
                old = self._travel.heading
                self._travel.tab_miss = 0
                self._travel.heading = {
                    "north": "east",
                    "east": "south",
                    "south": "west",
                    "west": "north",
                }.get(old, "east")
                self._travel.commit_left = 0
                turn = {
                    "north": "hold:d:0.6",
                    "east": "hold:s:0.6",
                    "south": "hold:a:0.6",
                    "west": "hold:w:0.6",
                }
                self._decision_reason = "travel:turn_after_tab_miss"
                return turn.get(old, "hold:d:0.6")
            do_tab = (self._tick % 3 == 0) or bool(obs.get("hostiles_near"))
            if do_tab and not obs.get("has_target") and random.random() < 0.8:
                self._travel.expect_target = True
                self._decision_reason = "travel:glance_tab"
                return "key:tab"
            act, reason = self._travel.action(obs)
            self._decision_reason = reason
            return act

        # 1) Stuck explore — never keep failing the same way.
        if self._stuck.is_stuck():
            act = self._stuck.recovery_action(obs)
            self._decision_reason = (
                f"stuck_recovery (x{self._stuck.same_fail_streak} after {self._stuck.last_action})"
            )
            if isinstance(planner, ScreenLLMPlanner):
                planner.last_mode = self._decision_reason
                planner.last_raw = f"(stuck) try {act} instead of {self._stuck.last_action}"
            return act

        # Ask VLM rarely — it is multi-second. Skip tick 0 cold-start stall:
        # let Q/life act first; VLM starts after vision_every.
        ask_llm = (
            self.cfg.use_ollama
            and isinstance(planner, ScreenLLMPlanner)
            and self._tick > 0
            and (
                self._stuck.same_fail_streak >= 3
                or (self._tick % max(1, self.cfg.vision_every) == 0)
                or random.random() < self.cfg.llm_mix
            )
        )

        llm_said = ""
        if ask_llm:
            llm_action = planner.plan(obs, self.directive, frame_path=frame_path)
            llm_said = (getattr(planner, "last_raw", "") or "").strip()
            self._last_llm_raw = llm_said
            if llm_action:
                self._last_llm_action = llm_action
                # If Q already knows this idea is bad, veto it.
                if self.cfg.learn and policy.value(obs, llm_action) < -0.4:
                    alt = policy.choose(obs, space)
                    self._decision_reason = (
                        f"learned_veto (LLM={llm_action} Q={policy.value(obs, llm_action):.2f} → {alt})"
                    )
                    if isinstance(planner, ScreenLLMPlanner):
                        planner.last_mode = self._decision_reason
                    return alt
                # Don't re-spam a failing action the model loves.
                if (
                    self._stuck.same_fail_streak >= 1
                    and llm_action == self._stuck.last_action
                ):
                    alt = self._stuck.recovery_action(obs)
                    self._decision_reason = f"stuck_override (LLM repeated {llm_action})"
                    if isinstance(planner, ScreenLLMPlanner):
                        planner.last_mode = self._decision_reason
                        planner.last_raw = f"LLM:{llm_said} → override:{alt}"
                    return alt
                self._decision_reason = f"vision_llm ({getattr(planner, 'last_mode', 'vision')})"
                return llm_action

        # 2) Act from learned Q (this is how learning changes behavior).
        if self.cfg.learn and self.cfg.use_learned_policy:
            goal = self._goal or parse_directive(self.directive)
            # Rare goal-aligned explore only — learning must carry the load.
            hint = goal_action_hint(goal, obs, self._tick)
            if hint and random.random() < 0.12:
                self._decision_reason = f"goal_explore ({goal.summary()})"
                if isinstance(planner, ScreenLLMPlanner):
                    planner.last_mode = self._decision_reason
                    planner.last_raw = f"(goal explore) {hint}"
                return hint
            act = policy.choose(obs, space)
            eps = self.cfg.epsilon
            if int(obs.get("stagnant") or 0) >= 4 or int(obs.get("no_damage_casts") or 0) >= 3:
                eps = min(0.85, eps + 0.35)
                if random.random() < 0.55:
                    act = random.choice(
                        ["hold:w:1.2", "hold:d:1.2", "hold:a:1.2", "key:tab", "hold:s:0.8"]
                    )
            self._decision_reason = f"learned_policy (ε={eps}) goal={goal.summary()}"
            if isinstance(planner, ScreenLLMPlanner):
                planner.last_mode = self._decision_reason
                if not getattr(planner, "last_raw", ""):
                    planner.last_raw = f"(no LLM this tick; Q chose {act})"
            return act

        # 3) Fallback heuristic aligned with directive
        goal = self._goal or parse_directive(self.directive)
        hint = goal_action_hint(goal, obs, self._tick)
        if hint:
            self._decision_reason = f"goal_heuristic ({goal.summary()})"
            return hint
        act = self._plan_owned(planner, obs, self.directive)
        self._decision_reason = "heuristic"
        return act

    def _plan_owned(self, planner, obs: dict[str, Any], directive: str | None) -> str:
        d = (directive or "").strip().lower()
        if obs.get("is_dead"):
            return "release_spirit"
        if obs.get("is_ghost"):
            # Run toward corpse / spirit healer — keep moving and interact.
            phase = self._tick % 5
            if phase in {0, 1, 2}:
                return "move_north"
            if phase == 3:
                return "move_east"
            return "interact"
        if d in {"stop", "wait"}:
            return "wait"
        if d == "logout":
            return "logout"
        if obs.get("vision_player_hp") is not None and obs["vision_player_hp"] < 0.20:
            # Ignore near-black / failed frames that under-read HP.
            if obs["vision_player_hp"] > 0.05 and not obs.get("is_dead"):
                return "move_south"  # try to disengage instead of freezing
        if "talk" in (obs.get("quest_text") or "").lower():
            return "interact"
        # Prefer attacking when a target frame is visible.
        if obs.get("has_target") and d in {"", "farm", "kill", "attack"}:
            phase = self._tick % 4
            if phase == 0:
                return "attack"
            if phase == 1:
                return "attack"
            if phase == 2:
                return "attack"
            return "target_nearest"
        if "kill" in (obs.get("quest_text") or "").lower() or d in {"farm", "kill", "attack"}:
            phase = self._tick % 6
            if phase == 0:
                return "target_nearest"
            if phase in {1, 2, 3}:
                return "attack"
            if phase == 4:
                return "move_north"
            return "move_east"
        # Default owned-game roam (do not use demo quest-log heuristic).
        phase = self._tick % 8
        if phase == 0:
            return "target_nearest"
        if phase in {1, 2}:
            return "attack"
        if phase in {3, 4}:
            return "move_north"
        if phase == 5:
            return "move_east"
        if phase == 6:
            return "attack"
        return "loot"


class _OwnedHeuristicFallback:
    """Fallback used when Ollama is down — same farm policy as owned heuristic."""

    def plan(self, obs: dict[str, Any], directive: str | None = None) -> str:
        loop = OwnedGameLoop()
        loop._tick = int(obs.get("steps") or 0)
        return loop._plan_owned(None, obs, directive)