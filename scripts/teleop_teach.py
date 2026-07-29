"""Hands-on teleop: drive Ascension briefly, log what works, teach Q + lessons.

Elevated: keys only reach Ascension when this process is admin.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from playmind.actuators import OwnedGameKeyboardActuator, load_keymap
from playmind.ability_memory import AbilityMemory, ensure_ability_seeded
from playmind.capture import capture_config_from_dict
from playmind.council import Lesson, TeacherBrain
from playmind.learning import ExperienceBuffer, OnlinePolicy, owned_state_key, reward_owned
from playmind.owned_loop import load_owned_config, vision_obs_from_frame
from playmind.screen_llm import enrich_obs_from_screen
from playmind.ui_memory import UIMemory, ensure_seeded
from playmind.vision import detect_hostile_nameplate_ocr, detect_target_bar

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "playmind" / "owned"
CFG = ROOT / "config" / "owned_game.json"


def observe(path: Path, rois: dict, prev: Path | None = None) -> dict:
    obs = vision_obs_from_frame(path, rois, prev_frame=prev, steps=0)
    obs = enrich_obs_from_screen(path, obs)
    ocr = f"{obs.get('screen_ocr') or ''} {obs.get('quest_text') or ''}"
    if detect_hostile_nameplate_ocr(ocr):
        obs["hostiles_near"] = True
    if obs.get("is_dead"):
        obs["has_target"] = False
        obs["in_combat"] = False
    return obs


def summarize(obs: dict) -> str:
    return (
        f"dead={obs.get('is_dead')} ghost={obs.get('is_ghost')} "
        f"tgt={obs.get('has_target')} hostiles={obs.get('hostiles_near')} "
        f"hp={obs.get('vision_player_hp')} modal={obs.get('modal_menu')} "
        f"ocr={(obs.get('screen_ocr') or '')[:140]!r}"
    )


def main() -> None:
    owned = load_owned_config(CFG)
    rois = owned.get("rois", {})
    keymap = load_keymap(Path(owned.get("keymap_path", "config/keymap.example.json")))
    ui = UIMemory(DATA / "ui_memory.json")
    ensure_seeded(ui)
    abil = AbilityMemory(DATA / "ability_memory.json")
    ensure_ability_seeded(abil)

    act = OwnedGameKeyboardActuator(
        keymap=keymap,
        enabled=True,
        i_own_this_game=True,
        window_title_substr=str(owned.get("capture", {}).get("window_title") or "Ascension"),
        ui_memory=ui,
        ability_memory=abil,
        log_path=ROOT / "data" / "playmind" / "teleop_actuator.jsonl",
    )

    policy = OnlinePolicy(epsilon=0.0, key_fn=owned_state_key)
    policy_path = DATA / "policy.json"
    if policy_path.exists():
        policy.load(policy_path)
    buf = ExperienceBuffer(DATA / "experience.jsonl", key_fn=owned_state_key)
    teacher = TeacherBrain(path=DATA / "lessons.jsonl")

    frame = DATA / "teleop_latest.png"
    prev = DATA / "teleop_prev.png"
    cap_cfg = owned.get("capture", {"mode": "window", "window_title": "Ascension"})

    def snap(dest: Path):
        return capture_config_from_dict(cap_cfg, dest)

    print("=== TELEOP TEACH start — focusing Ascension ===")
    snap(frame)
    obs = observe(frame, rois)
    print("START:", summarize(obs))

    # Curriculum: close UI → leave death/ghost → find mob → tab → attack
    plan: list[tuple[str, str]] = []
    ocr = (obs.get("screen_ocr") or "").lower()
    hits = " ".join(str(h) for h in (obs.get("ui_hits") or [])).lower()
    blob = ocr + " " + hits

    if "macro" in blob or "create macros" in blob:
        plan.append(("key:esc", "close macros/modal overlay"))
        plan.append(("key:esc", "second esc if nested"))

    if obs.get("is_dead") or obs.get("is_ghost") or "graveyard" in blob or "safe zone" in blob:
        # Prefer explicit resurrect/return buttons over vague release click spray
        if "safe zone" in blob or "resurrect" in blob:
            plan.append(("click_label:Resurrect in a Safe Zone", "resurrect at safe zone"))
        if "return to" in blob or "graveyard" in blob:
            plan.append(("click_label:Return to Graveyard", "return to graveyard button"))
        plan.append(("release_spirit", "fallback multi-click death UI"))
        plan.append(("key:esc", "dismiss leftover dialogs"))

    # Always practice a short farm cycle after recovery attempts
    plan.extend(
        [
            ("hold:w:1.2", "walk forward to find mobs"),
            ("key:tab", "acquire nearest target"),
            ("key:1", "primary attack"),
            ("key:1", "attack again"),
            ("key:2", "secondary ability"),
            ("key:1", "finish with primary"),
            ("hold:w:0.8", "close distance if casting out of range"),
            ("key:1", "attack after gap-close"),
            ("key:tab", "retarget if needed"),
            ("key:1", "attack"),
            ("key:1", "attack"),
            ("key:1", "attack"),
        ]
    )

    taught: list[dict] = []
    for i, (action, why) in enumerate(plan, 1):
        if frame.exists():
            frame.replace(prev)
        print(f"\n[{i}/{len(plan)}] DO {action!r} — {why}")
        print("  before:", summarize(obs))
        act.send(action)
        time.sleep(0.85 if action.startswith("hold:") else 0.55)
        snap(frame)
        nxt = observe(frame, rois, prev=prev if prev.exists() else None)

        r = reward_owned(obs, action, nxt)
        # Extra human credit for transitions we care about
        if obs.get("is_dead") and not nxt.get("is_dead"):
            r += 1.5
            why_ok = "LEFT DEATH UI"
        elif obs.get("is_ghost") and not nxt.get("is_ghost") and not nxt.get("is_dead"):
            r += 1.0
            why_ok = "LEFT GHOST"
        elif (not obs.get("has_target")) and nxt.get("has_target"):
            r += 0.8
            why_ok = "ACQUIRED TARGET"
        elif obs.get("has_target") and action in {"key:1", "attack", "key:2"}:
            r += 0.5
            why_ok = "ATTACK WHILE TARGETED"
        else:
            why_ok = "step"

        print(f"  after: {summarize(nxt)}  reward={r:.2f}  ({why_ok})")

        # Teach Q the successful / intended transitions
        if r > 0.05 or why_ok != "step":
            policy.teach(obs, action, boost=max(0.8, min(2.0, r)))
            lesson = Lesson(
                t=time.time(),
                goal="farm",
                bad_action="(teleop prior)",
                reward=r,
                better=action,
                reason=f"human teleop: {why} → {why_ok}",
                ocr=(obs.get("screen_ocr") or "")[:160],
            )
            teacher.lessons.append(lesson)
            teacher._save_lesson(lesson)
            taught.append({"action": action, "why": why, "signal": why_ok, "reward": r})

        buf.add(obs, action, round(r, 4), nxt, False, source="teleop")
        buf.append_save()
        obs = nxt

        # Early stop if we somehow fully recovered mid-plan for death section
        # (continue farm cycle anyway)

    policy.save(policy_path)
    # Persist lessons via TeacherBrain write path
    teacher.path.parent.mkdir(parents=True, exist_ok=True)
    with teacher.path.open("a", encoding="utf-8") as f:
        for row in taught:
            f.write(
                json.dumps(
                    {
                        "source": "teleop",
                        "action": row["action"],
                        "why": row["why"],
                        "signal": row["signal"],
                        "reward": row["reward"],
                    }
                )
                + "\n"
            )

    print("\n=== TELEOP DONE ===")
    print("Final:", summarize(obs))
    print(f"Taught {len(taught)} steps into Q + lessons")
    print("Target detect now:", detect_target_bar(frame))
    for row in taught:
        print(f"  teach {row['action']}: {row['signal']} r={row['reward']:.2f} ({row['why']})")


if __name__ == "__main__":
    main()
