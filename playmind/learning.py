"""Self-learning helpers: experience log + simple online action values."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def state_key(obs: dict[str, Any]) -> str:
    """Discretize observation into a stable key for tabular learning."""
    p = obs["player"]
    enemy = "none"
    if obs.get("adjacent_enemies"):
        e = obs["adjacent_enemies"][0]
        enemy = f"{e['name']}:{int(e['hp'] * 10)}"
    return "|".join(
        [
            f"pos:{p['x']},{p['y']}",
            f"hp:{int(p['hp'] * 5)}",
            f"enemy:{enemy}",
            f"herb:{int(bool(obs.get('herb_here')))}",
            f"npc:{int(bool(obs.get('npc_here')))}",
            f"qk:{obs.get('quest_kills', 0)}/{obs.get('quest_kills_needed', 0)}",
        ]
    )


def owned_state_key(obs: dict[str, Any]) -> str:
    """Discretize owned-game vision obs (no grid coords)."""
    if obs.get("is_dead"):
        return "dead|release"
    if obs.get("is_ghost"):
        return "ghost|runback"
    hp = obs.get("vision_player_hp")
    if hp is None:
        hp = obs.get("player", {}).get("hp", 1.0)
    hp_bin = int(max(0.0, min(1.0, float(hp))) * 5)
    target = "tgt" if obs.get("has_target") else "notgt"
    combat = "cbt" if obs.get("in_combat") else "peace"
    motion = "mv" if (obs.get("motion") or 0) > 4.0 else "still"
    near = "mob" if obs.get("hostiles_near") else "clear"
    # Progressive curriculum bucket so stuck vs moving learn different Q rows.
    stg = str(obs.get("progress_stage") or "explore")
    if stg not in {"explore", "seek", "engage", "push", "break_loop"}:
        stg = "explore"
    return f"hp:{hp_bin}|{target}|{combat}|{motion}|{near}|stg:{stg}"


def reward_owned(prev: dict[str, Any], action: str, nxt: dict[str, Any]) -> float:
    """Shaped reward from vision deltas for owned-game learning."""
    if prev.get("is_dead") or prev.get("is_ghost") or (
        prev.get("life_phase") in {"dead_dialog", "confirm", "rez_picker", "ghost"}
    ):
        a = (action or "").lower()
        ocr0 = (prev.get("screen_ocr") or "").lower()
        ocr1 = (nxt.get("screen_ocr") or "").lower()
        phase0 = str(prev.get("life_phase") or "")
        phase1 = str(nxt.get("life_phase") or "")
        # Progress through death UI is the learning signal (try→measure).
        picker_cleared = ("choose where" in ocr0) and ("choose where" not in ocr1)
        sure_cleared = ("are you sure" in ocr0) and ("are you sure" not in ocr1)
        dialog_cleared = ("you are dead" in ocr0 or "return to graveyard" in ocr0) and (
            "you are dead" not in ocr1 and "choose where" not in ocr1
        )
        left_death = bool(prev.get("is_dead")) and not bool(nxt.get("is_dead") or nxt.get("is_ghost"))
        became_ghost = (not bool(prev.get("is_ghost"))) and bool(nxt.get("is_ghost"))
        order = ("dead_dialog", "confirm", "rez_picker", "ghost", "alive")
        progressed = False
        if phase0 in order and phase1 in order:
            progressed = order.index(phase1) > order.index(phase0)
        if left_death or became_ghost or picker_cleared or sure_cleared or dialog_cleared or progressed:
            return 1.15
        # Soft cost while probing — do NOT punish random tries hard (that blocks learning).
        if a.startswith("click") or "graveyard" in a or "closest" in a or a in {
            "release_spirit",
            "key:enter",
        }:
            return -0.04
        if a.startswith("hold:w") or a.startswith("hold:a") or a.startswith("hold:d"):
            return 0.05 if prev.get("is_ghost") or phase0 == "ghost" else -0.08
        return -0.12
    reward = -0.01  # small living cost
    prev_hp = float(prev.get("vision_player_hp") or prev.get("player", {}).get("hp") or 0.5)
    next_hp = float(nxt.get("vision_player_hp") or nxt.get("player", {}).get("hp") or 0.5)
    # Ignore near-black failed frames
    if prev_hp <= 0.05 or next_hp <= 0.05:
        if nxt.get("is_dead"):
            return -1.0
        return 0.0

    dhp = next_hp - prev_hp
    reward += dhp * 2.0  # stronger HP signal

    had_target = bool(prev.get("has_target"))
    has_target = bool(nxt.get("has_target"))
    modal_cleared = bool(prev.get("modal_menu")) and not bool(nxt.get("modal_menu"))
    hostiles = bool(prev.get("hostiles_near") or nxt.get("hostiles_near"))
    a = (action or "").lower()

    prev_thp = float(prev.get("target_hp_est") or 0.0)
    next_thp = float(nxt.get("target_hp_est") or 0.0)
    # Target HP drop while engaged is the real kill signal.
    if had_target and has_target and prev_thp > 0 and next_thp > 0 and next_thp < prev_thp - 0.02:
        reward += 0.55
    combatish = (
        a == "attack"
        or a.startswith("key:1")
        or a.startswith("key:2")
        or a.startswith("key:3")
        or a.startswith("ability:")
    )
    if had_target and not has_target and combatish:
        # Lost target after attack — likely kill / despawn (or out of range).
        reward += 0.4

    combat_press = (
        a == "attack"
        or a.startswith("key:1")
        or a.startswith("key:2")
        or a.startswith("key:3")
        or a.startswith("key:4")
        or a.startswith("key:5")
        or a.startswith("ability:")
        or a.startswith("bind:")
    )
    if combat_press:
        if had_target or has_target:
            # Free +0.4 taught "stand still and mash 1" with sticky false targets.
            # Only pay for real engagement: target HP drop (above) or weak presence.
            if had_target and has_target and prev_thp > 0 and next_thp > 0 and next_thp < prev_thp - 0.02:
                reward += 0.15  # already got 0.55 above; small cast bonus
            else:
                reward += 0.08
        elif hostiles:
            reward += 0.05  # swing near mobs — weak explore, not free XP
        else:
            reward -= 0.18
    elif a in {"target_nearest", "key:tab"}:
        if has_target and not had_target:
            reward += 0.45  # acquired a target
        elif has_target:
            reward -= 0.08  # retarget spam while already locked
        elif hostiles:
            reward += 0.05
        else:
            reward -= 0.06
    elif a.startswith("move_") or a.startswith("hold:w") or a.startswith("hold:a") or a.startswith(
        "hold:d"
    ) or a.startswith("hold:s"):
        motion = float(nxt.get("motion") or 0)
        if had_target and not hostiles:
            reward -= 0.05  # slight cost to walk off a real fight
        else:
            reward += 0.12  # exploring is the job when grinding
        if motion > 4.0:
            reward += 0.22  # actually left the bubble
        elif motion < 2.0:
            reward -= 0.18  # walked into a wall / didn't move — learn another way
    elif a in {"key:esc", "key:escape"} or a.startswith("click_label:close"):
        reward += 0.8 if modal_cleared else -0.35
    elif a.startswith("click"):
        # Death-UI leftovers in Q — punish hard while alive (unless closing a modal).
        if modal_cleared or prev.get("modal_menu"):
            reward += 0.15
        else:
            reward -= 0.45
    elif a == "wait":
        reward += 0.12 if next_hp < 0.35 else -0.04
    elif a == "loot":
        reward += 0.05 if had_target and not has_target else 0.0
    elif a == "interact":
        reward += 0.08
    elif a == "release_spirit" or "click_label:release" in a or "click_label:resurrect" in a:
        # Only useful when dead — rewarding it while alive taught bad habits.
        if prev.get("is_dead") or nxt.get("is_dead"):
            reward += 0.5
        else:
            reward -= 0.35
    elif "click_label:accept" in a:
        if prev.get("is_dead") or "release" in (prev.get("screen_ocr") or "").lower():
            reward += 0.35
        elif prev.get("modal_menu"):
            reward += 0.1
        else:
            reward -= 0.05

    # Low HP: prefer disengaging over standing still casting.
    if next_hp < 0.35 and not nxt.get("is_dead"):
        if a.startswith("move_") or a.startswith("hold:"):
            reward += 0.2
        elif combat_press and next_hp < 0.2:
            reward -= 0.15

    if next_hp < 0.15:
        reward -= 0.5
    if nxt.get("is_dead") and not prev.get("is_dead"):
        reward -= 1.0

    # Progressive learning: idle combat / leave-bubble milestones.
    stagnant = int(prev.get("stagnant") or 0)
    no_dmg = int(prev.get("no_damage_casts") or 0)
    if combat_press and not (
        had_target and has_target and prev_thp > 0 and next_thp > 0 and next_thp < prev_thp - 0.02
    ):
        if no_dmg >= 2 or stagnant >= 4:
            reward -= 0.12 + 0.03 * min(10, max(no_dmg, stagnant))
    if (a.startswith("move_") or a.startswith("hold:")) and float(nxt.get("motion") or 0) >= 4.0:
        if stagnant >= 5 or no_dmg >= 4:
            reward += 0.35  # escaping a freeze is high-value progress

    return round(reward, 4)


OWNED_ACTIONS = (
    "move_north",
    "move_south",
    "move_east",
    "move_west",
    "attack",
    "target_nearest",
    "loot",
    "interact",
    "release_spirit",
    "wait",
)


def _slim_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky fields before logging experience."""
    keep = {
        "player",
        "vision_player_hp",
        "vision_quest_text",
        "quest_text",
        "has_target",
        "in_combat",
        "hostiles_near",
        "motion",
        "target_hp_est",
        "is_dead",
        "is_ghost",
        "desaturated",
        "ghost_buttons",
        "screen_ocr",
        "screen_see",
        "brain_mode",
    }
    return {k: obs[k] for k in keep if k in obs}


@dataclass
class ExperienceBuffer:
    path: Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    key_fn: Callable[[dict[str, Any]], str] = state_key

    def add(
        self,
        obs: dict[str, Any],
        action: str,
        reward: float,
        next_obs: dict[str, Any],
        done: bool,
        source: str = "self",
    ) -> None:
        row = {
            "state": self.key_fn(obs),
            "obs": _slim_obs(obs),
            "action": action,
            "reward": reward,
            "next_state": self.key_fn(next_obs),
            "done": done,
            "source": source,
        }
        self.rows.append(row)

    def append_save(self, row: dict[str, Any] | None = None) -> None:
        """Append the latest row (or provided row) to disk without rewriting history."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        target = row or (self.rows[-1] if self.rows else None)
        if target is None:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(target) + "\n")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")

    def load(self) -> int:
        if not self.path.exists():
            return 0
        self.rows = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        return len(self.rows)

    def export_finetune_jsonl(self, out: Path, min_reward: float = 0.1) -> int:
        """Export successful steps as instruction pairs for later LLM fine-tunes."""
        out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with out.open("w", encoding="utf-8") as f:
            for row in self.rows:
                if row["reward"] < min_reward and row["source"] != "teacher":
                    continue
                sample = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a game agent. Reply with one action name only.",
                        },
                        {
                            "role": "user",
                            "content": json.dumps(row["obs"], sort_keys=True),
                        },
                        {"role": "assistant", "content": row["action"]},
                    ]
                }
                f.write(json.dumps(sample) + "\n")
                n += 1
        return n


@dataclass
class OnlinePolicy:
    """Epsilon-greedy tabular learner (learns alone from rewards)."""

    epsilon: float = 0.3
    alpha: float = 0.45
    gamma: float = 0.9
    key_fn: Callable[[dict[str, Any]], str] = state_key
    q: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )

    def choose(self, obs: dict[str, Any], actions: list[str]) -> str:
        key = self.key_fn(obs)
        # Include previously learned (possibly invented) actions for this state.
        acts = list(dict.fromkeys(list(actions) + list(self.q[key].keys())))
        if random.random() < self.epsilon:
            return random.choice(acts or actions)
        values = self.q[key]
        best_v = None
        best_actions: list[str] = []
        for a in acts:
            v = values.get(a, 0.0)
            if best_v is None or v > best_v:
                best_v = v
                best_actions = [a]
            elif v == best_v:
                best_actions.append(a)
        return random.choice(best_actions or acts or actions)

    def q_snapshot(self, obs: dict[str, Any], limit: int = 5) -> str:
        key = self.key_fn(obs)
        items = sorted(self.q[key].items(), key=lambda kv: kv[1], reverse=True)[:limit]
        if not items:
            return "(empty)"
        return ", ".join(f"{a}={v:.2f}" for a, v in items)

    def value(self, obs: dict[str, Any], action: str) -> float:
        return float(self.q[self.key_fn(obs)].get(action, 0.0))

    def update(
        self,
        obs: dict[str, Any],
        action: str,
        reward: float,
        next_obs: dict[str, Any],
        done: bool,
        actions: list[str],
    ) -> None:
        key = self.key_fn(obs)
        next_key = self.key_fn(next_obs)
        acts = list(dict.fromkeys(list(actions) + [action] + list(self.q[next_key].keys())))
        old = self.q[key][action]
        if done:
            target = reward
        else:
            next_best = max((self.q[next_key].get(a, 0.0) for a in acts), default=0.0)
            target = reward + self.gamma * next_best
        self.q[key][action] = old + self.alpha * (target - old)

    def replay(
        self,
        rows: list[dict[str, Any]],
        actions: list[str],
        *,
        n: int = 8,
        window: int = 80,
    ) -> int:
        """Re-learn from recent experience so each tick teaches more."""
        if not rows or n <= 0:
            return 0
        pool = rows[-window:]
        sample = pool if len(pool) <= n else random.sample(pool, n)
        done_n = 0
        for row in sample:
            obs = row.get("obs") or {}
            # Reconstruct a minimal next obs from next_state isn't available — use reward TD(0) on logged states
            action = str(row.get("action") or "")
            reward = float(row.get("reward") or 0.0)
            done = bool(row.get("done"))
            # Fake next obs with only fields needed for key_fn if slim obs missing next
            next_obs = dict(obs)
            # Apply a tiny TD using stored next_state via temporary key bump
            key = str(row.get("state") or self.key_fn(obs))
            next_key = str(row.get("next_state") or key)
            acts = list(dict.fromkeys(list(actions) + [action] + list(self.q[next_key].keys())))
            old = self.q[key][action]
            if done:
                target = reward
            else:
                next_best = max((self.q[next_key].get(a, 0.0) for a in acts), default=0.0)
                target = reward + self.gamma * next_best
            self.q[key][action] = old + self.alpha * (target - old)
            done_n += 1
        return done_n

    def teach(self, obs: dict[str, Any], action: str, boost: float = 1.0) -> None:
        """Strongly prefer a human-provided action in this state."""
        key = self.key_fn(obs)
        self.q[key][action] += boost

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: dict(v) for k, v in self.q.items()}
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.q = defaultdict(lambda: defaultdict(float))
        for k, actions in raw.items():
            for a, v in actions.items():
                self.q[k][a] = float(v)
