"""Progressive learning: detect stagnation, escalate, shape rewards.

Stops "stand still mash 1 for 100 ticks" by tracking whether actions change
the world (motion, target HP drop) and forcing the next curriculum step.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


_COMBAT_PREFIXES = (
    "attack",
    "key:1",
    "key:2",
    "key:3",
    "key:4",
    "key:5",
    "ability:",
)


def _is_combat(action: str) -> bool:
    a = (action or "").lower()
    return a == "attack" or any(a.startswith(p) for p in _COMBAT_PREFIXES)


def _is_move(action: str) -> bool:
    a = (action or "").lower()
    return a.startswith("move_") or a.startswith("hold:")


def ocr_says_no_target(ocr: str) -> bool:
    low = (ocr or "").lower().replace("|", " ")
    return (
        "no target" in low
        or "have no targe" in low  # OCR truncates "target"
        or "you have no targe" in low
    )


@dataclass
class ProgressTracker:
    """Curriculum / stagnation memory for owned-game learning."""

    # explore → seek → engage → push (leave bubble) → break_loop
    stage: str = "explore"
    stagnant: int = 0
    no_damage_casts: int = 0
    motion_ok_streak: int = 0
    kills_signal: int = 0
    last_thp: float | None = None
    forced_travel_left: int = 0

    def note(
        self,
        prev: dict[str, Any],
        action: str,
        nxt: dict[str, Any],
        reward: float,
    ) -> None:
        motion = float(nxt.get("motion") or 0)
        prev_thp = float(prev.get("target_hp_est") or 0)
        next_thp = float(nxt.get("target_hp_est") or 0)
        thp_drop = (
            bool(prev.get("has_target"))
            and bool(nxt.get("has_target"))
            and prev_thp > 0
            and next_thp > 0
            and next_thp < prev_thp - 0.02
        )
        lost_target = bool(prev.get("has_target")) and not bool(nxt.get("has_target"))

        if thp_drop or (lost_target and _is_combat(action) and reward >= 0.3):
            self.kills_signal += 1
            self.no_damage_casts = 0
            self.stagnant = max(0, self.stagnant - 3)
        elif _is_combat(action) and bool(prev.get("has_target")):
            self.no_damage_casts += 1
        elif _is_move(action):
            self.no_damage_casts = max(0, self.no_damage_casts - 1)

        if motion >= 4.0:
            self.motion_ok_streak += 1
            self.stagnant = max(0, self.stagnant - 2)
        elif motion < 2.0 and (
            _is_combat(action) or (bool(prev.get("has_target")) and not thp_drop)
        ):
            self.stagnant += 1
        elif motion < 2.0 and _is_move(action):
            self.stagnant += 1
        else:
            self.stagnant = max(0, self.stagnant - 1)

        if self.forced_travel_left > 0:
            self.forced_travel_left -= 1

        self._update_stage()

    def _update_stage(self) -> None:
        if self.stagnant >= 8 or self.no_damage_casts >= 6:
            self.stage = "break_loop"
            if self.forced_travel_left <= 0:
                self.forced_travel_left = 6
        elif self.stagnant >= 4 or self.no_damage_casts >= 3:
            self.stage = "push"
        elif self.kills_signal > 0 and self.no_damage_casts == 0:
            self.stage = "engage"
        elif self.motion_ok_streak >= 2:
            self.stage = "seek"
        else:
            self.stage = "explore"

    def patch_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Annotate obs + veto sticky false targets that freeze progress."""
        out = dict(obs)
        ocr = str(out.get("screen_ocr") or "")
        if ocr_says_no_target(ocr):
            out["has_target"] = False
            out["in_combat"] = False
            out["target_hp_est"] = None
            out["adjacent_enemies"] = []
            out["target_veto"] = "ocr_no_target"
        # Casting with zero world change ⇒ treat as no target so travel can run.
        if self.no_damage_casts >= 4 or self.stage == "break_loop":
            out["has_target"] = False
            out["in_combat"] = False
            out["target_suspect"] = True
            if self.no_damage_casts >= 4:
                out["target_veto"] = out.get("target_veto") or "no_damage"
        out["progress_stage"] = self.stage
        out["stagnant"] = self.stagnant
        out["no_damage_casts"] = self.no_damage_casts
        out["forced_travel"] = self.forced_travel_left > 0
        return out

    def force_action(self, obs: dict[str, Any]) -> tuple[str, str] | None:
        """Override when the curriculum says leave / seek — not mash 1."""
        if obs.get("is_dead") or obs.get("is_ghost"):
            return None
        if obs.get("life_phase") and obs.get("life_phase") != "alive":
            return None
        if self.forced_travel_left > 0 or self.stage in {"push", "break_loop"}:
            # Escalate: Tab once, then long strafe / forward commits.
            if self.forced_travel_left % 3 == 0 and not obs.get("has_target"):
                return "key:tab", f"progress:{self.stage} tab_escape"
            act = random.choice(
                [
                    "hold:w:1.4",
                    "hold:w:1.4",
                    "hold:d:1.2",
                    "hold:a:1.2",
                    "hold:s:0.8",
                ]
            )
            return act, f"progress:{self.stage} leave_bubble→{act}"
        if self.stage == "explore" and self.motion_ok_streak < 1 and random.random() < 0.45:
            return "hold:w:1.2", "progress:explore first_motion"
        return None

    def reward_bonus(self, prev: dict[str, Any], action: str, nxt: dict[str, Any]) -> float:
        """Extra shaping so Q learns the curriculum, not idle spam."""
        bonus = 0.0
        motion = float(nxt.get("motion") or 0)
        stagnant = int(prev.get("stagnant") or self.stagnant)
        no_dmg = int(prev.get("no_damage_casts") or self.no_damage_casts)

        if _is_combat(action):
            prev_thp = float(prev.get("target_hp_est") or 0)
            next_thp = float(nxt.get("target_hp_est") or 0)
            dropped = (
                bool(prev.get("has_target"))
                and prev_thp > 0
                and next_thp > 0
                and next_thp < prev_thp - 0.02
            )
            if not dropped:
                # Progressive: each wasted cast hurts more.
                bonus -= 0.08 + 0.04 * min(12, no_dmg) + 0.03 * min(10, stagnant)
            else:
                bonus += 0.25  # real damage = curriculum progress
        if _is_move(action):
            if motion >= 4.0:
                bonus += 0.2 + (0.1 if stagnant >= 3 else 0.0)
            elif motion < 2.0:
                bonus -= 0.1
        # Milestone: broke a long idle streak by moving.
        if stagnant >= 6 and motion >= 4.0:
            bonus += 0.45
        return round(bonus, 4)
