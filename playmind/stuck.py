"""Stuck / no-effect detection: stop spamming dead actions and recover."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


_MODAL_MARKERS = (
    "options",
    "key bindings",
    "keybindings",
    "video",
    "sound",
    "interface",
    "macros",
    "addons",
    "add ons",
    "logout",
    "exit game",
    "join discord",
    "report bug",
    "quick binding",
)


@dataclass
class StuckTracker:
    """If the same action keeps producing no change, escalate recovery."""

    window: int = 8
    fail_before_escape: int = 2
    history: deque = field(default_factory=lambda: deque(maxlen=16))
    last_action: str | None = None
    last_state: str | None = None
    last_reward: float = 0.0
    same_fail_streak: int = 0
    recovery_step: int = 0

    def note_outcome(
        self,
        action: str,
        state: str,
        reward: float,
        *,
        motion: float = 0.0,
        modal: bool = False,
    ) -> None:
        # Core learning signal: same action + non-positive reward ⇒ failing / stuck.
        failing = reward <= 0.05
        if action == self.last_action and failing:
            self.same_fail_streak += 1
        else:
            self.same_fail_streak = 0
            self.recovery_step = 0
        self.last_action = action
        self.last_state = state
        self.last_reward = reward
        self.history.append(
            {
                "action": action,
                "state": state,
                "reward": reward,
                "no_effect": failing and motion < 2.0,
                "streak": self.same_fail_streak,
            }
        )

    def is_stuck(self) -> bool:
        return self.same_fail_streak >= self.fail_before_escape

    def recovery_action(self, obs: dict[str, Any]) -> str:
        """When stuck: try alternatives — never re-pick the failing action."""
        self.recovery_step += 1
        step = self.recovery_step
        hits = [str(h) for h in (obs.get("ui_hits") or [])]
        banned = (self.last_action or "").lower()

        from playmind.vision import is_world_mob_label

        candidates: list[str] = []
        # Prefer combat when a target/hostile is already known.
        if obs.get("has_target") or obs.get("in_combat"):
            candidates.extend(("key:1", "attack", "key:2", "hold:w:0.5", "key:tab"))
        ui_hits = [h for h in hits if not is_world_mob_label(h)]
        if ui_hits:
            candidates.append(f"click_label:{ui_hits[0]}")
        candidates.extend(
            (
                "key:esc",
                "key:1",
                "key:2",
                "key:tab",
                "hold:w:0.6",
                "move_south",
                "move_east",
                "interact",
                "key:3",
                "click_label:Close",
                "click_label:Accept",
                "key:4",
                "hold:d:0.5",
            )
        )
        # Filter out the action that is already failing + never click world nameplates
        filtered: list[str] = []
        for c in candidates:
            if c.lower() == banned:
                continue
            if c.lower().startswith("click_label:"):
                label = c.split(":", 1)[1]
                if is_world_mob_label(label):
                    continue
            filtered.append(c)
        candidates = filtered
        if not candidates:
            return "key:1"
        return candidates[(step - 1) % len(candidates)]


def detect_blocking_modal(obs: dict[str, Any]) -> bool:
    """True if a full-screen Options / settings style menu is up."""
    ocr = (obs.get("screen_ocr") or "").lower()
    hits = [str(h).lower() for h in (obs.get("ui_hits") or [])]
    blob = ocr + " " + " ".join(hits)
    if "close" in blob and any(m in blob for m in _MODAL_MARKERS):
        return True
    # Strong single markers often mean the Options panel.
    strong = ("options", "key bindings", "exit game", "join discord")
    hits_strong = sum(1 for m in strong if m in blob)
    return hits_strong >= 2


def modal_close_action(obs: dict[str, Any]) -> str:
    ocr = (obs.get("screen_ocr") or "").lower()
    hits = " ".join(str(h) for h in (obs.get("ui_hits") or [])).lower()
    if "close" in ocr or "close" in hits:
        # Alternate esc / click so one of them lands.
        return "click_label:Close" if (obs.get("steps") or 0) % 2 else "key:esc"
    return "key:esc"
