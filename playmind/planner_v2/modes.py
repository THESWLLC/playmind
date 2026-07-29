"""Planner operating modes and centralized input authorization."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class PlannerMode(str, Enum):
    observe = "observe"
    shadow = "shadow"
    assist = "assist"
    hybrid = "hybrid"
    autonomous = "autonomous"
    replay = "replay"

    # Uppercase aliases follow the usual constant style while the lowercase
    # members match the external configuration contract.
    OBSERVE = observe
    SHADOW = shadow
    ASSIST = assist
    HYBRID = hybrid
    AUTONOMOUS = autonomous
    REPLAY = replay

    @classmethod
    def parse(cls, value: "PlannerMode | str") -> "PlannerMode":
        if isinstance(value, cls):
            return value
        return cls(str(value or "").strip().lower())


Mode = PlannerMode


def _flag(auth_flags: Mapping[str, Any] | Any, *names: str) -> bool:
    for name in names:
        if isinstance(auth_flags, Mapping) and name in auth_flags:
            return bool(auth_flags[name])
        if hasattr(auth_flags, name):
            return bool(getattr(auth_flags, name))
    return False


def can_send_input(
    mode: PlannerMode | str,
    auth_flags: Mapping[str, Any] | Any,
) -> bool:
    """Return whether this mode and explicit ownership flags permit input."""
    selected = PlannerMode.parse(mode)
    if selected in {
        PlannerMode.OBSERVE,
        PlannerMode.SHADOW,
        PlannerMode.REPLAY,
    }:
        return False
    owned = _flag(auth_flags, "i_own_this_game", "owns_game")
    keyboard = _flag(auth_flags, "enable_keyboard", "input_enabled")
    if not (owned and keyboard):
        return False
    if selected is PlannerMode.ASSIST:
        return _flag(
            auth_flags,
            "approved",
            "assist_approved",
            "plan_approved",
            "approval",
        )
    return selected in {PlannerMode.HYBRID, PlannerMode.AUTONOMOUS}


__all__ = ["Mode", "PlannerMode", "can_send_input"]
