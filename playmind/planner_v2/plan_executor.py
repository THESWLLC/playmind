"""Plan queue coordination above the Learning V2 skill runtime."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Any, Callable

from .contract import Plan, SkillStep

_EVENT_ALIASES = {
    "skill_failed": "skill_fail",
    "skill_failure": "skill_fail",
    "critical_health": "health_critical",
    "dead": "death",
    "is_dead": "death",
    "is_ghost": "ghost",
    "blocking_modal": "modal",
    "expired": "plan_expiry",
}


def _normalise_event(event: Any) -> str:
    value = str(event or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _EVENT_ALIASES.get(value, value)


def _active_triggers(triggers: Any) -> set[str]:
    if triggers is None:
        return set()
    if isinstance(triggers, str):
        return {_normalise_event(triggers)}
    if isinstance(triggers, Mapping):
        return {
            _normalise_event(name)
            for name, active in triggers.items()
            if bool(active)
        }
    if isinstance(triggers, Iterable):
        return {_normalise_event(item) for item in triggers}
    return {_normalise_event(triggers)}


class PlanExecutor:
    """Own a validated plan queue and expose skill names only."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.plan: Plan | None = None
        self._index = 0
        self._plan_started_at: float | None = None
        self._skill_started_at: float | None = None

    @property
    def index(self) -> int:
        return self._index

    @property
    def current_plan(self) -> Plan | None:
        return self.plan

    @property
    def queue(self) -> list[SkillStep]:
        if self.plan is None:
            return []
        return list(self.plan.skills[self._index :])

    @property
    def complete(self) -> bool:
        return self.plan is not None and self._index >= len(self.plan.skills)

    @property
    def remaining(self) -> int:
        if self.plan is None:
            return 0
        return max(0, len(self.plan.skills) - self._index)

    def set_plan(self, plan: Plan) -> None:
        self.plan = plan
        self._index = 0
        now = float(self.clock())
        self._plan_started_at = now
        self._skill_started_at = now

    load = set_plan

    def clear(self) -> None:
        self.plan = None
        self._index = 0
        self._plan_started_at = None
        self._skill_started_at = None

    def current_step(self) -> SkillStep | None:
        if self.plan is None or self.complete:
            return None
        return self.plan.skills[self._index]

    def next_skill(self) -> str | None:
        step = self.current_step()
        return step.name if step is not None else None

    def advance(self, success: bool = True) -> str | None:
        """Advance only after success and return the new current skill name."""
        if self.plan is None or self.complete:
            return None
        if success:
            self._index += 1
            self._skill_started_at = float(self.clock())
        return self.next_skill()

    advance_on_success = advance

    def note_skill_outcome(self, status: str) -> str | None:
        normal = str(status or "").strip().lower()
        return self.advance(success=normal == "success")

    def is_expired(self, *, now: float | None = None) -> bool:
        step = self.current_step()
        if step is None or self._skill_started_at is None:
            return False
        current = float(self.clock()) if now is None else float(now)
        return current - self._skill_started_at >= float(step.max_seconds)

    def should_replan(self, triggers: Any = None, *, now: float | None = None) -> bool:
        if self.plan is None:
            return True
        if self.complete:
            return True
        active = _active_triggers(triggers)
        if self.is_expired(now=now):
            active.add("plan_expiry")
        configured = {_normalise_event(event) for event in self.plan.replan_on}
        return bool(active & configured) or "plan_expiry" in active

    def snapshot(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict() if self.plan else None,
            "index": self._index,
            "next_skill": self.next_skill(),
            "remaining": self.remaining,
            "complete": self.complete,
            "expired": self.is_expired(),
            "plan_started_at": self._plan_started_at,
            "skill_started_at": self._skill_started_at,
        }


__all__ = ["PlanExecutor"]
