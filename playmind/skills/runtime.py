"""Skill runtime: holds active skill, steps it, handles death interrupts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from playmind.skills.base import TERMINAL_STATUSES, Skill, SkillContext, SkillStepResult


InterruptFn = Callable[[SkillContext], str | None]


def _resolve_skill(name: str) -> Skill:
    # Late import avoids package init cycles.
    from playmind.skills import get_skill

    return get_skill(name)


def _default_death_interrupt(ctx: SkillContext) -> str | None:
    """If dying/ghost while a non-death skill is active, force recovery skill name."""
    phase = str(ctx.obs.get("life_phase") or "")
    if ctx.is_ghost or phase == "ghost":
        return "ghost_runback"
    if ctx.is_dead or phase in {"dead_dialog", "confirm", "rez_picker"}:
        return "death_recovery"
    return None


@dataclass
class SkillRuntime:
    """Owns the active skill and returns low-level actions each tick."""

    active: Skill | None = None
    last_result: SkillStepResult | None = None
    interrupt_on_death: bool = True
    interrupt_fn: InterruptFn = field(default=_default_death_interrupt)
    _started_name: str | None = None
    _status: str = "idle"

    @property
    def active_name(self) -> str | None:
        return self.active.name if self.active is not None else None

    @property
    def status(self) -> str:
        """Current runtime lifecycle status."""
        return self._status

    def clear(self) -> None:
        if self.active is not None:
            # Soft cancel without requiring a context.
            self.active._failed = True
            self.active._done = True
            self.active._fail_reason = "cancelled"
            self.active._status = "cancelled"
        self.active = None
        self._started_name = None
        self._status = "idle"

    def cancel(self, ctx: SkillContext) -> None:
        """Cancel the active skill while preserving its terminal status."""
        if self.active is None:
            self._status = "idle"
            return
        self.active.cancel(ctx)
        self.last_result = SkillStepResult(
            requested_action="wait",
            reason=f"{self.active.name}:cancelled",
            status="cancelled",
            failure_evidence=["cancelled"],
        )
        self._status = "cancelled"

    def start(self, skill: Skill | str, ctx: SkillContext) -> Skill:
        if isinstance(skill, str):
            skill = _resolve_skill(skill)
        if self.active is not None:
            self.active.cancel(ctx)
        self.active = skill
        self._started_name = skill.name
        self.last_result = None
        skill.start(ctx)
        self._status = "starting"
        return skill

    def _maybe_interrupt(self, ctx: SkillContext) -> bool:
        if not self.interrupt_on_death:
            return False
        want = self.interrupt_fn(ctx)
        if not want:
            return False
        if self.active is not None and self.active.name == want:
            return False
        self.start(want, ctx)
        return True

    def _clear_completed(self, ctx: SkillContext) -> None:
        if self.active is None:
            return
        if self.active.is_complete(ctx) or self.active.status in TERMINAL_STATUSES:
            self.active = None
            self._started_name = None

    def step(self, ctx: SkillContext) -> SkillStepResult:
        """Step the active skill; auto-start wait if none. Returns step result."""
        # A terminal skill may have produced an action on its finishing tick,
        # but must never be stepped again on a later tick.
        self._clear_completed(ctx)
        self._maybe_interrupt(ctx)
        if self.active is None:
            self.start("wait", ctx)
        assert self.active is not None
        if self.active.retries_exhausted:
            self.active._mark_failed("retry_limit_exceeded")
            result = SkillStepResult(
                requested_action="wait",
                reason=f"{self.active.name}:retry_limit_exceeded",
                status="failed",
                failure_evidence=["retry_limit_exceeded"],
            )
            self.last_result = result
            self._status = result.status
            return result
        result = self.active.step(ctx)
        if self.active.retries_exhausted and result.status not in TERMINAL_STATUSES:
            self.active._mark_failed("retry_limit_exceeded")
            result = SkillStepResult(
                requested_action="wait",
                reason=f"{self.active.name}:retry_limit_exceeded",
                status="failed",
                failure_evidence=[
                    *result.failure_evidence,
                    "retry_limit_exceeded",
                ],
                debug=dict(result.debug),
            )
        self.last_result = result
        self._status = result.status
        # Custom/legacy skills may construct SkillStepResult directly instead
        # of calling Skill._result, so synchronize their lifecycle flags here.
        self.active._status = result.status
        if result.status in TERMINAL_STATUSES:
            self.active._done = True
            self.active._failed = result.status in {"failed", "timeout", "cancelled"}
        return result

    def action(self, ctx: SkillContext) -> str:
        """Convenience: step and return the requested low-level action string."""
        return self.step(ctx).requested_action

    def is_idle(self) -> bool:
        if self.active is None:
            return True
        if self.last_result and self.last_result.status in TERMINAL_STATUSES:
            return True
        return bool(self.active._done or self.active.status in TERMINAL_STATUSES)

    def snapshot(self) -> dict[str, Any]:
        return {
            "active": self.active_name,
            "started": self._started_name,
            "status": self.status,
            "last_status": self.last_result.status if self.last_result else None,
            "last_action": self.last_result.requested_action if self.last_result else None,
            "last_reason": self.last_result.reason if self.last_result else None,
        }
