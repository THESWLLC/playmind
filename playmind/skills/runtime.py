"""Skill runtime: holds active skill, steps it, handles death interrupts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from playmind.skills.base import Skill, SkillContext, SkillStepResult


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

    @property
    def active_name(self) -> str | None:
        return self.active.name if self.active is not None else None

    def clear(self) -> None:
        if self.active is not None:
            # Soft cancel without requiring a context.
            self.active._failed = True
            self.active._done = True
            self.active._fail_reason = "cleared"
        self.active = None
        self._started_name = None

    def start(self, skill: Skill | str, ctx: SkillContext) -> Skill:
        if isinstance(skill, str):
            skill = _resolve_skill(skill)
        if self.active is not None and self.active is not skill:
            self.active.cancel(ctx)
        self.active = skill
        self._started_name = skill.name
        skill.start(ctx)
        return skill

    def _maybe_interrupt(self, ctx: SkillContext) -> bool:
        if not self.interrupt_on_death or self.active is None:
            return False
        want = self.interrupt_fn(ctx)
        if not want:
            return False
        if self.active.name in {"death_recovery", "ghost_runback"} and self.active.name == want:
            return False
        if self.active.name == want:
            return False
        self.start(want, ctx)
        return True

    def step(self, ctx: SkillContext) -> SkillStepResult:
        """Step the active skill; auto-start wait if none. Returns step result."""
        self._maybe_interrupt(ctx)
        if self.active is None:
            self.start("wait", ctx)
        assert self.active is not None
        result = self.active.step(ctx)
        self.last_result = result
        if result.status in {"success", "failed", "timeout"} or self.active.is_complete(ctx):
            # Keep reference until caller starts next skill, but mark finished.
            if result.status == "timeout":
                self.active._failed = True
                self.active._done = True
        return result

    def action(self, ctx: SkillContext) -> str:
        """Convenience: step and return the requested low-level action string."""
        return self.step(ctx).requested_action

    def is_idle(self) -> bool:
        if self.active is None:
            return True
        if self.last_result and self.last_result.status in {"success", "failed", "timeout"}:
            return True
        return bool(self.active._done)

    def snapshot(self) -> dict[str, Any]:
        return {
            "active": self.active_name,
            "started": self._started_name,
            "last_status": self.last_result.status if self.last_result else None,
            "last_action": self.last_result.requested_action if self.last_result else None,
            "last_reason": self.last_result.reason if self.last_result else None,
        }
