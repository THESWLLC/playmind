"""Skill protocol: multi-tick behaviors that emit masked low-level actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


Status = str  # running | success | failed | timeout


@dataclass
class SkillStepResult:
    requested_action: str
    reason: str
    status: Status = "running"
    success_evidence: list[str] = field(default_factory=list)
    failure_evidence: list[str] = field(default_factory=list)
    timed_out: bool = False
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillContext:
    """Runtime context passed into skill lifecycle methods."""

    obs: Mapping[str, Any] | dict[str, Any] = field(default_factory=dict)
    history_summary: str = ""
    tick: int = 0
    goal: str = ""
    now: float = 0.0  # monotonic seconds
    meta: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.obs.get(key, default) if hasattr(self.obs, "get") else default

    @property
    def hp(self) -> float:
        o = self.obs
        if isinstance(o.get("player"), dict) and o["player"].get("hp") is not None:
            try:
                return float(o["player"]["hp"])
            except (TypeError, ValueError):
                pass
        try:
            return float(o.get("vision_player_hp") or 0.5)
        except (TypeError, ValueError):
            return 0.5

    @property
    def has_target(self) -> bool:
        return bool(self.obs.get("has_target"))

    @property
    def in_combat(self) -> bool:
        return bool(self.obs.get("in_combat"))

    @property
    def is_dead(self) -> bool:
        phase = str(self.obs.get("life_phase") or "")
        if phase in {"dead_dialog", "confirm", "rez_picker"}:
            return True
        return bool(self.obs.get("is_dead"))

    @property
    def is_ghost(self) -> bool:
        return bool(self.obs.get("is_ghost")) or str(self.obs.get("life_phase") or "") == "ghost"

    @property
    def alive(self) -> bool:
        return not self.is_dead and not self.is_ghost

    @property
    def modal_menu(self) -> bool:
        return bool(self.obs.get("modal_menu"))

    @property
    def confirm_pending(self) -> bool:
        return bool(self.obs.get("confirm_pending")) or str(self.obs.get("life_phase") or "") == "confirm"

    @property
    def target_hp(self) -> float | None:
        v = self.obs.get("target_hp_est")
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @property
    def ocr_blob(self) -> str:
        ocr = str(self.obs.get("screen_ocr") or "")
        hits = " ".join(str(h) for h in (self.obs.get("ui_hits") or []))
        return f"{ocr} {hits}".lower()


class Skill(ABC):
    """Multi-tick skill that requests low-level actions until success/fail/timeout."""

    name: str = "skill"
    timeout_s: float = 8.0
    retry_limit: int = 3

    def __init__(self) -> None:
        self._started_at: float | None = None
        self._steps: int = 0
        self._retries: int = 0
        self._done: bool = False
        self._failed: bool = False
        self._fail_reason: str = ""

    def reset(self) -> None:
        self._started_at = None
        self._steps = 0
        self._retries = 0
        self._done = False
        self._failed = False
        self._fail_reason = ""

    def can_start(self, ctx: SkillContext) -> bool:
        return True

    def start(self, ctx: SkillContext) -> None:
        self.reset()
        self._started_at = float(ctx.now or 0.0)

    def is_complete(self, ctx: SkillContext) -> bool:
        return self._done

    def has_failed(self, ctx: SkillContext) -> bool:
        return self._failed

    def cancel(self, ctx: SkillContext) -> None:
        self._failed = True
        self._fail_reason = "cancelled"
        self._done = True

    def timed_out(self, ctx: SkillContext) -> bool:
        if self._started_at is None:
            return False
        elapsed = float(ctx.now or 0.0) - self._started_at
        return elapsed >= float(self.timeout_s) or self._steps >= max(1, int(self.timeout_s * 4))

    def _mark_success(self, evidence: list[str] | None = None) -> None:
        self._done = True
        self._failed = False
        if evidence:
            ctx_ev = evidence  # noqa: F841 — kept for callers

    def _mark_failed(self, reason: str) -> None:
        self._failed = True
        self._done = True
        self._fail_reason = reason

    def _result(
        self,
        action: str,
        reason: str,
        *,
        status: Status = "running",
        success_evidence: list[str] | None = None,
        failure_evidence: list[str] | None = None,
        timed_out: bool = False,
        **debug: Any,
    ) -> SkillStepResult:
        return SkillStepResult(
            requested_action=action,
            reason=reason,
            status=status,
            success_evidence=list(success_evidence or []),
            failure_evidence=list(failure_evidence or []),
            timed_out=timed_out,
            debug=dict(debug),
        )

    @abstractmethod
    def step(self, ctx: SkillContext) -> SkillStepResult:
        ...

    def allowed_actions(self) -> list[str]:
        """Low-level actions this skill may emit (for docs / masking hints)."""
        return ["wait"]
