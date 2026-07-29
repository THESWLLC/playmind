"""Skill protocol: multi-tick behaviors that emit masked low-level actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


SkillStatus = Literal[
    "idle",
    "starting",
    "running",
    "success",
    "failed",
    "timeout",
    "cancelled",
    "blocked",
]
# ``Status`` remains ``str`` because external skills historically returned
# custom status strings.  SkillStatus documents the statuses owned here
# without making those existing results invalid at runtime.
Status = str
SKILL_STATUSES: frozenset[str] = frozenset(
    {
        "idle",
        "starting",
        "running",
        "success",
        "failed",
        "timeout",
        "cancelled",
        "blocked",
    }
)
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"success", "failed", "timeout", "cancelled"}
)


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
        self._status: Status = "idle"

    def reset(self) -> None:
        self._started_at = None
        self._steps = 0
        self._retries = 0
        self._done = False
        self._failed = False
        self._fail_reason = ""
        self._status = "idle"

    @property
    def status(self) -> Status:
        return self._status

    @property
    def retries_exhausted(self) -> bool:
        """Whether attempts have moved beyond the configured retry allowance."""
        return self._retries > max(0, int(self.retry_limit))

    def record_retry(self) -> bool:
        """Record one retry and return whether the retry limit was exceeded."""
        self._retries += 1
        if self.retries_exhausted:
            self._mark_failed("retry_limit_exceeded")
            return False
        return True

    def can_start(self, ctx: SkillContext) -> bool:
        return True

    def start(self, ctx: SkillContext) -> None:
        self.reset()
        self._started_at = float(ctx.now or 0.0)
        self._status = "starting"

    def is_complete(self, ctx: SkillContext) -> bool:
        return self._done or self._status in TERMINAL_STATUSES

    def has_failed(self, ctx: SkillContext) -> bool:
        return self._failed

    def cancel(self, ctx: SkillContext) -> None:
        self._failed = True
        self._fail_reason = "cancelled"
        self._done = True
        self._status = "cancelled"

    def timed_out(self, ctx: SkillContext) -> bool:
        if self._started_at is None:
            return False
        elapsed = float(ctx.now or 0.0) - self._started_at
        return elapsed >= float(self.timeout_s) or self._steps >= max(1, int(self.timeout_s * 4))

    def _mark_success(self, evidence: list[str] | None = None) -> None:
        self._done = True
        self._failed = False
        self._status = "success"
        if evidence:
            ctx_ev = evidence  # noqa: F841 — kept for callers

    def _mark_failed(self, reason: str) -> None:
        self._failed = True
        self._done = True
        self._fail_reason = reason
        self._status = "failed"

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
        failures = list(failure_evidence or [])
        if status == "blocked":
            self._retries += 1
        if self.retries_exhausted and status not in TERMINAL_STATUSES:
            self._mark_failed("retry_limit_exceeded")
            status = "failed"
            reason = f"{reason}:retry_limit_exceeded"
            failures.append("retry_limit_exceeded")
        elif status in TERMINAL_STATUSES:
            self._done = True
            self._failed = status in {"failed", "timeout", "cancelled"}
            self._status = status
            if self._failed and not self._fail_reason:
                self._fail_reason = reason
        else:
            self._status = status
        return SkillStepResult(
            requested_action=action,
            reason=reason,
            status=status,
            success_evidence=list(success_evidence or []),
            failure_evidence=failures,
            timed_out=timed_out,
            debug=dict(debug),
        )

    @abstractmethod
    def step(self, ctx: SkillContext) -> SkillStepResult:
        ...

    def allowed_actions(self) -> list[str]:
        """Low-level actions this skill may emit (for docs / masking hints)."""
        return ["wait"]
