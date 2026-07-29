"""Skill commitment, hysteresis, and emergency interruption policy."""

from __future__ import annotations

import time
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


FINISHED_STATUSES: frozenset[str] = frozenset(
    {"success", "failed", "timeout", "cancelled"}
)
CRITICAL_INTERRUPT_REASONS: frozenset[str] = frozenset(
    {
        "death_confirmed",
        "ghost",
        "blocking_modal",
        "critical_health",
        "severe_stuck",
        "loading",
        "lost_focus",
        "fatal_sensor_disagreement",
    }
)
_MISSING = object()


@dataclass
class SkillCommitment:
    skill_name: str
    started_at: float
    started_tick: int
    minimum_commitment_seconds: float
    maximum_commitment_seconds: float
    interruptible: bool
    interrupt_reasons: set[str]
    policy_confidence_at_start: float
    decision_reason: str


@dataclass
class ReconsiderationDecision:
    reconsider: bool
    reason: str
    force_interrupt: bool = False
    proposed_skill: str | None = None


def _normalise_reason(reason: Any) -> str:
    return str(reason or "").strip().lower().replace("-", "_").replace(" ", "_")


def _lookup(source: Any, *names: str, default: Any = None) -> Any:
    """Read direct, meta, observation, or decision fields from dicts/objects."""
    if source is None:
        return default
    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    for container_name in ("meta", "obs", "observation", "decision"):
        if isinstance(source, Mapping):
            nested = source.get(container_name)
        else:
            nested = getattr(source, container_name, None)
        if nested is None or nested is source:
            continue
        for name in names:
            if isinstance(nested, Mapping) and name in nested:
                return nested[name]
            if hasattr(nested, name):
                return getattr(nested, name)
    return default


def _float_value(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _emergency_reasons(emergency_state: Any) -> set[str]:
    reasons: set[str] = set()
    if emergency_state is None or emergency_state is False:
        return reasons
    if isinstance(emergency_state, str):
        reason = _normalise_reason(emergency_state)
        if reason:
            reasons.add(reason)
        return reasons
    if isinstance(emergency_state, Mapping):
        for key, value in emergency_state.items():
            key_name = _normalise_reason(key)
            if (
                key_name not in {
                    "active",
                    "force",
                    "reason",
                    "interrupt_reason",
                    "reasons",
                    "interrupt_reasons",
                    "active_reasons",
                    "type",
                    "state",
                }
                and isinstance(value, bool)
                and value
            ):
                reasons.add(key_name)
        for key in ("reason", "interrupt_reason", "type", "state"):
            if emergency_state.get(key):
                reasons.update(_emergency_reasons(emergency_state[key]))
        for key in ("reasons", "interrupt_reasons", "active_reasons"):
            if emergency_state.get(key):
                reasons.update(_emergency_reasons(emergency_state[key]))
        return reasons
    if isinstance(emergency_state, Sequence) or isinstance(
        emergency_state, (set, frozenset)
    ):
        for item in emergency_state:
            reasons.update(_emergency_reasons(item))
        return reasons
    for reason in CRITICAL_INTERRUPT_REASONS:
        if bool(getattr(emergency_state, reason, False)):
            reasons.add(reason)
    for name in ("reason", "interrupt_reason", "reasons"):
        value = getattr(emergency_state, name, None)
        if value:
            reasons.update(_emergency_reasons(value))
    return reasons


def _context_emergency_reasons(context: Any) -> set[str]:
    """Recognise critical state when callers put it in the context/observation."""
    reasons: set[str] = set()
    phase = _normalise_reason(_lookup(context, "life_phase", default=""))
    if bool(_lookup(context, "death_confirmed", "is_dead", default=False)):
        reasons.add("death_confirmed")
    if bool(_lookup(context, "is_ghost", default=False)) or phase == "ghost":
        reasons.add("ghost")
    if phase == "loading":
        reasons.add("loading")
    for reason in CRITICAL_INTERRUPT_REASONS - {"death_confirmed", "ghost"}:
        if bool(_lookup(context, reason, default=False)):
            reasons.add(reason)
    if (
        _lookup(
            context,
            "focused",
            "has_focus",
            "window_focused",
            default=_MISSING,
        )
        is False
    ):
        reasons.add("lost_focus")
    return reasons


def _preconditions_valid(commitment: SkillCommitment, context: Any) -> bool:
    explicit = _lookup(
        context,
        "preconditions_valid",
        "preconditions_met",
        "skill_preconditions_valid",
        "current_skill_valid",
        "can_continue_skill",
        default=_MISSING,
    )
    if explicit is not _MISSING:
        return bool(explicit)
    allowed = _lookup(context, "allowed_skills", "valid_skills", default=_MISSING)
    if allowed is not _MISSING and allowed is not None:
        try:
            return commitment.skill_name in allowed
        except TypeError:
            return True
    return True


def should_reconsider_skill(
    commitment: SkillCommitment,
    runtime_result: Any,
    context: Any,
    emergency_state: Any,
) -> ReconsiderationDecision:
    """Decide whether an active skill may be released or interrupted.

    Context may be a mapping or a ``SkillContext``-like object. Candidate
    fields understood are ``proposed_skill``/``candidate_skill`` and
    ``policy_confidence``/``confidence``.
    """
    proposed_raw = _lookup(
        context,
        "proposed_skill",
        "candidate_skill",
        "candidate",
        "selected_skill",
        "policy_skill",
        "skill",
        default=None,
    )
    proposed = str(proposed_raw) if proposed_raw not in (None, "") else None

    emergency_reasons = _emergency_reasons(emergency_state)
    emergency_reasons.update(_context_emergency_reasons(context))
    critical = sorted(emergency_reasons & CRITICAL_INTERRUPT_REASONS)
    if critical:
        reason = critical[0]
        return ReconsiderationDecision(
            True,
            f"critical_interrupt:{reason}",
            force_interrupt=True,
            proposed_skill=proposed,
        )

    allowed_interrupts = {
        _normalise_reason(reason) for reason in commitment.interrupt_reasons
    }
    allowed = sorted(emergency_reasons & allowed_interrupts)
    if allowed and commitment.interruptible:
        return ReconsiderationDecision(
            True,
            f"interrupt:{allowed[0]}",
            proposed_skill=proposed,
        )

    status_value = (
        runtime_result
        if isinstance(runtime_result, str)
        else _lookup(runtime_result, "status", default="running")
    )
    status = _normalise_reason(status_value)
    if status in FINISHED_STATUSES:
        return ReconsiderationDecision(
            True,
            f"skill_finished:{status}",
            proposed_skill=proposed,
        )

    if not _preconditions_valid(commitment, context):
        return ReconsiderationDecision(
            True,
            "preconditions_invalid",
            proposed_skill=proposed,
        )

    now = _float_value(_lookup(context, "now", default=None), None)
    if now is None:
        now = time.monotonic()
    elapsed = max(0.0, now - float(commitment.started_at))
    if elapsed >= max(0.0, float(commitment.maximum_commitment_seconds)):
        return ReconsiderationDecision(
            True,
            "maximum_commitment_exceeded",
            proposed_skill=proposed,
        )

    if proposed is None or proposed == commitment.skill_name:
        return ReconsiderationDecision(False, "continue_committed_skill", proposed_skill=proposed)
    if not commitment.interruptible:
        return ReconsiderationDecision(False, "commitment_not_interruptible", proposed_skill=proposed)
    if elapsed < max(0.0, float(commitment.minimum_commitment_seconds)):
        return ReconsiderationDecision(False, "minimum_commitment_active", proposed_skill=proposed)

    confidence = _float_value(
        _lookup(
            context,
            "proposed_skill_confidence",
            "candidate_confidence",
            "policy_confidence",
            "confidence",
            default=None,
        ),
        None,
    )
    if confidence is None:
        return ReconsiderationDecision(
            False,
            "candidate_confidence_unavailable",
            proposed_skill=proposed,
        )
    margin = max(
        0.0,
        _float_value(_lookup(context, "confidence_margin", default=0.15), 0.15)
        or 0.0,
    )
    threshold = float(commitment.policy_confidence_at_start) + margin
    if confidence < threshold:
        return ReconsiderationDecision(
            False,
            f"confidence_hysteresis:{confidence:.3f}<{threshold:.3f}",
            proposed_skill=proposed,
        )
    return ReconsiderationDecision(
        True,
        f"confidence_margin_met:{confidence:.3f}>={threshold:.3f}",
        proposed_skill=proposed,
    )


class SkillCommitmentTracker:
    """Own the current commitment and aggregate switch stability metrics."""

    def __init__(
        self,
        *,
        confidence_margin: float = 0.15,
        minimum_commitment_seconds: float = 0.4,
        maximum_commitment_seconds: float = 25.0,
        oscillation_window_seconds: float = 2.0,
        oscillation_block_threshold: int = 1,
        oscillation_limit: int | None = None,
    ) -> None:
        self.confidence_margin = max(0.0, float(confidence_margin))
        self.minimum_commitment_seconds = max(0.0, float(minimum_commitment_seconds))
        self.maximum_commitment_seconds = max(0.0, float(maximum_commitment_seconds))
        self.oscillation_window_seconds = max(0.0, float(oscillation_window_seconds))
        if oscillation_limit is not None:
            oscillation_block_threshold = oscillation_limit
        self.oscillation_block_threshold = max(0, int(oscillation_block_threshold))

        self.commitment: SkillCommitment | None = None
        self.switches = 0
        self.prevented_switches = 0
        self.oscillation_count = 0
        self.commitments_started = 0
        self.interrupt_reasons: Counter[str] = Counter()
        self._switch_history: deque[tuple[float, str]] = deque(maxlen=8)
        self._last_oscillation_marker: tuple[float, str, str] | None = None

    @property
    def active(self) -> SkillCommitment | None:
        return self.commitment

    @property
    def active_skill(self) -> str | None:
        return self.commitment.skill_name if self.commitment is not None else None

    def _would_oscillate(self, proposed_skill: str, now: float) -> bool:
        if len(self._switch_history) < 2 or self.commitment is None:
            return False
        prior_time, prior_skill = self._switch_history[-2]
        latest_time, latest_skill = self._switch_history[-1]
        if (
            latest_skill != self.commitment.skill_name
            or prior_skill != proposed_skill
            or now - latest_time > self.oscillation_window_seconds
        ):
            return False
        marker = (latest_time, latest_skill, proposed_skill)
        if marker != self._last_oscillation_marker:
            self.oscillation_count += 1
            self._last_oscillation_marker = marker
        return (
            self.oscillation_block_threshold > 0
            and self.oscillation_count >= self.oscillation_block_threshold
        )

    def begin_commitment(
        self,
        skill_name: str,
        *,
        now: float | None = None,
        started_at: float | None = None,
        tick: int = 0,
        started_tick: int | None = None,
        policy_confidence: float = 0.0,
        policy_confidence_at_start: float | None = None,
        confidence: float | None = None,
        decision_reason: str = "",
        interruptible: bool = True,
        interrupt_reasons: set[str] | None = None,
        minimum_commitment_seconds: float | None = None,
        maximum_commitment_seconds: float | None = None,
    ) -> SkillCommitment:
        """Begin tracking a skill after the runtime starts it."""
        when = started_at if started_at is not None else now
        if when is None:
            when = time.monotonic()
        start_tick = tick if started_tick is None else started_tick
        start_confidence = policy_confidence
        if confidence is not None:
            start_confidence = confidence
        if policy_confidence_at_start is not None:
            start_confidence = policy_confidence_at_start

        previous = self.commitment
        if previous is not None and previous.skill_name != str(skill_name):
            self.switches += 1
        if self._switch_history and self._switch_history[-1][1] != str(skill_name):
            if (
                len(self._switch_history) >= 2
                and self._switch_history[-2][1] == str(skill_name)
                and float(when) - self._switch_history[-1][0]
                <= self.oscillation_window_seconds
            ):
                marker = (
                    self._switch_history[-1][0],
                    self._switch_history[-1][1],
                    str(skill_name),
                )
                if marker != self._last_oscillation_marker:
                    self.oscillation_count += 1
                    self._last_oscillation_marker = marker
            self._switch_history.append((float(when), str(skill_name)))
        elif not self._switch_history:
            self._switch_history.append((float(when), str(skill_name)))

        minimum = (
            self.minimum_commitment_seconds
            if minimum_commitment_seconds is None
            else max(0.0, float(minimum_commitment_seconds))
        )
        maximum = (
            self.maximum_commitment_seconds
            if maximum_commitment_seconds is None
            else max(0.0, float(maximum_commitment_seconds))
        )
        self.commitment = SkillCommitment(
            skill_name=str(skill_name),
            started_at=float(when),
            started_tick=int(start_tick),
            minimum_commitment_seconds=minimum,
            maximum_commitment_seconds=maximum,
            interruptible=bool(interruptible),
            interrupt_reasons={
                _normalise_reason(reason) for reason in (interrupt_reasons or set())
            },
            policy_confidence_at_start=float(start_confidence),
            decision_reason=str(decision_reason),
        )
        self.commitments_started += 1
        return self.commitment

    begin = begin_commitment
    start = begin_commitment

    def should_reconsider(
        self,
        runtime_result: Any,
        context: Any = None,
        emergency_state: Any = None,
        *,
        proposed_skill: str | None = None,
        candidate_skill: str | None = None,
        policy_confidence: float | None = None,
        confidence: float | None = None,
    ) -> ReconsiderationDecision:
        proposal = proposed_skill if proposed_skill is not None else candidate_skill
        proposal_confidence = (
            policy_confidence if policy_confidence is not None else confidence
        )
        if self.commitment is None:
            return ReconsiderationDecision(
                True,
                "no_active_commitment",
                proposed_skill=proposal,
            )

        effective: dict[str, Any] = {}
        if isinstance(context, Mapping):
            effective.update(context)
        elif context is not None:
            for name in (
                "now",
                "tick",
                "preconditions_valid",
                "skill_preconditions_valid",
                "allowed_skills",
                "valid_skills",
                "proposed_skill",
                "candidate_skill",
                "policy_confidence",
                "confidence",
            ):
                value = _lookup(context, name, default=_MISSING)
                if value is not _MISSING:
                    effective[name] = value
            for name in ("obs", "meta", "decision"):
                value = getattr(context, name, None)
                if value is not None:
                    effective[name] = value
        effective["confidence_margin"] = self.confidence_margin
        if proposal is not None:
            effective["proposed_skill"] = proposal
        if proposal_confidence is not None:
            effective["policy_confidence"] = proposal_confidence

        decision = should_reconsider_skill(
            self.commitment,
            runtime_result,
            effective,
            emergency_state,
        )
        emergency_reasons = _emergency_reasons(emergency_state)
        emergency_reasons.update(_context_emergency_reasons(effective))
        if decision.force_interrupt:
            for reason in sorted(emergency_reasons & CRITICAL_INTERRUPT_REASONS):
                self.interrupt_reasons[reason] += 1
            return decision
        if decision.reason.startswith("interrupt:"):
            self.interrupt_reasons[decision.reason.partition(":")[2]] += 1
            return decision

        candidate = decision.proposed_skill
        is_switch_attempt = bool(candidate and candidate != self.commitment.skill_name)
        if (
            decision.reconsider
            and is_switch_attempt
            and decision.reason.startswith("confidence_margin_met:")
        ):
            now = _float_value(_lookup(effective, "now", default=None), None)
            if now is None:
                now = time.monotonic()
            if self._would_oscillate(str(candidate), now):
                self.prevented_switches += 1
                return ReconsiderationDecision(
                    False,
                    "oscillation_blocked",
                    proposed_skill=str(candidate),
                )
        elif not decision.reconsider and is_switch_attempt:
            self.prevented_switches += 1
        return decision

    consider = should_reconsider

    def release(self) -> SkillCommitment | None:
        previous = self.commitment
        self.commitment = None
        return previous

    clear = release

    def stats(self) -> dict[str, Any]:
        return {
            "active_skill": self.active_skill,
            "commitments_started": self.commitments_started,
            "switches": self.switches,
            "prevented_switches": self.prevented_switches,
            "interrupt_reasons": dict(sorted(self.interrupt_reasons.items())),
            "oscillation_count": self.oscillation_count,
            "confidence_margin": self.confidence_margin,
            "minimum_commitment_seconds": self.minimum_commitment_seconds,
            "maximum_commitment_seconds": self.maximum_commitment_seconds,
        }


__all__ = [
    "CRITICAL_INTERRUPT_REASONS",
    "FINISHED_STATUSES",
    "ReconsiderationDecision",
    "SkillCommitment",
    "SkillCommitmentTracker",
    "should_reconsider_skill",
]
