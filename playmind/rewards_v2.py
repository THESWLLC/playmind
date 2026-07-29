"""Confirmed-event rewards for Learning Architecture V2 (Phase 11 defaults)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from playmind.events import Event, EventType, kill_evidence_classes

# Phase 11 default magnitudes (full-confidence).
DEFAULT_REWARDS: dict[str, float] = {
    "objective_completed": 10.0,
    "kill_confirmed": 3.0,
    "objective_progressed": 1.0,
    "skill_succeeded": 0.5,
    "skill_timeout": -1.0,
    "unrecoverable_stuck": -2.0,
    "death": -5.0,
    "time_per_second": -0.01,
}

DEFAULT_THRESHOLDS: dict[str, float] = {
    "kill_confirmed": 0.7,
}

# Events that must not drive high-level reward when speculative.
_NO_REWARD_HINTS = frozenset(
    {
        "attack_press",
        "has_target",
        "pixel_motion",
        "target_loss_alone",
        "ocr_repeat",
        "irrelevant_ui",
        "random_movement",
    }
)


@dataclass
class RewardBreakdown:
    """Per-component reward log for debugging credit assignment."""

    total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    events_applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def add(self, name: str, value: float, *, event: str | None = None) -> None:
        if abs(value) < 1e-12:
            return
        self.components[name] = self.components.get(name, 0.0) + float(value)
        self.total += float(value)
        if event:
            self.events_applied.append(event)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 6),
            "components": {k: round(v, 6) for k, v in self.components.items()},
            "events_applied": list(self.events_applied),
            "skipped": list(self.skipped),
        }


def _scaled(base: float, confidence: float) -> float:
    """Low-confidence events must not produce full reward."""
    c = max(0.0, min(1.0, float(confidence)))
    return float(base) * c


def reward_from_events(
    events: Sequence[Event] | Iterable[Event],
    dt: float,
    *,
    values: Mapping[str, float] | None = None,
    thresholds: Mapping[str, float] | None = None,
    unrecoverable_stuck: bool = False,
) -> RewardBreakdown:
    """Map confirmed events (+ elapsed time) to a logged reward breakdown.

    Skill-local shaping is intentionally absent here — only high-level task
    rewards derived from confirmed events (and the time cost).
    """
    table = {**DEFAULT_REWARDS, **(dict(values) if values else {})}
    threshold_table = {**DEFAULT_THRESHOLDS, **(dict(thresholds) if thresholds else {})}
    breakdown = RewardBreakdown()

    # Time cost always applies.
    dt = max(0.0, float(dt))
    breakdown.add("time", table["time_per_second"] * dt)

    if unrecoverable_stuck:
        breakdown.add("unrecoverable_stuck", table["unrecoverable_stuck"])

    for ev in events:
        et = ev.type if isinstance(ev.type, EventType) else EventType(str(ev.type))
        name = et.value
        conf = float(getattr(ev, "confidence", 1.0) or 0.0)
        evidence = list(getattr(ev, "evidence", []) or [])

        if et is EventType.OBJECTIVE_COMPLETED:
            breakdown.add(
                "objective_completed",
                _scaled(table["objective_completed"], conf),
                event=name,
            )
        elif et is EventType.KILL_CONFIRMED:
            if conf < threshold_table["kill_confirmed"]:
                breakdown.skipped.append("kill_below_confidence_threshold")
                continue
            if not kill_evidence_classes(evidence):
                breakdown.skipped.append("kill_missing_orthogonal_evidence")
                breakdown.skipped.append("kill_insufficient_evidence")
                continue
            breakdown.add(
                "kill_confirmed",
                _scaled(table["kill_confirmed"], conf),
                event=name,
            )
        elif et is EventType.OBJECTIVE_PROGRESSED:
            breakdown.add(
                "objective_progressed",
                _scaled(table["objective_progressed"], conf),
                event=name,
            )
        elif et in {EventType.SKILL_SUCCESS, EventType.SKILL_SUCCEEDED}:
            breakdown.add(
                "skill_succeeded",
                _scaled(table["skill_succeeded"], conf),
                event=name,
            )
        elif et in {EventType.SKILL_FAILURE, EventType.SKILL_FAILED}:
            if "timeout" in evidence or (ev.payload or {}).get("reason") == "timeout":
                breakdown.add(
                    "skill_timeout",
                    _scaled(table["skill_timeout"], conf),
                    event=name,
                )
            else:
                breakdown.skipped.append("skill_failed_no_hl_penalty")
        elif et is EventType.DEATH_CONFIRMED:
            breakdown.add("death", _scaled(table["death"], conf), event=name)
        elif et is EventType.MOVEMENT_BLOCKED:
            # Not automatically unrecoverable — only note skip unless flagged.
            breakdown.skipped.append("movement_blocked_no_default_penalty")
        else:
            # TargetAcquired, Damage*, ModalCleared, etc. — no high-level shaping.
            breakdown.skipped.append(f"no_hl_reward:{name}")

    breakdown.total = round(breakdown.total, 6)
    return breakdown
