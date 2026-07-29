"""Tests for skill persistence, hysteresis, and emergency release."""

from __future__ import annotations

from playmind.skill_commitment import (
    SkillCommitmentTracker,
    should_reconsider_skill,
)
from playmind.skills.base import SkillStepResult


def _result(status: str = "running") -> SkillStepResult:
    return SkillStepResult(requested_action="wait", reason="test", status=status)


def test_persists_across_ticks_within_minimum() -> None:
    tracker = SkillCommitmentTracker(minimum_commitment_seconds=0.4)
    commitment = tracker.begin(
        "explore",
        now=10.0,
        tick=1,
        policy_confidence=0.6,
    )

    decision = should_reconsider_skill(
        commitment,
        _result(),
        {"now": 10.2, "tick": 2, "proposed_skill": "unstuck", "confidence": 0.99},
        None,
    )

    assert not decision.reconsider
    assert decision.reason == "minimum_commitment_active"


def test_small_confidence_change_does_not_switch() -> None:
    tracker = SkillCommitmentTracker(confidence_margin=0.15)
    tracker.begin("explore", now=1.0, policy_confidence=0.60)

    decision = tracker.should_reconsider(
        _result(),
        {"now": 2.0},
        proposed_skill="acquire_target",
        policy_confidence=0.70,
    )

    assert not decision.reconsider
    assert decision.reason.startswith("confidence_hysteresis:")
    assert tracker.stats()["prevented_switches"] == 1


def test_large_justified_confidence_switches_after_minimum() -> None:
    tracker = SkillCommitmentTracker(confidence_margin=0.15)
    tracker.begin("explore", now=1.0, policy_confidence=0.55)

    decision = tracker.should_reconsider(
        _result(),
        {"now": 1.5},
        proposed_skill="acquire_target",
        policy_confidence=0.80,
    )

    assert decision.reconsider
    assert decision.proposed_skill == "acquire_target"
    assert decision.reason.startswith("confidence_margin_met:")


def test_critical_emergency_interrupts_immediately() -> None:
    tracker = SkillCommitmentTracker(minimum_commitment_seconds=10.0)
    tracker.begin(
        "explore",
        now=4.0,
        policy_confidence=0.9,
        interruptible=False,
    )

    decision = tracker.should_reconsider(
        _result(),
        {"now": 4.01},
        {"critical_health": True},
    )

    assert decision.reconsider
    assert decision.force_interrupt
    assert "critical_health" in decision.reason
    assert tracker.stats()["interrupt_reasons"]["critical_health"] == 1


def test_completed_skill_releases_commitment_gate() -> None:
    tracker = SkillCommitmentTracker(minimum_commitment_seconds=10.0)
    tracker.begin("explore", now=2.0, policy_confidence=0.8)

    decision = tracker.should_reconsider(_result("success"), {"now": 2.01})

    assert decision.reconsider
    assert not decision.force_interrupt
    assert decision.reason == "skill_finished:success"
    released = tracker.release()
    assert released is not None and released.skill_name == "explore"
    assert tracker.active is None


def test_quick_a_to_b_to_a_is_counted_and_blocked() -> None:
    tracker = SkillCommitmentTracker(
        minimum_commitment_seconds=0.0,
        oscillation_window_seconds=2.0,
        oscillation_block_threshold=1,
    )
    tracker.begin("A", now=0.0, policy_confidence=0.4)
    tracker.begin("B", now=0.5, policy_confidence=0.4)

    decision = tracker.should_reconsider(
        _result(),
        {"now": 0.8},
        proposed_skill="A",
        policy_confidence=0.9,
    )

    assert not decision.reconsider
    assert decision.reason == "oscillation_blocked"
    stats = tracker.stats()
    assert stats["oscillation_count"] == 1
    assert stats["prevented_switches"] == 1
    assert stats["switches"] == 1
