"""Kill confirmation requires evidence orthogonal to target/combat loss."""

from __future__ import annotations

from playmind.events import EventType, KillConfirmed, detect_events, kill_evidence_classes
from playmind.rewards_v2 import reward_from_events


def _combat(**over: object) -> dict[str, object]:
    obs: dict[str, object] = {
        "has_target": True,
        "in_combat": True,
        "target_hp_est": 0.5,
        "vision_player_hp": 0.8,
    }
    obs.update(over)
    return obs


def _kills(prev: dict[str, object], nxt: dict[str, object]) -> list[object]:
    return [
        event
        for event in detect_events(prev, "attack", nxt)
        if event.type is EventType.KILL_CONFIRMED
    ]


def test_target_loss_and_combat_end_are_not_kill_confirmation() -> None:
    prev = _combat()
    nxt = _combat(has_target=False, in_combat=False)
    events = detect_events(prev, "attack", nxt)
    assert not [event for event in events if event.type is EventType.KILL_CONFIRMED]
    assert [event for event in events if event.type is EventType.SUSPECTED_KILL]


def test_hp_zero_is_orthogonal_kill_evidence() -> None:
    prev = _combat(target_hp_est=0.2)
    nxt = _combat(has_target=False, in_combat=False, target_hp_est=0.0)
    kills = _kills(prev, nxt)
    assert len(kills) == 1
    assert "hp_zero" in kill_evidence_classes(kills[0].evidence)


def test_loot_or_xp_is_orthogonal_kill_evidence() -> None:
    prev = _combat(screen_ocr="")
    nxt = _combat(has_target=False, in_combat=False, screen_ocr="You receive experience")
    kills = _kills(prev, nxt)
    assert len(kills) == 1
    assert "loot_or_xp" in kill_evidence_classes(kills[0].evidence)


def test_objective_and_explicit_evidence_classes() -> None:
    classes = kill_evidence_classes(
        ["objective_kill_count_increased", "explicit_kill_flag"]
    )
    assert classes == {"objective_kill", "explicit_flag"}


def test_kill_reward_requires_confidence_and_orthogonal_class() -> None:
    low_confidence = KillConfirmed(
        type=EventType.KILL_CONFIRMED,
        confidence=0.69,
        evidence=["target_hp_near_zero", "target_lost_after_combat"],
    )
    low = reward_from_events([low_confidence], 0.0)
    assert "kill_confirmed" not in low.components
    assert "kill_below_confidence_threshold" in low.skipped

    no_orthogonal = KillConfirmed(
        type=EventType.KILL_CONFIRMED,
        confidence=1.0,
        evidence=["target_lost_after_combat", "combat_ended"],
    )
    weak = reward_from_events([no_orthogonal], 0.0)
    assert "kill_confirmed" not in weak.components
    assert "kill_missing_orthogonal_evidence" in weak.skipped


def test_objective_completion_is_edge_triggered() -> None:
    complete = {"goal_complete": True}
    first = detect_events({}, "wait", complete)
    repeated = detect_events(complete, "wait", complete)
    assert [event for event in first if event.type is EventType.OBJECTIVE_COMPLETED]
    assert not [event for event in repeated if event.type is EventType.OBJECTIVE_COMPLETED]
