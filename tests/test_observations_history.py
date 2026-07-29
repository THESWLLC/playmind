"""Unit tests for typed Observation and TemporalHistory (learning v2)."""

from __future__ import annotations

from playmind.history import HISTORY_MAXLEN, TemporalHistory
from playmind.observations import Observation, SensorValue


def test_from_legacy_missing_sensors_stay_unknown() -> None:
    obs = Observation.from_legacy_dict({"frame_id": 3, "timestamp": 1.5})
    assert obs.frame_id == 3
    assert obs.timestamp == 1.5
    assert obs.player_hp is None
    assert obs.player_hp_confidence is None
    assert obs.has_target is None
    assert obs.has_target_confidence is None
    assert obs.in_combat is None
    assert obs.is_dead is None
    assert obs.is_ghost is None
    assert obs.motion is None
    assert obs.hostiles_near is None
    assert obs.blocking_modal is None
    assert obs.life_phase == "unknown"
    assert any("has_target" in w for w in obs.sensor_warnings)
    assert any("player_hp" in w for w in obs.sensor_warnings)
    assert any("in_combat" in w for w in obs.sensor_warnings)
    # Must not invent confident False.
    legacy_out = obs.to_legacy_dict()
    assert "has_target" not in legacy_out or legacy_out.get("has_target") is None
    assert legacy_out.get("has_target") is not False


def test_from_legacy_preserves_known_values_and_roundtrip() -> None:
    raw = {
        "timestamp": 10.0,
        "frame_id": 7,
        "vision_player_hp": 0.8,
        "player_hp_confidence": 0.9,
        "target_hp_est": 0.4,
        "has_target": True,
        "has_target_confidence": 0.85,
        "in_combat": True,
        "is_dead": False,
        "is_ghost": False,
        "life_phase": "alive",
        "motion": 6.0,
        "hostiles_near": True,
        "modal_menu": False,
        "screen_ocr": "hello world",
        "ui_hits": ["Release Spirit"],
        "known_abilities": ["Fireball"],
        "stagnant": 2,
        "failed_action_streak": 1,
        "goal_kind": "kill",
        "goal_summary": "kill wolves",
        "progress_stage": "engage",
        "quest_text": "Kill 5 wolves",
    }
    obs = Observation.from_legacy_dict(raw)
    assert obs.player_hp == 0.8
    assert obs.player_hp_confidence == 0.9
    assert obs.target_hp == 0.4
    assert obs.has_target is True
    assert obs.in_combat is True
    assert obs.is_dead is False
    assert obs.life_phase == "alive"
    assert obs.motion == 6.0
    assert obs.hostiles_near is True
    assert obs.hostile_count == 1  # inferred from hostiles_near
    assert obs.blocking_modal is False
    assert obs.ocr_text == "hello world"
    assert obs.ui_detections == ["Release Spirit"]
    assert obs.known_abilities == ["Fireball"]
    assert obs.stagnation_count == 2
    assert obs.objective_text == "Kill 5 wolves"
    assert obs.goal_kind == "kill"
    assert obs.progress_stage == "engage"
    assert obs.legacy["vision_player_hp"] == 0.8

    back = obs.to_legacy_dict()
    assert back["vision_player_hp"] == 0.8
    assert back["has_target"] is True
    assert back["in_combat"] is True
    assert back["target_hp_est"] == 0.4
    assert back["modal_menu"] is False
    assert back["screen_ocr"] == "hello world"
    assert back["stagnant"] == 2
    assert back["progress_stage"] == "engage"


def test_sensor_value_helper() -> None:
    obs = Observation.from_legacy_dict(
        {"vision_player_hp": 0.5, "player_hp_confidence": 0.7, "has_target": True}
    )
    hp = obs.sensor("player_hp")
    assert isinstance(hp, SensorValue)
    assert hp.value == 0.5
    assert hp.confidence == 0.7
    assert hp.known is True
    assert obs.sensor("has_target").value is True


def test_history_bounded_and_empty_summary() -> None:
    hist = TemporalHistory()
    assert hist.summarize().current_skill_duration == 0.0
    assert hist.summarize().repeated_action_count == 0

    for i in range(HISTORY_MAXLEN + 5):
        hist.push(
            Observation.from_legacy_dict(
                {
                    "frame_id": i,
                    "vision_player_hp": 1.0 - i * 0.01,
                    "has_target": False,
                    "in_combat": False,
                    "motion": 0.0,
                    "is_dead": False,
                    "is_ghost": False,
                    "life_phase": "alive",
                }
            ),
            requested_action="key:1",
            executed_action="key:1",
            reward=0.0,
            outcome="noop",
            dt_seconds=0.5,
        )
    assert len(hist) == HISTORY_MAXLEN
    assert hist.observations[0].frame_id == 5
    assert hist.observations[-1].frame_id == HISTORY_MAXLEN + 4


def test_health_and_motion_trends() -> None:
    hist = TemporalHistory()
    for i, (hp, motion) in enumerate([(1.0, 1.0), (0.8, 3.0), (0.5, 8.0)]):
        hist.push(
            Observation(
                frame_id=i,
                player_hp=hp,
                motion=motion,
                has_target=True,
                target_hp=1.0 - i * 0.2,
                in_combat=True,
                life_phase="alive",
            ),
            executed_action="attack",
            dt_seconds=1.0,
        )
    summary = hist.summarize()
    assert summary.health_trend == -0.5  # 1.0 -> 0.5
    assert summary.motion_trend == 7.0  # 1.0 -> 8.0
    assert summary.target_health_trend == -0.4  # 1.0 -> 0.6
    assert summary.recent_damage_received_est > 0
    assert summary.recent_damage_dealt_est > 0


def test_target_and_combat_flicker_counts() -> None:
    hist = TemporalHistory()
    # Target: T F T F → 3 flickers; Combat: T T F T → 2 flickers
    sequence = [
        (True, True),
        (False, True),
        (True, False),
        (False, True),
    ]
    for i, (tgt, cbt) in enumerate(sequence):
        hist.push(
            Observation(
                frame_id=i,
                has_target=tgt,
                in_combat=cbt,
                player_hp=1.0,
                life_phase="alive",
            ),
            dt_seconds=0.2,
        )
    summary = hist.summarize()
    assert summary.target_flicker_count == 3
    assert summary.combat_flicker_count == 2


def test_flicker_ignores_unknown_none() -> None:
    hist = TemporalHistory()
    for flag in [True, None, False, True]:
        hist.push(
            Observation(has_target=flag, in_combat=flag, life_phase="alive"),
            dt_seconds=0.1,
        )
    summary = hist.summarize()
    # True→None (break), None→False (no prev), False→True (1 flicker)
    assert summary.target_flicker_count == 1
    assert summary.combat_flicker_count == 1


def test_repeated_action_and_sensor_disagreement() -> None:
    hist = TemporalHistory()
    hist.push(
        Observation(
            life_phase="alive",
            sensor_warnings=["has_target: unknown (missing)"],
            stagnation_count=1,
            motion=0.5,
        ),
        executed_action="key:tab",
        dt_seconds=1.0,
    )
    hist.push(
        Observation(
            life_phase="alive",
            sensor_warnings=["in_combat: unknown (missing)", "motion: unknown (missing)"],
            stagnation_count=2,
            motion=0.0,
        ),
        executed_action="key:1",
        dt_seconds=1.0,
    )
    hist.push(
        Observation(
            life_phase="alive",
            sensor_warnings=[],
            stagnation_count=3,
            motion=0.0,
        ),
        executed_action="key:1",
        dt_seconds=1.0,
    )
    summary = hist.summarize()
    assert summary.repeated_action_count == 2
    assert summary.recent_sensor_disagreement == 3
    assert summary.no_progress_duration == 3.0


def test_unknown_hp_does_not_fabricate_trend() -> None:
    hist = TemporalHistory()
    hist.push(Observation(player_hp=None, life_phase="unknown"), dt_seconds=1.0)
    hist.push(Observation(player_hp=None, life_phase="unknown"), dt_seconds=1.0)
    summary = hist.summarize()
    assert summary.health_trend == 0.0
    assert summary.recent_damage_received_est == 0.0
