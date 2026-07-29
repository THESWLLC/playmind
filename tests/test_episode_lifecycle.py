"""Episode lifecycle gates around death and resurrection."""

from __future__ import annotations

from pathlib import Path

from playmind.events import DeathConfirmed, EventType
from playmind.life_episode import EpisodeLifecycleController


def _alive(**over: object) -> dict[str, object]:
    obs: dict[str, object] = {
        "life_phase": "alive",
        "is_dead": False,
        "is_ghost": False,
        "loading": False,
        "controls_responsive": True,
        "ui_stable": True,
    }
    obs.update(over)
    return obs


def _start_gameplay(tmp_path: Path) -> tuple[EpisodeLifecycleController, str]:
    lifecycle = EpisodeLifecycleController(
        persist_dir=tmp_path,
        frames_alive_controllable=2,
    )
    lifecycle.update(_alive(), [], 1.0)
    status = lifecycle.update(_alive(), [], 2.0)
    assert status["episode_kind"] == "gameplay"
    return lifecycle, status["episode_id"]


def _death_event() -> DeathConfirmed:
    return DeathConfirmed(
        type=EventType.DEATH_CONFIRMED,
        confidence=0.99,
        evidence=["life_to_dead_transition"],
        metadata={"event_id": "death-1"},
    )


def test_death_ends_gameplay_episode(tmp_path: Path) -> None:
    lifecycle, gameplay_id = _start_gameplay(tmp_path)
    status = lifecycle.update(
        {"life_phase": "dead_dialog", "is_dead": True},
        [_death_event()],
        3.0,
    )
    assert status["episode_ended"] is True
    assert status["ended_episode_id"] == gameplay_id
    assert status["end_reason"] == "death"
    assert status["terminal"] is True


def test_death_does_not_immediately_start_resurrected_gameplay(tmp_path: Path) -> None:
    lifecycle, _ = _start_gameplay(tmp_path)
    status = lifecycle.update(
        {"life_phase": "dead_dialog", "is_dead": True},
        [_death_event()],
        3.0,
    )
    assert status["episode_kind"] == "recovery"
    assert lifecycle.current_episode is not None
    assert lifecycle.current_episode.start_reason == "death_recovery"


def test_ghost_runback_is_recovery_not_normal_gameplay(tmp_path: Path) -> None:
    lifecycle, _ = _start_gameplay(tmp_path)
    lifecycle.update(
        {"life_phase": "dead_dialog", "is_dead": True},
        [_death_event()],
        3.0,
    )
    status = lifecycle.update(
        {"life_phase": "ghost", "is_dead": False, "is_ghost": True},
        [],
        4.0,
    )
    assert status["state"] == "ghost"
    assert status["episode_kind"] == "recovery"


def test_loading_does_not_start_episode(tmp_path: Path) -> None:
    lifecycle = EpisodeLifecycleController(
        persist_dir=tmp_path,
        frames_alive_controllable=1,
    )
    status = lifecycle.update(
        _alive(life_phase="loading", loading=True),
        [],
        1.0,
    )
    assert status["state"] == "loading"
    assert status["episode_active"] is False


def test_confirmed_controllable_starts_next_episode(tmp_path: Path) -> None:
    lifecycle, gameplay_id = _start_gameplay(tmp_path)
    lifecycle.update(
        {"life_phase": "dead_dialog", "is_dead": True},
        [_death_event()],
        3.0,
    )
    lifecycle.update(
        {"life_phase": "ghost", "is_ghost": True, "is_dead": False},
        [],
        4.0,
    )
    first = lifecycle.update(_alive(), [], 5.0)
    assert first["episode_kind"] == "recovery"
    status = lifecycle.update(_alive(), [], 6.0)
    assert status["episode_kind"] == "gameplay"
    assert status["state"] == "alive_controllable_after_resurrection"
    assert status["previous_episode_id"] == gameplay_id
    assert status["death_event_id"] == "death-1"
    assert status["recovery_segment_id"]
    assert status["resurrection_ts"] == 5.0
    assert status["controllable_ts"] == 6.0
    assert status["time_death_to_rez"] == 2.0
    assert status["time_rez_to_controllable"] == 1.0
    assert status["recovery_result"] == "success"


def test_goal_completion_closes_correctly(tmp_path: Path) -> None:
    lifecycle, gameplay_id = _start_gameplay(tmp_path)
    status = lifecycle.update(_alive(goal_complete=True), [], 3.0)
    assert status["ended_episode_id"] == gameplay_id
    assert status["end_reason"] == "goal_complete"
    assert status["terminal"] is True
    assert status["truncated"] is False
    assert status["episode_active"] is False


def test_manual_reset_truncates(tmp_path: Path) -> None:
    lifecycle, gameplay_id = _start_gameplay(tmp_path)
    status = lifecycle.update(_alive(manual_reset=True), [], 3.0)
    assert status["ended_episode_id"] == gameplay_id
    assert status["end_reason"] == "manual_reset"
    assert status["terminal"] is False
    assert status["truncated"] is True
    assert status["episode_active"] is False
