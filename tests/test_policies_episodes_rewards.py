"""Unit tests for policies, episodes, events, and rewards_v2."""

from __future__ import annotations

import json
from pathlib import Path

from playmind.episodes import EpisodeManager
from playmind.events import EventType, KillConfirmed, detect_events
from playmind.policies import HybridPolicy, ScriptedPolicy
from playmind.policies.scripted import (
    ACQUIRE_TARGET,
    BASIC_COMBAT,
    DEATH_RECOVERY,
    DEFAULT_SKILL_ORDER,
    ENGAGE_TARGET,
    GHOST_RUNBACK,
)
from playmind.rewards_v2 import reward_from_events


ALLOWED = list(DEFAULT_SKILL_ORDER)


def test_scripted_chooses_death_recovery_when_dead() -> None:
    policy = ScriptedPolicy()
    decision = policy.choose_skill(
        {"obs": {"is_dead": True, "life_phase": "dead_dialog", "vision_player_hp": 0.0}},
        ALLOWED,
    )
    assert decision.skill == DEATH_RECOVERY
    assert decision.confidence >= 0.9
    assert "dead" in decision.reason.lower()


def test_scripted_ghost_and_combat_paths() -> None:
    policy = ScriptedPolicy()
    ghost = policy.choose_skill(
        {"is_ghost": True, "life_phase": "ghost", "vision_player_hp": 0.0},
        ALLOWED,
    )
    assert ghost.skill == GHOST_RUNBACK

    engage = policy.choose_skill(
        {
            "has_target": True,
            "target_validated": True,
            "in_combat": True,
            "vision_player_hp": 0.9,
            "life_phase": "alive",
        },
        ALLOWED,
    )
    assert engage.skill in {BASIC_COMBAT, ENGAGE_TARGET}

    acquire = policy.choose_skill(
        {
            "has_target": False,
            "hostiles_near": True,
            "vision_player_hp": 0.9,
            "life_phase": "alive",
        },
        ALLOWED,
    )
    assert acquire.skill == ACQUIRE_TARGET


def test_kill_not_confirmed_on_mere_target_loss() -> None:
    prev = {
        "has_target": True,
        "in_combat": True,
        "target_hp_est": 0.55,
        "vision_player_hp": 0.8,
    }
    nxt = {
        "has_target": False,
        "in_combat": True,
        "target_hp_est": 0.55,
        "vision_player_hp": 0.8,
    }
    events = detect_events(prev, "attack", nxt)
    kinds = {e.type for e in events}
    assert EventType.KILL_CONFIRMED not in kinds


def test_kill_confirmed_with_multi_evidence() -> None:
    prev = {
        "has_target": True,
        "in_combat": True,
        "target_hp_est": 0.08,
        "quest_kills": 2,
        "vision_player_hp": 0.7,
    }
    nxt = {
        "has_target": False,
        "in_combat": False,
        "target_hp_est": 0.0,
        "quest_kills": 3,
        "vision_player_hp": 0.7,
        "screen_ocr": "You receive experience",
    }
    events = detect_events(prev, "attack", nxt)
    kills = [e for e in events if e.type is EventType.KILL_CONFIRMED]
    assert len(kills) == 1
    assert len(kills[0].evidence) >= 2


def test_episode_terminal_on_death(tmp_path: Path) -> None:
    mgr = EpisodeManager(persist_dir=tmp_path)
    ep = mgr.start("new_run")
    assert ep.done is False
    assert mgr.done is False

    finished = mgr.end("death")
    assert finished.done is True
    assert finished.terminal is True
    assert finished.truncated is False
    assert finished.end_reason == "death"
    assert finished.death_count >= 1
    assert mgr.done is True

    lines = (tmp_path / "episodes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["schema_version"] == 1
    assert row["done"] is True
    assert row["terminal"] is True
    assert row["end_reason"] == "death"


def test_reward_breakdown_components() -> None:
    events = [
        KillConfirmed(
            type=EventType.KILL_CONFIRMED,
            confidence=1.0,
            evidence=["target_hp_near_zero", "combat_ended", "objective_kill_count_increased"],
        )
    ]
    bd = reward_from_events(events, dt=2.0)
    assert "kill_confirmed" in bd.components
    assert bd.components["kill_confirmed"] == 3.0
    assert "time" in bd.components
    assert abs(bd.components["time"] - (-0.02)) < 1e-9
    assert abs(bd.total - (3.0 - 0.02)) < 1e-9
    assert "KillConfirmed" in bd.events_applied

    # Insufficient evidence → no kill reward
    weak = reward_from_events(
        [
            KillConfirmed(
                type=EventType.KILL_CONFIRMED,
                confidence=1.0,
                evidence=["target_lost_after_combat"],
            )
        ],
        dt=0.0,
    )
    assert "kill_confirmed" not in weak.components
    assert "kill_insufficient_evidence" in weak.skipped


def test_hybrid_emergency_uses_scripted() -> None:
    hybrid = HybridPolicy(confidence_threshold=0.45)
    decision = hybrid.choose_skill(
        {"obs": {"is_dead": True, "life_phase": "dead_dialog"}},
        ALLOWED,
    )
    assert decision.skill == DEATH_RECOVERY
    assert decision.debug_scores.get("emergency") == 1.0
