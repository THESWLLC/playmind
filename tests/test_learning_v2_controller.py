"""Integration smoke tests for Learning Architecture V2 controller."""

from __future__ import annotations

from playmind.learning import OnlinePolicy, owned_state_key
from playmind.learning_v2_controller import LearningV2Config, LearningV2Controller


def _alive_obs(**over):
    base = {
        "player": {"x": 0, "y": 0, "hp": 0.9},
        "vision_player_hp": 0.9,
        "has_target": False,
        "in_combat": False,
        "is_dead": False,
        "is_ghost": False,
        "life_phase": "alive",
        "hostiles_near": True,
        "motion": 5.0,
        "modal_menu": False,
        "screen_ocr": "",
        "ui_hits": [],
        "stuck_hint": "none",
        "progress_stage": "explore",
    }
    base.update(over)
    return base


def test_v2_controller_chooses_masked_action() -> None:
    ctl = LearningV2Controller(LearningV2Config(enabled=True, policy_mode="scripted"))
    ctl.attach_legacy_q(OnlinePolicy(key_fn=owned_state_key))
    act = ctl.choose_action(_alive_obs(), tick=1, goal_summary="farm")
    assert isinstance(act, str) and act
    assert ctl.last_skill is not None


def test_v2_death_selects_recovery_skill() -> None:
    ctl = LearningV2Controller(LearningV2Config(enabled=True, policy_mode="hybrid"))
    ctl.attach_legacy_q(OnlinePolicy(key_fn=owned_state_key))
    obs = _alive_obs(
        is_dead=True,
        life_phase="dead_dialog",
        vision_player_hp=0.0,
        hostiles_near=False,
        screen_ocr="You are dead Return to Graveyard",
    )
    ctl.choose_action(obs, tick=2, goal_summary="farm")
    assert ctl.last_skill in {"death_recovery", "ghost_runback", "wait"}


def test_v2_note_transition_records_history() -> None:
    ctl = LearningV2Controller(
        LearningV2Config(enabled=True, policy_mode="scripted", use_rewards_v2=True)
    )
    prev = _alive_obs(has_target=True, target_hp_est=0.5)
    nxt = _alive_obs(has_target=False, target_hp_est=0.0, motion=1.0)
    r = ctl.note_transition(prev, "attack", "attack", nxt, dt=0.2)
    assert isinstance(r, float)
    assert len(ctl.history) >= 1
