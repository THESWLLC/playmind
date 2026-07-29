"""Integration smoke tests for Learning Architecture V2 controller."""

from __future__ import annotations

from playmind.learning import OnlinePolicy, owned_state_key
from playmind.learning_v2_controller import LearningV2Config, LearningV2Controller
from playmind.policies.base import PolicyDecision


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
        "controls_responsive": True,
        "ui_stable": True,
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


def test_v2_from_owned_dict_validates_and_loads_bc(tmp_path) -> None:
    from playmind.models.policy_v2 import SkillPolicyV2

    ckpt = tmp_path / "bc.json"
    SkillPolicyV2(skill_names=["explore", "wait", "acquire_target"]).save(ckpt)
    cfg = LearningV2Config.from_owned_dict(
        {
            "learning_v2": {
                "enabled": True,
                "policy_mode": "bc",
                "bc_checkpoint": str(ckpt),
                "history_length": 8,
            }
        }
    )
    assert cfg.policy_mode == "behavior_clone"
    assert cfg.settings is not None
    ctl = LearningV2Controller(cfg)
    ctl.attach_legacy_q(OnlinePolicy(key_fn=owned_state_key))
    assert ctl.hybrid is not None
    primary = ctl.hybrid.primary
    assert getattr(primary, "_policy", None) is not None
    assert "bc:missing" not in str(primary.model_version)


def test_v2_applies_skill_timeout_overrides() -> None:
    from playmind.config_v2 import LearningV2Settings

    settings = LearningV2Settings(
        enabled=True,
        policy_mode="scripted",
        skill_timeouts={"explore": 1.5, "wait": 0.5},
        skill_retry_limits={"explore": 1},
    )
    ctl = LearningV2Controller(
        LearningV2Config(enabled=True, policy_mode="scripted", settings=settings)
    )
    ctl.attach_legacy_q(OnlinePolicy(key_fn=owned_state_key))
    ctl.choose_action(_alive_obs(hostiles_near=False), tick=1, goal_summary="explore")
    assert ctl.runtime.active is not None
    if ctl.runtime.active.name == "explore":
        assert ctl.runtime.active.timeout_s == 1.5
        assert ctl.runtime.active.retry_limit == 1


def test_active_skill_persists_without_polling_oscillating_policy() -> None:
    class OscillatingPolicy:
        def __init__(self) -> None:
            self.calls = 0

        def choose_skill(self, context, allowed_skills):
            self.calls += 1
            skill = "explore" if self.calls % 2 else "acquire_target"
            return PolicyDecision(
                skill=skill,
                confidence=0.9,
                reason="oscillating test policy",
                model_version="test",
                allowed_skills=list(allowed_skills),
            )

    ctl = LearningV2Controller(
        LearningV2Config(
            enabled=True,
            policy_mode="hybrid",
            minimum_commitment_seconds=10.0,
        )
    )
    policy = OscillatingPolicy()
    ctl.hybrid = policy  # type: ignore[assignment]

    ctl.choose_action(_alive_obs(hostiles_near=False), tick=1)
    first_skill = ctl.runtime.active_name
    ctl.choose_action(_alive_obs(hostiles_near=False), tick=2)

    assert first_skill == "explore"
    assert ctl.runtime.active_name == first_skill
    assert policy.calls == 1


def test_death_opens_recovery_not_resurrected_gameplay(tmp_path) -> None:
    ctl = LearningV2Controller(
        LearningV2Config(
            enabled=True,
            policy_mode="scripted",
            controllable_frames=1,
        )
    )
    ctl.ensure_episode(tmp_path)
    alive = _alive_obs()
    ctl.note_transition(alive, "wait", "wait", alive, dt=0.1)
    assert ctl.episode_mgr is not None
    assert ctl.episode_mgr.current is not None
    assert ctl.episode_mgr.current.episode_kind == "gameplay"

    dead = _alive_obs(
        is_dead=True,
        life_phase="dead_dialog",
        vision_player_hp=0.0,
        controls_responsive=False,
    )
    ctl.note_transition(alive, "wait", "wait", dead, dt=0.1)

    assert ctl.episode_mgr.current is not None
    assert ctl.episode_mgr.current.episode_kind == "recovery"
    assert ctl.episode_mgr.current.start_reason == "death_recovery"


def test_commitment_stats_appear_in_status_patch() -> None:
    ctl = LearningV2Controller(
        LearningV2Config(enabled=True, policy_mode="scripted")
    )
    ctl.choose_action(_alive_obs(hostiles_near=False), tick=1)

    status = ctl.status_patch()
    assert status["commitment_stats"]["active_skill"] == ctl.runtime.active_name
    assert status["commitment_stats"]["commitments_started"] == 1
    assert status["commitment"] == status["commitment_stats"]
