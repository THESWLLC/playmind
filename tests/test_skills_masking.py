"""Tests for skill framework + strict action masking."""

from __future__ import annotations

from playmind.action_masking import (
    REGISTERED_ACTIONS,
    mask_actions,
    mask_skills,
    validate_action,
)
from playmind.learning import OWNED_ACTIONS
from playmind.skills import (
    default_registry,
    get_skill,
    list_skills,
)
from playmind.skills.base import SkillContext
from playmind.skills.runtime import SkillRuntime


def _alive_obs(**kwargs):
    base = {
        "life_phase": "alive",
        "is_dead": False,
        "is_ghost": False,
        "has_target": False,
        "in_combat": False,
        "vision_player_hp": 0.9,
        "player": {"hp": 0.9},
        "modal_menu": False,
        "confirm_pending": False,
        "motion": 0.0,
        "target_hp_est": None,
    }
    base.update(kwargs)
    return base


def test_registry_lists_all_skills() -> None:
    names = list_skills()
    expected = {
        "acquire_target",
        "validate_target",
        "approach_target",
        "engage_target",
        "basic_combat_rotation",
        "loot_target",
        "disengage",
        "recover_health",
        "explore",
        "unstuck",
        "clear_modal",
        "death_recovery",
        "ghost_runback",
        "interact",
        "wait",
    }
    assert expected.issubset(set(names))
    reg = default_registry()
    assert len(reg) >= 15
    sk = get_skill("acquire_target")
    assert sk.name == "acquire_target"


def test_acquire_target_preconditions_and_tab() -> None:
    skill = get_skill("acquire_target")
    no_tgt = SkillContext(obs=_alive_obs(has_target=False), now=0.0, tick=1)
    assert skill.can_start(no_tgt)
    skill.start(no_tgt)
    r = skill.step(no_tgt)
    assert r.requested_action == "key:tab"
    assert r.status == "running"

    with_tgt = SkillContext(obs=_alive_obs(has_target=True), now=0.1, tick=2)
    assert not skill.can_start(with_tgt)
    skill2 = get_skill("acquire_target")
    skill2.start(with_tgt)
    r2 = skill2.step(with_tgt)
    assert r2.status == "success"


def test_combat_rotation_requires_target() -> None:
    skill = get_skill("basic_combat_rotation")
    ctx = SkillContext(obs=_alive_obs(has_target=False), now=0.0)
    assert not skill.can_start(ctx)
    skill.start(ctx)
    r = skill.step(ctx)
    assert r.status == "failed"
    assert "no_target" in r.failure_evidence or r.status == "failed"


def test_combat_rotation_casts_with_target() -> None:
    skill = get_skill("basic_combat_rotation")
    ctx = SkillContext(obs=_alive_obs(has_target=True, in_combat=True, target_hp_est=0.8), now=0.0)
    assert skill.can_start(ctx)
    skill.start(ctx)
    r = skill.step(ctx)
    assert r.requested_action in {"key:1", "key:2", "key:3", "attack"}
    assert r.status == "running"


def test_loot_rejected_while_target_alive() -> None:
    skill = get_skill("loot_target")
    ctx = SkillContext(
        obs=_alive_obs(has_target=True, target_hp_est=0.7),
        now=0.0,
    )
    assert not skill.can_start(ctx)
    skill.start(ctx)
    r = skill.step(ctx)
    assert r.status == "failed"


def test_death_masking_while_alive() -> None:
    obs = _alive_obs()
    ok, reason = validate_action(obs, "click_label:Release Spirit")
    assert not ok
    assert "death" in reason or "alive" in reason

    ok2, reason2 = validate_action(obs, "release_spirit")
    assert not ok2

    ok3, _ = validate_action(obs, "click_label:Return to Graveyard")
    assert not ok3

    ok4, _ = validate_action(obs, "click_label:Yes")
    assert not ok4

    candidates = [
        "hold:w:1.1",
        "key:tab",
        "release_spirit",
        "click_label:Release Spirit",
        "key:1",
        "invent:Fireball=9",
        "key:9",
        "wait",
    ]
    masked = mask_actions(obs, candidates)
    assert "release_spirit" not in masked
    assert "click_label:Release Spirit" not in masked
    assert "invent:Fireball=9" not in masked
    assert "key:9" not in masked
    assert "hold:w:1.1" in masked
    assert "key:tab" in masked
    assert "wait" in masked
    # key:1 needs a target
    assert "key:1" not in masked


def test_cast_without_target_rejected() -> None:
    obs = _alive_obs(has_target=False)
    ok, reason = validate_action(obs, "key:1")
    assert not ok
    assert reason == "cast_without_target"
    ok2, _ = validate_action(obs, "attack")
    assert not ok2

    obs_t = _alive_obs(has_target=True)
    ok3, reason3 = validate_action(obs_t, "key:1")
    assert ok3 and reason3 == "ok"


def test_loot_action_masking() -> None:
    alive_tgt = _alive_obs(has_target=True, target_hp_est=0.6)
    ok, reason = validate_action(alive_tgt, "loot")
    assert not ok
    assert reason == "loot_while_target_alive"

    dead_tgt = _alive_obs(has_target=True, target_hp_est=0.0)
    ok2, _ = validate_action(dead_tgt, "loot")
    assert ok2


def test_confirm_modal_blocks_movement() -> None:
    obs = {
        "life_phase": "confirm",
        "is_dead": True,
        "is_ghost": False,
        "confirm_pending": True,
        "has_target": False,
    }
    ok, reason = validate_action(obs, "hold:w:1.1")
    assert not ok
    assert "confirm" in reason or "dead" in reason or "movement" in reason

    ok2, _ = validate_action(obs, "click_label:Yes")
    assert ok2


def test_death_recovery_skill_actions() -> None:
    skill = get_skill("death_recovery")
    ctx = SkillContext(
        obs={
            "life_phase": "dead_dialog",
            "is_dead": True,
            "is_ghost": False,
            "screen_ocr": "You are dead | Release Spirit",
            "ui_hits": ["Release Spirit"],
        },
        now=0.0,
    )
    assert skill.can_start(ctx)
    skill.start(ctx)
    r = skill.step(ctx)
    assert "release" in r.requested_action.lower() or "graveyard" in r.requested_action.lower()


def test_mask_skills_by_life_state() -> None:
    alive = _alive_obs(has_target=False)
    names = list_skills()
    masked = mask_skills(alive, names)
    assert "death_recovery" not in masked
    assert "ghost_runback" not in masked
    assert "acquire_target" in masked
    assert "basic_combat_rotation" not in masked  # needs target
    assert "wait" in masked

    dead = {
        "life_phase": "dead_dialog",
        "is_dead": True,
        "is_ghost": False,
        "has_target": False,
    }
    masked_d = mask_skills(dead, names)
    assert "death_recovery" in masked_d
    assert "acquire_target" not in masked_d
    assert "explore" not in masked_d

    ghost = {
        "life_phase": "ghost",
        "is_dead": False,
        "is_ghost": True,
        "has_target": False,
    }
    masked_g = mask_skills(ghost, names)
    assert "ghost_runback" in masked_g
    assert "death_recovery" not in masked_g


def test_none_stuck_hint_does_not_enable_unstuck() -> None:
    masked = mask_skills(_alive_obs(stuck_hint="none", stagnant=0), list_skills())
    assert "unstuck" not in masked


def test_blocking_modal_is_masked_like_modal_menu() -> None:
    obs = _alive_obs(blocking_modal=True, modal_menu=False)
    masked = mask_skills(obs, list_skills())
    assert "clear_modal" in masked
    assert set(masked).issubset({"clear_modal", "wait"})
    ok, reason = validate_action(obs, "key:esc")
    assert ok, reason


def test_runtime_interrupt_on_death() -> None:
    rt = SkillRuntime()
    ctx = SkillContext(obs=_alive_obs(has_target=False), now=0.0, tick=1)
    rt.start("explore", ctx)
    assert rt.active_name == "explore"
    r1 = rt.step(ctx)
    assert r1.requested_action.startswith("hold:") or r1.requested_action == "wait"

    dead_ctx = SkillContext(
        obs={
            "life_phase": "dead_dialog",
            "is_dead": True,
            "is_ghost": False,
            "screen_ocr": "You are dead",
            "has_target": False,
        },
        now=1.0,
        tick=2,
    )
    r2 = rt.step(dead_ctx)
    assert rt.active_name == "death_recovery"
    assert "click_label" in r2.requested_action or r2.requested_action == "release_spirit"


def test_registered_actions_covers_owned() -> None:
    for a in OWNED_ACTIONS:
        assert a in REGISTERED_ACTIONS
    ok, _ = validate_action(_alive_obs(), "totally_fake_action_xyz")
    assert not ok


def test_clear_modal_skill() -> None:
    skill = get_skill("clear_modal")
    ctx = SkillContext(
        obs=_alive_obs(modal_menu=True, screen_ocr="Options | Exit Game | Close"),
        now=0.0,
    )
    assert skill.can_start(ctx)
    skill.start(ctx)
    r = skill.step(ctx)
    assert r.requested_action in {"key:esc", "click_label:Close"}


def test_ghost_runback() -> None:
    skill = get_skill("ghost_runback")
    ctx = SkillContext(
        obs={"life_phase": "ghost", "is_ghost": True, "is_dead": False, "screen_ocr": "12 yds"},
        now=0.0,
    )
    assert skill.can_start(ctx)
    skill.start(ctx)
    r = skill.step(ctx)
    assert r.requested_action.startswith("hold:") or r.requested_action == "interact"
