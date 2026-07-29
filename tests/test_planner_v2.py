from __future__ import annotations

import json

import pytest

from playmind.planner_v2.contract import Plan, PlannerState, SkillStep
from playmind.planner_v2.model_registry import ModelRegistry
from playmind.planner_v2.modes import PlannerMode, can_send_input
from playmind.planner_v2.ollama_client import OllamaClient
from playmind.planner_v2.plan_executor import PlanExecutor
from playmind.planner_v2.plan_validator import validate_or_parse
from playmind.planner_v2.runtime import PlannerV2Runtime
from playmind.planner_v2.state_builder import build_planner_state


SKILLS = ["explore", "wait", "death_recovery", "basic_combat_rotation"]


def _plan_json(skill: str = "explore") -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "goal": "find a quest",
            "skills": [
                {
                    "name": skill,
                    "until": None,
                    "max_seconds": 30,
                    "constraints": {},
                }
            ],
            "replan_on": ["death", "health_critical", "plan_expiry"],
            "confidence": 0.8,
            "reason_code": "explore",
            "summary": "Explore safely",
        }
    )


def test_state_serialization_preserves_unknown_vs_false() -> None:
    unknown = build_planner_state(
        {},
        goal="test",
        profile={},
        available_skills=SKILLS,
        current_skill=None,
        recent_skills=[],
        previous_plan=None,
        memory=[],
        game_id="game",
    )
    known_false = build_planner_state(
        {"has_target": False, "has_target_confidence": 0.9},
        goal="test",
        profile={},
        available_skills=SKILLS,
        current_skill=None,
        recent_skills=[],
        previous_plan=None,
        memory=[],
        game_id="game",
    )

    assert unknown.sensors["has_target"]["known"] is False
    assert known_false.sensors["has_target"] == {
        "value": False,
        "known": True,
        "confidence": 0.9,
    }
    restored = PlannerState.from_json(known_false.to_json())
    assert restored.sensors["has_target"]["value"] is False
    assert restored.sensors["has_target"]["known"] is True


def test_plan_parse_and_validate() -> None:
    result = validate_or_parse(_plan_json(), SKILLS)
    assert result.ok
    assert result.plan is not None
    assert result.plan.skills[0].name == "explore"


def test_illegal_skill_is_rejected() -> None:
    invented = validate_or_parse(_plan_json("teleport_to_boss"), SKILLS)
    unavailable = validate_or_parse(_plan_json("acquire_target"), SKILLS)
    assert not invented.ok
    assert "invented skill" in " ".join(invented.errors)
    assert not unavailable.ok
    assert "unavailable skill" in " ".join(unavailable.errors)


def test_ollama_timeout_is_propagated_with_configured_timeout() -> None:
    calls: list[float] = []

    def opener(request, timeout):
        calls.append(timeout)
        raise TimeoutError("slow model")

    client = OllamaClient(timeout=0.25, opener=opener)
    with pytest.raises(TimeoutError):
        client.generate_plan(PlannerState(goal="x"), "model")
    assert calls == [0.25]


def test_runtime_uses_scripted_fallback_after_validation_failure() -> None:
    runtime = PlannerV2Runtime(
        {"mode": "shadow"},
        planner=lambda state: "not json",
    )
    plan = runtime.request_plan(
        {"has_target": False, "life_phase": "alive"},
        goal="explore",
        available_skills=["explore", "wait"],
    )
    assert plan is not None
    assert plan.reason_code == "heuristic_fallback"
    assert plan.skills[0].name in {"explore", "wait"}
    assert runtime.last_plan_source == "heuristic_fallback"


def test_plan_executor_replan_triggers_and_expiry() -> None:
    now = [10.0]
    executor = PlanExecutor(clock=lambda: now[0])
    executor.set_plan(
        Plan(
            goal="x",
            skills=[SkillStep("explore", max_seconds=3)],
            replan_on=["death"],
        )
    )
    assert not executor.should_replan(["modal"])
    assert executor.should_replan(["death"])
    now[0] = 13.0
    assert executor.should_replan([])


def test_shadow_never_dispatches_input() -> None:
    dispatched: list[str] = []
    runtime = PlannerV2Runtime(
        {
            "mode": "shadow",
            "i_own_this_game": True,
            "enable_keyboard": True,
        },
        planner=lambda state: _plan_json(),
        execute_skill=dispatched.append,
    )
    runtime.request_plan({}, goal="x", available_skills=SKILLS)
    assert runtime.execute_current() is None
    assert dispatched == []


def _metrics(score: float, scenarios: int = 200) -> dict[str, float | int]:
    return {
        "valid_plan_rate": 1.0,
        "illegal_skill_rate": 0.0,
        "scenario_count": scenarios,
        "benchmark_score": score,
    }


def test_registry_promotion_gates_override_and_rollback(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "registry.sqlite")
    registry.register(
        "production-v1",
        eval_metrics=_metrics(0.7),
        status="candidate",
    )
    registry.promote("production-v1")
    registry.register(
        "bad",
        eval_metrics=_metrics(0.8, scenarios=2),
        status="candidate",
    )
    with pytest.raises(ValueError, match="promotion gates failed"):
        registry.promote("bad")
    promoted = registry.promote("bad", manual_override=True, reason="operator override")
    assert promoted["warning"] is True
    assert registry.audit_log(model_id="bad")[0]["warning"] is True

    rolled_back = registry.rollback()
    assert rolled_back["model_id"] == "production-v1"
    assert registry.get_production()["model_id"] == "production-v1"


@pytest.mark.parametrize(
    ("mode", "flags", "expected"),
    [
        (PlannerMode.OBSERVE, {"i_own_this_game": True, "enable_keyboard": True}, False),
        (PlannerMode.SHADOW, {"i_own_this_game": True, "enable_keyboard": True}, False),
        (PlannerMode.REPLAY, {"i_own_this_game": True, "enable_keyboard": True}, False),
        (PlannerMode.HYBRID, {"i_own_this_game": True, "enable_keyboard": True}, True),
        (PlannerMode.AUTONOMOUS, {"i_own_this_game": True, "enable_keyboard": False}, False),
        (
            PlannerMode.ASSIST,
            {
                "i_own_this_game": True,
                "enable_keyboard": True,
                "plan_approved": True,
            },
            True,
        ),
    ],
)
def test_modes_can_send_input(mode, flags, expected) -> None:
    assert can_send_input(mode, flags) is expected
