"""Outcome-oriented offline replay regression tests."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from playmind.evaluation.metrics import summarize_replay_results
from playmind.evaluation.scenarios import make_baseline_policies
from playmind.policies.base import PolicyDecision
from playmind.replay_env import ReplayEnv, compare_policies
from playmind.skills.runtime import SkillRuntime


class SequencePolicy:
    def __init__(self, skills: Sequence[str], *, fallback: bool = False) -> None:
        self.skills = list(skills)
        self.fallback = fallback
        self.index = 0

    def reset_state(self) -> None:
        self.index = 0

    def choose_skill(
        self, context: Mapping[str, Any], allowed_skills: Sequence[str]
    ) -> PolicyDecision:
        skill = self.skills[min(self.index, len(self.skills) - 1)]
        self.index += 1
        return PolicyDecision(
            skill=skill,
            confidence=0.9,
            reason="scripted fallback" if self.fallback else "test",
            allowed_skills=list(allowed_skills),
            used_fallback=self.fallback,
        )


def _samples(labels: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": float(index),
            "skill": label,
            "allowed_skills": ["a", "b"],
            "observation": {"objective_progress": index / 10.0},
            "key_events": ["confirmed_kill"] if index == len(labels) - 1 else [],
        }
        for index, label in enumerate(labels)
    ]


def test_report_separates_observed_and_counterfactual() -> None:
    results = ReplayEnv.from_samples(
        _samples(["a", "b"]), policy=SequencePolicy(["a", "a"])
    ).run()
    report = summarize_replay_results(results)
    assert report["observed_outcomes"]["confirmed_kill_count"] == 1
    assert report["counterfactual_estimates"]["is_confirmed"] is False
    assert "not_confirmed" in report["counterfactual_estimates"]["status"]
    assert "confirmed_kill_count" not in report["counterfactual_estimates"]


def test_baseline_comparison_works() -> None:
    policies = make_baseline_policies()
    compared = compare_policies(_samples(["a", "b"]), policies)
    assert {"scripted", "legacy_stub", "hybrid", "human_demo"} <= set(compared)
    assert compared["human_demo"]["label_agreement"]["accuracy"] == 1.0


def test_skill_switch_metrics() -> None:
    results = ReplayEnv.from_samples(
        _samples(["a", "a", "b", "b", "a"]),
        policy=SequencePolicy(["a", "a", "b", "b", "a"]),
    ).run()
    temporal = summarize_replay_results(results)["temporal"]
    assert temporal["skill_switch_count"] == 2
    assert temporal["skill_switch_rate"] == pytest.approx(0.5)
    assert temporal["repeated_action_rate"] == pytest.approx(0.5)


def test_scripted_fallback_and_invalid_proposal_counted() -> None:
    fallback = ReplayEnv.from_samples(
        _samples(["a"]), policy=SequencePolicy(["a"], fallback=True)
    ).run()
    assert (
        summarize_replay_results(fallback)["decision_validity"][
            "scripted_fallback_count"
        ]
        == 1
    )

    invalid = ReplayEnv.from_samples(
        _samples(["a"]), policy=SequencePolicy(["not-valid"])
    ).run()
    validity = summarize_replay_results(invalid)["decision_validity"]
    assert validity["invalid_skill_proposal_count"] == 1
    assert validity["invalid_skill_proposal_rate"] == 1.0


def test_optional_skill_runtime_is_dry_stepped() -> None:
    rows = [
        {
            "timestamp": 0.0,
            "skill": "wait",
            "allowed_skills": ["wait"],
            "observation": {},
        }
    ]
    result = ReplayEnv.from_samples(
        rows,
        policy=SequencePolicy(["wait"]),
        skill_runtime=SkillRuntime(),
    ).run()[0]
    assert result.runtime_result is not None
    assert result.runtime_result.requested_action == "wait"
