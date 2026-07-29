"""Offline evaluation metrics, scenarios, and report writers for Learning V2."""

from playmind.evaluation.metrics import (
    aggregate_episode_metrics,
    classification_metrics,
    decision_validity_metrics,
    kills_per_hour,
    observed_outcome_metrics,
    outcome_evaluation_report,
    skill_success_rates,
    summarize_replay_results,
    temporal_metrics,
)
from playmind.evaluation.report import write_evaluation_report
from playmind.evaluation.scenarios import (
    SCENARIO_SPECS,
    build_synthetic_session,
    run_all_scenarios,
    run_scenario,
)

__all__ = [
    "SCENARIO_SPECS",
    "aggregate_episode_metrics",
    "build_synthetic_session",
    "classification_metrics",
    "decision_validity_metrics",
    "kills_per_hour",
    "observed_outcome_metrics",
    "outcome_evaluation_report",
    "run_all_scenarios",
    "run_scenario",
    "skill_success_rates",
    "summarize_replay_results",
    "temporal_metrics",
    "write_evaluation_report",
]
