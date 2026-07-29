"""Offline evaluation metrics, scenarios, and report writers for Learning V2."""

from playmind.evaluation.metrics import (
    aggregate_episode_metrics,
    kills_per_hour,
    skill_success_rates,
    summarize_replay_results,
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
    "kills_per_hour",
    "run_all_scenarios",
    "run_scenario",
    "skill_success_rates",
    "summarize_replay_results",
    "write_evaluation_report",
]
