"""Planner benchmark metrics and documented aggregate score."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

# Score rewards valid, legal JSON plans and planning quality.  Every benchmark
# report includes both these weights and the unweighted component values.
BENCHMARK_WEIGHTS: dict[str, float] = {
    "valid_plan_rate": 0.30,
    "legal_skill_rate": 0.20,
    "json_success_rate": 0.15,
    "first_skill_correct": 0.15,
    "full_plan_score": 0.20,
}


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * max(0.0, min(100.0, percent)) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def skill_names(plan: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(plan, Mapping) or not isinstance(plan.get("skills"), list):
        return []
    result: list[str] = []
    for value in plan["skills"]:
        if isinstance(value, Mapping):
            name = value.get("name") or value.get("skill")
        else:
            name = value
        if name not in (None, ""):
            result.append(str(name))
    return result


def full_plan_similarity(actual: Sequence[str], expected: Sequence[str]) -> float:
    """Return normalized longest-common-subsequence similarity."""
    if not actual and not expected:
        return 1.0
    if not actual or not expected:
        return 0.0
    previous = [0] * (len(expected) + 1)
    for actual_name in actual:
        current = [0]
        for index, expected_name in enumerate(expected, 1):
            if actual_name == expected_name:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1] / max(len(actual), len(expected))


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        components = {name: 0.0 for name in BENCHMARK_WEIGHTS}
        return {
            "scenario_count": 0,
            "valid_plan_rate": 0.0,
            "illegal_skill_rate": 0.0,
            "json_fail_rate": 0.0,
            "first_skill_correct": 0.0,
            "full_plan_score": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "benchmark_score": 0.0,
            "benchmark_components": components,
            "benchmark_weights": dict(BENCHMARK_WEIGHTS),
            "overfitting_signals": {
                "duplicate_output_rate": 0.0,
                "output_diversity_rate": 0.0,
                "exact_match_rate": 0.0,
                "warning": False,
            },
        }

    def rate(field: str) -> float:
        return sum(float(bool(row.get(field))) for row in rows) / count

    valid_rate = rate("valid")
    illegal_rate = rate("illegal_skill")
    json_fail_rate = rate("json_failed")
    first_correct = rate("first_skill_correct")
    full_score = sum(float(row.get("full_plan_score") or 0.0) for row in rows) / count
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    outputs = [str(row.get("normalized_output") or "") for row in rows]
    frequencies = Counter(outputs)
    duplicate_count = sum(amount - 1 for amount in frequencies.values() if amount > 1)
    duplicate_rate = duplicate_count / count
    diversity_rate = len(frequencies) / count
    exact_match_rate = sum(
        float(
            list(row.get("actual_skills") or [])
            == list(row.get("expected_skills") or [])
        )
        for row in rows
    ) / count
    components = {
        "valid_plan_rate": valid_rate,
        "legal_skill_rate": 1.0 - illegal_rate,
        "json_success_rate": 1.0 - json_fail_rate,
        "first_skill_correct": first_correct,
        "full_plan_score": full_score,
    }
    benchmark_score = sum(
        components[name] * weight for name, weight in BENCHMARK_WEIGHTS.items()
    )
    overfitting_warning = count >= 10 and (
        duplicate_rate > 0.8 or (exact_match_rate > 0.99 and diversity_rate < 0.2)
    )
    return {
        "scenario_count": count,
        "valid_plan_rate": valid_rate,
        "illegal_skill_rate": illegal_rate,
        "json_fail_rate": json_fail_rate,
        "first_skill_correct": first_correct,
        "full_plan_score": full_score,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "benchmark_score": benchmark_score,
        "benchmark_components": components,
        "benchmark_weights": dict(BENCHMARK_WEIGHTS),
        "overfitting_signals": {
            "duplicate_output_rate": duplicate_rate,
            "output_diversity_rate": diversity_rate,
            "exact_match_rate": exact_match_rate,
            "warning": overfitting_warning,
        },
    }


__all__ = [
    "BENCHMARK_WEIGHTS",
    "aggregate_metrics",
    "full_plan_similarity",
    "percentile",
    "skill_names",
]
