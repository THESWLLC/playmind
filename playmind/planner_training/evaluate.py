"""Compare planner backends on the frozen suite and report promotion gates."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playmind.planner_data.export_eval_suite import (
    DEFAULT_EVALUATION_ROOT,
    FROZEN_EVAL_SCENARIOS,
)
from playmind.planner_training.metrics import (
    aggregate_metrics,
    full_plan_similarity,
    skill_names,
)
from playmind.planner_v2.model_registry import DEFAULT_REGISTRY_PATH, ModelRegistry
from playmind.planner_v2.ollama_client import OllamaClient
from playmind.policies.scripted import ScriptedPolicy
from playmind.skills import list_skills


def load_scenarios(path: str | Path | None = None) -> list[dict[str, Any]]:
    if path is None:
        return [scenario.to_dict() for scenario in FROZEN_EVAL_SCENARIOS]
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(dict(value))
    if not rows:
        raise ValueError(f"evaluation suite is empty: {path}")
    return rows


def _sensor_observation(state: Mapping[str, Any]) -> dict[str, Any]:
    observation: dict[str, Any] = {}
    sensors = state.get("sensors")
    if isinstance(sensors, Mapping):
        for name, payload in sensors.items():
            if isinstance(payload, Mapping) and payload.get("known", True):
                observation[str(name)] = payload.get("value")
    lifecycle = state.get("life_phase") or state.get("lifecycle_state")
    if lifecycle is not None:
        observation["life_phase"] = lifecycle
    for key in ("objective_text", "sensor_warnings", "recent_action", "recent_action_outcome"):
        if key in state:
            observation[key] = state[key]
    return observation


class ScriptedPlannerBackend:
    """Dependency-free baseline backed by the existing scripted policy."""

    def __init__(self) -> None:
        self.policy = ScriptedPolicy()

    def generate_plan(
        self, planner_state: Mapping[str, Any], scenario: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        allowed = list(planner_state.get("available_skills") or list_skills())
        decision = self.policy.choose_skill(
            {
                "obs": _sensor_observation(planner_state),
                "goal": planner_state.get("goal"),
                "stuck": (
                    planner_state.get("lifecycle_state") == "stuck"
                    or "stagnation" in str(planner_state)
                ),
            },
            allowed,
        )
        return {
            "skills": [decision.skill],
            "rationale": decision.reason,
        }


class OllamaPlannerBackend:
    def __init__(
        self,
        model: str,
        *,
        host: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
        client: OllamaClient | None = None,
    ) -> None:
        self.model = model
        self.client = client or OllamaClient(host, timeout=timeout)
        self.timeout = timeout

    def generate_plan(
        self, planner_state: Mapping[str, Any], scenario: Mapping[str, Any] | None = None
    ) -> str:
        return self.client.generate_plan(
            planner_state, self.model, timeout=self.timeout
        )


def _invoke_backend(
    backend: Any,
    state: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> Any:
    function = getattr(backend, "generate_plan", backend)
    if not callable(function):
        raise TypeError("backend must be callable or expose generate_plan")
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(state)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return function(state, scenario) if len(positional) >= 2 else function(state)


def _parse_plan(raw: Any) -> tuple[dict[str, Any] | None, bool, str]:
    if hasattr(raw, "to_dict") and callable(raw.to_dict):
        raw = raw.to_dict()
    if isinstance(raw, Mapping):
        plan = dict(raw)
    elif isinstance(raw, str):
        clean = raw.strip()
        if clean.startswith("```") and clean.endswith("```"):
            lines = clean.splitlines()
            clean = "\n".join(lines[1:-1]).strip()
        try:
            decoded = json.loads(clean)
        except (json.JSONDecodeError, TypeError):
            return None, True, str(raw)
        if not isinstance(decoded, Mapping):
            return None, False, clean
        plan = dict(decoded)
    else:
        return None, False, str(raw)
    return plan, False, json.dumps(plan, sort_keys=True, default=str)


def _evaluate_backend(
    name: str,
    backend: Any,
    scenarios: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    known_skills = set(list_skills())
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        state_value = scenario.get("planner_state")
        state = dict(state_value) if isinstance(state_value, Mapping) else {}
        expected_value = scenario.get("expected_plan")
        expected = (
            dict(expected_value) if isinstance(expected_value, Mapping) else {}
        )
        expected_skills = skill_names(expected)
        available = set(str(item) for item in state.get("available_skills") or [])
        allowed = available or known_skills
        started = time.perf_counter()
        error = ""
        try:
            raw = _invoke_backend(backend, state, scenario)
        except Exception as exc:  # Backends are intentionally fault-isolated.
            raw = ""
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000.0
        plan, json_failed, normalized = _parse_plan(raw)
        actual_skills = skill_names(plan)
        structural_valid = bool(
            isinstance(plan, Mapping)
            and isinstance(plan.get("skills"), list)
            and actual_skills
            and len(actual_skills) == len(plan["skills"])
        )
        illegal_names = sorted(set(actual_skills) - allowed)
        illegal = bool(illegal_names)
        rows.append(
            {
                "backend": name,
                "scenario_id": str(scenario.get("scenario_id") or ""),
                "category": str(scenario.get("category") or ""),
                "valid": structural_valid and not illegal,
                "json_failed": json_failed,
                "illegal_skill": illegal,
                "illegal_skills": illegal_names,
                "first_skill_correct": bool(
                    actual_skills
                    and expected_skills
                    and actual_skills[0] == expected_skills[0]
                ),
                "full_plan_score": full_plan_similarity(
                    actual_skills, expected_skills
                ),
                "actual_skills": actual_skills,
                "expected_skills": expected_skills,
                "latency_ms": latency_ms,
                "normalized_output": normalized,
                "error": error,
            }
        )
    return aggregate_metrics(rows), rows


def _report_stem() -> str:
    return "planner_benchmark_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )


def _write_reports(
    output_dir: Path,
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _report_stem()
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    markdown_path = output_dir / f"{stem}.md"
    artifacts = {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
    }
    persisted_report = dict(report)
    persisted_report["artifacts"] = artifacts
    json_path.write_text(
        json.dumps(persisted_report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "backend",
        "scenario_id",
        "category",
        "valid",
        "json_failed",
        "illegal_skill",
        "first_skill_correct",
        "full_plan_score",
        "latency_ms",
        "actual_skills",
        "expected_skills",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    lines = [
        "# PlayMind planner benchmark",
        "",
        "Weighted score = 30% valid plans + 20% legal skills + 15% JSON "
        "success + 15% correct first skill + 20% full-plan similarity.",
        "",
        "| Backend | Score | Valid | Illegal | JSON fail | First correct | Full plan | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backend_name, backend_report in report["backends"].items():
        metrics = backend_report["metrics"]
        lines.append(
            "| {name} | {score:.4f} | {valid:.4f} | {illegal:.4f} | "
            "{json_fail:.4f} | {first:.4f} | {full:.4f} | {p50:.2f} | "
            "{p95:.2f} |".format(
                name=backend_name,
                score=metrics["benchmark_score"],
                valid=metrics["valid_plan_rate"],
                illegal=metrics["illegal_skill_rate"],
                json_fail=metrics["json_fail_rate"],
                first=metrics["first_skill_correct"],
                full=metrics["full_plan_score"],
                p50=metrics["latency_p50_ms"],
                p95=metrics["latency_p95_ms"],
            )
        )
        lines.extend(
            [
                "",
                f"Components for `{backend_name}`: "
                f"`{json.dumps(metrics['benchmark_components'], sort_keys=True)}`",
                f"Overfitting signals: "
                f"`{json.dumps(metrics['overfitting_signals'], sort_keys=True)}`",
            ]
        )
        gates = backend_report.get("promotion_gates")
        if gates is not None:
            lines.append(
                "Promotion gates: "
                + ("PASS (not promoted)" if gates["passed"] else "FAIL")
                + f"; `{json.dumps(gates['errors'])}`"
            )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifacts


def evaluate_backends(
    backends: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]] | None = None,
    *,
    output_dir: str | Path = DEFAULT_EVALUATION_ROOT,
    registry: ModelRegistry | None = None,
    registry_model_ids: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if len(backends) < 2:
        raise ValueError("planner evaluation requires at least two backends")
    suite = list(scenarios) if scenarios is not None else load_scenarios()
    if not suite:
        raise ValueError("planner evaluation requires at least one scenario")
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario_count": len(suite),
        "score_documentation": {
            "formula": (
                "0.30*valid_plan_rate + 0.20*(1-illegal_skill_rate) + "
                "0.15*(1-json_fail_rate) + 0.15*first_skill_correct + "
                "0.20*full_plan_score"
            )
        },
        "backends": {},
    }
    all_rows: list[dict[str, Any]] = []
    ids = dict(registry_model_ids or {})
    for name, backend in backends.items():
        metrics, rows = _evaluate_backend(name, backend, suite)
        all_rows.extend(rows)
        backend_report: dict[str, Any] = {"metrics": metrics}
        model_id = ids.get(name)
        if registry is not None and model_id:
            model = registry.update_metrics(
                model_id,
                eval_metrics=metrics,
                reason="planner benchmark completed; status unchanged",
            )
            gate_errors = registry.promotion_errors(model_id)
            backend_report["registry_model_id"] = model_id
            backend_report["registry_status"] = model["status"]
            backend_report["promotion_gates"] = {
                "passed": not gate_errors,
                "errors": gate_errors,
                "note": "Evaluation never promotes a model; promotion is explicit.",
            }
        report["backends"][name] = backend_report
    report["artifacts"] = _write_reports(Path(output_dir), report, all_rows)
    return report


evaluate = evaluate_backends


def _registry_backend_model(record: Mapping[str, Any]) -> str:
    for key in ("gguf_path", "merged_path", "display_name", "base_model"):
        if record.get(key):
            return str(record[key])
    return str(record["model_id"])


def build_default_backends(
    *,
    generic_model: str,
    host: str,
    timeout: float,
    registry: ModelRegistry,
    candidate_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    backends: dict[str, Any] = {
        "scripted": ScriptedPlannerBackend(),
        "ollama_generic": OllamaPlannerBackend(
            generic_model, host=host, timeout=timeout
        ),
    }
    ids: dict[str, str] = {}
    production = registry.get_production()
    if production is not None:
        backends["production"] = OllamaPlannerBackend(
            _registry_backend_model(production), host=host, timeout=timeout
        )
        ids["production"] = str(production["model_id"])
    candidate = registry.get(candidate_id) if candidate_id else None
    if candidate is None:
        candidates = registry.list(status="candidate", limit=1)
        candidate = candidates[0] if candidates else None
    if candidate is not None:
        backends["candidate"] = OllamaPlannerBackend(
            _registry_backend_model(candidate), host=host, timeout=timeout
        )
        ids["candidate"] = str(candidate["model_id"])
    return backends, ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite")
    parser.add_argument("--output-dir", default=str(DEFAULT_EVALUATION_ROOT))
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--generic-model", default="llama3.2")
    parser.add_argument("--candidate-id")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = ModelRegistry(args.registry_path)
    backends, ids = build_default_backends(
        generic_model=args.generic_model,
        host=args.ollama_host,
        timeout=args.timeout,
        registry=registry,
        candidate_id=args.candidate_id,
    )
    report = evaluate_backends(
        backends,
        load_scenarios(args.suite),
        output_dir=args.output_dir,
        registry=registry,
        registry_model_ids=ids,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OllamaPlannerBackend",
    "ScriptedPlannerBackend",
    "build_default_backends",
    "evaluate",
    "evaluate_backends",
    "load_scenarios",
    "main",
]
