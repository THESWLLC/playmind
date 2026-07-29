#!/usr/bin/env python3
"""Compare gameplay policies using dry demonstration replay or synthetic scenarios."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playmind.demonstrations import list_sessions, load_session_samples
from playmind.evaluation.metrics import aggregate_episode_metrics, summarize_replay_results
from playmind.evaluation.report import write_evaluation_report
from playmind.evaluation.scenarios import make_baseline_policies, run_all_scenarios
from playmind.models.policy_v2 import SkillPolicyV2
from playmind.models.recurrent_policy import RecurrentSkillPolicyV2
from playmind.policies.hybrid import HybridPolicy
from playmind.policies.scripted import DEFAULT_SKILL_ORDER
from playmind.replay_env import ReplayEnv


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Outcome-oriented, actuator-free PlayMind policy evaluation"
    )
    p.add_argument(
        "--out-dir",
        default="data/playmind/evaluation/reports",
        help="Legacy report root (a run-id subdirectory is created)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Exact directory for report.json, metrics.csv, and report.md",
    )
    p.add_argument("--run-id", default=None, help="Report run id (default: timestamp)")
    p.add_argument(
        "--data-dir",
        default="data/playmind/demonstrations",
        help="Demonstration root; synthetic scenarios are used when it is empty",
    )
    p.add_argument(
        "--checkpoints",
        nargs="*",
        default=[],
        help="Old MLP and/or recurrent checkpoint metadata paths",
    )
    p.add_argument(
        "--compare-scripted",
        action="store_true",
        help="Include scripted baseline (included by default for compatibility)",
    )
    p.add_argument(
        "--work-dir",
        default=None,
        help="Optional dir to materialize synthetic demo sessions",
    )
    p.add_argument(
        "--bc-checkpoint",
        default=None,
        help="Optional SkillPolicyV2 checkpoint to include in comparison",
    )
    p.add_argument(
        "--episodes-jsonl",
        default=None,
        help="Optional episodes.jsonl for offline KPI stubs (kills/hour etc.)",
    )
    return p


def _load_episodes(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    candidates = [path, path.with_suffix(".json")]
    for candidate in candidates:
        if not candidate.exists() or candidate.suffix != ".json":
            continue
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(loaded, dict):
            return loaded
    return {}


def _load_checkpoint(path: Path) -> tuple[str, Any]:
    metadata = _checkpoint_metadata(path)
    model_type = str(
        metadata.get("model_type")
        or (metadata.get("config") or {}).get("model_type")
        or ""
    ).lower()
    if "recurrent" in model_type:
        return "recurrent", RecurrentSkillPolicyV2.load(path)
    if model_type in {"structured_mlp_legacy", "skill_policy_v2"} or "mlp" in model_type:
        return "old_mlp", SkillPolicyV2.load(path)
    # Let architecture-specific loaders provide useful compatibility errors.
    try:
        return "recurrent", RecurrentSkillPolicyV2.load(path)
    except (ValueError, KeyError):
        return "old_mlp", SkillPolicyV2.load(path)


def _load_demonstrations(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session_dir in list_sessions(root):
        session_meta: dict[str, Any] = {}
        session_json = session_dir / "session.json"
        if session_json.exists():
            try:
                loaded = json.loads(session_json.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    session_meta = loaded
            except (OSError, ValueError):
                pass
        for index, raw in enumerate(load_session_samples(session_dir)):
            sample = dict(raw)
            sample["_session_outcome"] = session_meta.get("outcome")
            sample["_session_metadata"] = session_meta
            sample["_sequence_start"] = index == 0
            rows.append(sample)
    return rows


def _comparison_row(report: dict[str, Any]) -> dict[str, Any]:
    labels = report.get("label_agreement") or {}
    validity = report.get("decision_validity") or {}
    temporal = report.get("temporal") or {}
    observed = report.get("observed_outcomes") or {}
    return {
        "policy": report.get("policy"),
        "n_steps": report.get("n_steps", 0),
        "agreement_rate": labels.get("accuracy", report.get("agreement_rate", 0.0)),
        "top2_accuracy": labels.get("top2_accuracy"),
        "top3_accuracy": labels.get("top3_accuracy"),
        "invalid_skill_proposal_rate": validity.get("invalid_skill_proposal_rate", 0.0),
        "masked_rate": validity.get("masked_rate", 0.0),
        "scripted_fallback_rate": validity.get("scripted_fallback_rate", 0.0),
        "skill_switch_rate": temporal.get("skill_switch_rate", 0.0),
        "oscillation_rate": temporal.get("oscillation_rate", 0.0),
        "confirmed_kill_count": observed.get("confirmed_kill_count", 0),
        "death_count": observed.get("death_count", 0),
        "objective_progress_delta": observed.get("objective_progress_delta", 0.0),
        "evidence": "observed outcomes are from demonstrations; policy outcomes are counterfactual",
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    work_dir = Path(args.work_dir) if args.work_dir else None

    policies = make_baseline_policies()
    checkpoint_paths = [Path(value) for value in args.checkpoints]
    if args.bc_checkpoint:
        checkpoint_paths.append(Path(args.bc_checkpoint))
    loaded_models: dict[str, Any] = {}
    for ckpt in checkpoint_paths:
        if not ckpt.exists() and not ckpt.with_suffix(".json").exists():
            print(f"Checkpoint not found: {ckpt}; skipping")
            continue
        try:
            base_name, policy = _load_checkpoint(ckpt)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            print(f"Could not load checkpoint {ckpt}: {exc}; skipping")
            continue
        name = base_name
        suffix = 2
        while name in loaded_models:
            name = f"{base_name}_{suffix}"
            suffix += 1
        loaded_models[name] = policy
        policies[name] = policy

    primary = loaded_models.get("recurrent") or loaded_models.get("old_mlp")
    policies["hybrid"] = HybridPolicy(primary=primary) if primary is not None else HybridPolicy()

    comparisons: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    samples = _load_demonstrations(Path(args.data_dir))
    if samples:
        allowed = list(
            dict.fromkeys(
                [*DEFAULT_SKILL_ORDER]
                + [str(row["skill"]) for row in samples if row.get("skill")]
            )
        )
        for name, policy in policies.items():
            env = ReplayEnv.from_samples(samples, policy=policy, allowed_skills=allowed)
            report = summarize_replay_results(env.run(), policy_name=name)
            comparisons[name] = report
            csv_rows.append({"policy": name, **_comparison_row(report)})
            print(
                f"{name}: agreement={report['agreement_rate']:.3f} "
                f"steps={report['n_steps']}"
            )
    else:
        print(f"No demonstrations found under {args.data_dir}; using synthetic scenarios.")
        for name, policy in policies.items():
            result = run_all_scenarios(policy, work_dir=work_dir)
            comparisons[name] = {
                "mean_agreement": result["mean_agreement"],
                "n_scenarios": result["n_scenarios"],
                "policy": result["policy"],
                "scenarios": {
                    scenario: {
                        "agreement_rate": scenario_result.get("agreement_rate"),
                        "n_steps": scenario_result.get("n_steps"),
                        "fallback_rate": scenario_result.get("fallback_rate"),
                    }
                    for scenario, scenario_result in result["scenarios"].items()
                }
            }
            for scenario, scenario_result in result["scenarios"].items():
                csv_rows.append(
                    {
                        "policy": name,
                        "scenario": scenario,
                        "agreement_rate": scenario_result.get("agreement_rate"),
                        "n_steps": scenario_result.get("n_steps"),
                        "fallback_rate": scenario_result.get("fallback_rate"),
                    }
                )
            print(
                f"{name}: mean_agreement={result['mean_agreement']:.3f} "
                f"scenarios={result['n_scenarios']}"
            )

    episode_metrics: dict[str, Any] | None = None
    if args.episodes_jsonl:
        episodes = _load_episodes(Path(args.episodes_jsonl))
        episode_metrics = aggregate_episode_metrics(episodes)
        print("episode_metrics:", json.dumps(episode_metrics, sort_keys=True))

    payload = {
        "title": "Outcome-based offline gameplay evaluation",
        "run_id": run_id,
        "summary": {
            "n_policies": len(comparisons),
            "n_demo_steps": len(samples),
            "mode": "demonstration_replay" if samples else "synthetic_scenarios",
            "best_policy": max(
                comparisons.items(),
                key=lambda kv: float(
                    kv[1].get(
                        "mean_agreement",
                        (kv[1].get("label_agreement") or {}).get("accuracy", 0),
                    )
                    or 0
                ),
            )[0]
            if comparisons
            else None,
        },
        "comparisons": comparisons,
        "episode_metrics": episode_metrics,
        "notes": [
            "ReplayEnv is dry: no actuators or live game are used.",
            "Observed outcomes come from demonstration evidence and are shared across policies.",
            "Policy-dependent outcome estimates are counterfactual, never confirmed.",
            "Legacy Q uses an empty CPU-only OnlinePolicy bridge.",
        ],
        "csv_rows": csv_rows,
    }
    if args.output_dir:
        report_root, report_run_id = Path(args.output_dir), ""
    else:
        report_root, report_run_id = Path(args.out_dir), run_id
    paths = write_evaluation_report(
        report_root, payload, run_id=report_run_id, csv_rows=csv_rows
    )
    print("Wrote reports:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
