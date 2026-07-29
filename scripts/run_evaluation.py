#!/usr/bin/env python3
"""Run offline evaluation scenarios: scripted vs legacy stub comparison."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from playmind.evaluation.metrics import aggregate_episode_metrics
from playmind.evaluation.report import write_evaluation_report
from playmind.evaluation.scenarios import make_baseline_policies, run_all_scenarios
from playmind.models.policy_v2 import SkillPolicyV2


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare scripted vs legacy stub (and optional BC) on synthetic scenarios"
    )
    p.add_argument(
        "--out-dir",
        default="data/playmind/evaluation/reports",
        help="Root directory for evaluation reports",
    )
    p.add_argument("--run-id", default=None, help="Report run id (default: timestamp)")
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


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    work_dir = Path(args.work_dir) if args.work_dir else None

    policies = make_baseline_policies()
    if args.bc_checkpoint:
        ckpt = Path(args.bc_checkpoint)
        if ckpt.exists():
            policies["behavior_clone"] = SkillPolicyV2.load(ckpt)
        else:
            print(f"BC checkpoint not found: {ckpt}; skipping")

    comparisons: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for name, policy in policies.items():
        result = run_all_scenarios(policy, work_dir=work_dir)
        comparisons[name] = {
            "mean_agreement": result["mean_agreement"],
            "n_scenarios": result["n_scenarios"],
            "policy": result["policy"],
            "scenarios": {
                s: {
                    "agreement_rate": sc.get("agreement_rate"),
                    "n_steps": sc.get("n_steps"),
                    "fallback_rate": sc.get("fallback_rate"),
                }
                for s, sc in result["scenarios"].items()
            },
        }
        for scen, sc in result["scenarios"].items():
            csv_rows.append(
                {
                    "policy": name,
                    "scenario": scen,
                    "agreement_rate": sc.get("agreement_rate"),
                    "n_steps": sc.get("n_steps"),
                    "fallback_rate": sc.get("fallback_rate"),
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
        "title": "Scripted vs legacy stub evaluation",
        "run_id": run_id,
        "summary": {
            "n_policies": len(comparisons),
            "best_policy": max(
                comparisons.items(),
                key=lambda kv: float(kv[1].get("mean_agreement") or 0),
            )[0]
            if comparisons
            else None,
        },
        "comparisons": comparisons,
        "episode_metrics": episode_metrics,
        "notes": [
            "Synthetic ReplayEnv scenarios — no actuators / live game.",
            "Legacy stub uses LegacyQPolicy with empty OnlinePolicy bridge.",
            "kills/hour and related KPIs are stubs from episode records when provided.",
        ],
        "csv_rows": csv_rows,
    }
    paths = write_evaluation_report(args.out_dir, payload, run_id=run_id, csv_rows=csv_rows)
    print("Wrote reports:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
