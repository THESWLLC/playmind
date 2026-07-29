"""JSON / CSV / Markdown writers for evaluation reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json_report(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return path


def write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    # Union of keys preserving first-row order then extras.
    fieldnames: list[str] = list(rows[0].keys())
    for row in rows[1:]:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return path


def write_markdown_report(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# PlayMind Evaluation Report", ""]
    title = payload.get("title")
    if title:
        lines.append(f"**{title}**")
        lines.append("")

    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        lines.append("## Summary")
        lines.append("")
        for k, v in summary.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    comparisons = payload.get("comparisons")
    if isinstance(comparisons, Mapping):
        lines.append("## Policy comparison")
        lines.append("")
        lines.append("| Policy | Label agreement | Steps/scenarios | Invalid proposals | Switch rate |")
        lines.append("|--------|-----------------|-----------------|-------------------|-------------|")
        for name, row in comparisons.items():
            if not isinstance(row, Mapping):
                continue
            labels = row.get("label_agreement") if isinstance(row.get("label_agreement"), Mapping) else {}
            validity = row.get("decision_validity") if isinstance(row.get("decision_validity"), Mapping) else {}
            temporal = row.get("temporal") if isinstance(row.get("temporal"), Mapping) else {}
            agreement = row.get(
                "mean_agreement",
                row.get("agreement_rate", labels.get("accuracy") if labels else 0.0),
            )
            volume = row.get("n_steps", row.get("n_scenarios", 0))
            lines.append(
                f"| {name} | {float(agreement or 0):.3f} | "
                f"{int(volume or 0)} | "
                f"{float(validity.get('invalid_skill_proposal_rate') or 0):.3f} | "
                f"{float(temporal.get('skill_switch_rate') or 0):.3f} |"
            )
        lines.append("")

        def score(*names: str) -> tuple[str | None, float | None]:
            for wanted in names:
                for actual, row in comparisons.items():
                    if str(actual).lower() != wanted.lower() or not isinstance(row, Mapping):
                        continue
                    labels = row.get("label_agreement")
                    value = row.get("mean_agreement", row.get("agreement_rate"))
                    if value is None and isinstance(labels, Mapping):
                        value = labels.get("accuracy")
                    try:
                        return str(actual), float(value)
                    except (TypeError, ValueError):
                        return str(actual), 0.0
            return None, None

        recurrent_name, recurrent = score("recurrent", "recurrent_skill_policy_v2")
        lines.append("## Recurrent baseline conclusion")
        lines.append("")
        if recurrent is None:
            lines.append(
                "Recurrent outperformance could not be determined because no recurrent checkpoint was evaluated."
            )
            lines.append("- Recurrent vs old MLP: not evaluated.")
            lines.append("- Recurrent vs scripted: not evaluated.")
            lines.append("- Recurrent vs random-valid-skill: not evaluated.")
        else:
            labels = (
                ("old MLP", score("old_mlp", "behavior_clone", "skill_policy_v2")[1]),
                ("scripted", score("scripted")[1]),
                ("random-valid-skill", score("random_valid_skill", "random-valid-skill")[1]),
            )
            for baseline, baseline_score in labels:
                if baseline_score is None:
                    lines.append(f"- Recurrent vs {baseline}: not evaluated.")
                else:
                    relation = (
                        "outperformed"
                        if recurrent > baseline_score
                        else "tied"
                        if recurrent == baseline_score
                        else "did not outperform"
                    )
                    lines.append(
                        f"- Recurrent {relation} {baseline} on label agreement "
                        f"({recurrent:.3f} vs {baseline_score:.3f})."
                    )
        lines.append("")

        first_report = next(
            (row for row in comparisons.values() if isinstance(row, Mapping)),
            None,
        )
        if isinstance(first_report, Mapping) and any(
            key in first_report
            for key in (
                "observed_outcomes",
                "label_agreement",
                "model_predicted",
                "counterfactual_estimates",
            )
        ):
            lines.extend(
                [
                    "## Evidence sections",
                    "",
                    "### Observed outcomes",
                    "",
                    "These values come only from recorded demonstration events and observations; "
                    "they are not outcomes caused by the replayed policy.",
                    "",
                ]
            )
            for name, row in comparisons.items():
                if not isinstance(row, Mapping):
                    continue
                observed = row.get("observed_outcomes")
                if isinstance(observed, Mapping):
                    lines.append(
                        f"- **{name}**: confirmed kills={int(observed.get('confirmed_kill_count') or 0)}, "
                        f"deaths={int(observed.get('death_count') or 0)}, "
                        f"objective progress={float(observed.get('objective_progress_delta') or 0):.3f}"
                    )
            lines.extend(
                [
                    "",
                    "### Label agreement",
                    "",
                    "Agreement measures imitation of demonstration labels, not gameplay success.",
                    "",
                    "### Model predicted",
                    "",
                    "Auxiliary-head values are model predictions and are not observed gameplay outcomes.",
                    "",
                    "### Counterfactual estimates",
                    "",
                    "**Not confirmed:** replay decisions were never actuated. Counterfactual estimates "
                    "must not be interpreted as kills, deaths, progress, or other confirmed outcomes.",
                    "",
                ]
            )

    metrics = payload.get("episode_metrics")
    if isinstance(metrics, Mapping):
        lines.append("## Episode metrics")
        lines.append("")
        for k, v in metrics.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")

    extras = payload.get("notes")
    if extras:
        lines.append("## Notes")
        lines.append("")
        if isinstance(extras, (list, tuple)):
            for n in extras:
                lines.append(f"- {n}")
        else:
            lines.append(str(extras))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_evaluation_report(
    out_dir: str | Path,
    payload: Mapping[str, Any],
    *,
    run_id: str = "latest",
    csv_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Write JSON + Markdown (+ optional CSV) under ``out_dir/run_id/``."""
    root = _ensure_dir(Path(out_dir) / run_id)
    paths = {
        "json": str(write_json_report(root / "report.json", payload)),
        "markdown": str(write_markdown_report(root / "report.md", payload)),
    }
    rows = list(csv_rows) if csv_rows is not None else list(payload.get("csv_rows") or [])
    if rows:
        paths["csv"] = str(write_csv_rows(root / "metrics.csv", rows))
    return paths
