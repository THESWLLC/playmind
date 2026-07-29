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
        lines.append("| Policy | Mean agreement | Scenarios |")
        lines.append("|--------|----------------|-----------|")
        for name, row in comparisons.items():
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"| {name} | {float(row.get('mean_agreement') or 0):.3f} | "
                f"{int(row.get('n_scenarios') or 0)} |"
            )
        lines.append("")

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
