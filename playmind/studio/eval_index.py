"""Canonical discovery/index for planner evaluation reports."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_EVALUATION_ROOT = Path("data/playmind/planner/evaluation")
DEFAULT_EVALUATION_INDEX = DEFAULT_EVALUATION_ROOT / "index.json"


def normalize_comparisons(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    existing = report.get("comparisons")
    if isinstance(existing, Mapping):
        return {
            str(name): dict(value)
            for name, value in existing.items()
            if isinstance(value, Mapping)
        }
    if isinstance(existing, list):
        return {
            str(item.get("backend") or item.get("name")): {
                key: value
                for key, value in item.items()
                if key not in {"backend", "name"}
            }
            for item in existing
            if isinstance(item, Mapping)
            and (item.get("backend") or item.get("name"))
        }
    backends = report.get("backends")
    if not isinstance(backends, Mapping):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for name, value in backends.items():
        item = dict(value) if isinstance(value, Mapping) else {}
        metrics = item.get("metrics", {})
        item["metrics"] = dict(metrics) if isinstance(metrics, Mapping) else {}
        result[str(name)] = item
    return result


def normalize_report(report: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    source = Path(path)
    run_id = (
        source.parent.name
        if source.name == "report.json" and source.parent.name != "runs"
        else source.stem
    )
    return {
        "run_id": str(report.get("run_id") or run_id),
        "path": str(source),
        "created_at": report.get("created_at"),
        "scenario_count": int(report.get("scenario_count") or 0),
        "comparisons": normalize_comparisons(report),
        "artifacts": dict(report.get("artifacts") or {})
        if isinstance(report.get("artifacts"), Mapping)
        else {},
        "smoke": bool(report.get("smoke", False)),
    }


def discover_reports(
    root: str | Path = DEFAULT_EVALUATION_ROOT,
) -> list[dict[str, Any]]:
    directory = Path(root)
    if not directory.exists():
        return []
    candidates = list(directory.glob("planner_benchmark_*.json"))
    candidates.extend(directory.glob("runs/*/report.json"))
    by_run: dict[str, dict[str, Any]] = {}
    for path in sorted(candidates):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, Mapping):
            continue
        normalized = normalize_report(value, path)
        run_id = normalized["run_id"]
        if run_id not in by_run or path.name == "report.json":
            by_run[run_id] = normalized
    return sorted(
        by_run.values(),
        key=lambda item: (str(item.get("created_at") or ""), item["run_id"]),
        reverse=True,
    )


discover_evaluations = discover_reports


def write_index(
    root: str | Path = DEFAULT_EVALUATION_ROOT,
    *,
    index_path: str | Path | None = None,
) -> dict[str, Any]:
    directory = Path(root)
    destination = Path(index_path) if index_path is not None else directory / "index.json"
    reports = discover_reports(directory)
    payload = {
        "schema_version": 1,
        "report_count": len(reports),
        "reports": reports,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)
    payload["path"] = str(destination)
    return payload


update_index = write_index


def load_latest(
    root: str | Path = DEFAULT_EVALUATION_ROOT,
) -> dict[str, Any] | None:
    """Load the newest canonical report, normalizing legacy ``backends`` data."""

    reports = discover_reports(root)
    if not reports:
        return None
    latest = dict(reports[0])
    path = Path(str(latest["path"]))
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return latest
    if not isinstance(raw, Mapping):
        return latest
    report = dict(raw)
    report["comparisons"] = normalize_comparisons(report)
    latest["report"] = report
    latest["comparisons"] = report["comparisons"]
    try:
        latest["modified_at"] = path.stat().st_mtime
    except OSError:
        pass
    return latest


__all__ = [
    "DEFAULT_EVALUATION_INDEX",
    "DEFAULT_EVALUATION_ROOT",
    "discover_evaluations",
    "discover_reports",
    "load_latest",
    "normalize_comparisons",
    "normalize_report",
    "update_index",
    "write_index",
]
