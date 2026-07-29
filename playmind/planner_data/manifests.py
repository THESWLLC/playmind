"""Planner dataset manifest generation and integrity hashes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playmind.planner_data.schemas import PLANNER_DATA_SCHEMA_VERSION


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coverage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    skills: set[str] = set()
    categories: set[str] = set()
    lifecycle_states: set[str] = set()
    unknown_sensors: set[str] = set()
    input_sources: set[str] = set()
    for record in records:
        category = record.get("category")
        if category:
            categories.add(str(category))
        source = record.get("input_source")
        if source:
            input_sources.add(str(source))
        state = record.get("planner_state")
        if isinstance(state, Mapping):
            lifecycle = state.get("lifecycle_state")
            if lifecycle:
                lifecycle_states.add(str(lifecycle))
            unknown_sensors.update(str(item) for item in state.get("unknown_sensors") or [])
        for name in ("plan", "chosen", "rejected", "expected_plan"):
            plan = record.get(name)
            if isinstance(plan, Mapping):
                for item in plan.get("skills") or []:
                    if isinstance(item, Mapping):
                        skill = item.get("name") or item.get("skill")
                        if skill:
                            skills.add(str(skill))
                    else:
                        skills.add(str(item))
        metadata = record.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("skill"):
            skills.add(str(metadata["skill"]))
    return {
        "skills": sorted(skills),
        "categories": sorted(categories),
        "lifecycle_states": sorted(lifecycle_states),
        "unknown_sensors": sorted(unknown_sensors),
        "input_sources": sorted(input_sources),
    }


def build_manifest(
    dataset_type: str,
    records: Iterable[Mapping[str, Any]],
    files: Iterable[str | Path],
    *,
    schema_version: int = PLANNER_DATA_SCHEMA_VERSION,
) -> dict[str, Any]:
    rows = [dict(record) for record in records]
    file_paths = [Path(path) for path in files]
    split_counts = Counter(str(record.get("split") or "unsplit") for record in rows)
    eligible = sum(bool(record.get("eligible", record.get("training_eligible", True))) for record in rows)
    return {
        "schema_version": schema_version,
        "dataset_type": dataset_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "total": len(rows),
            "by_split": dict(sorted(split_counts.items())),
        },
        "hashes": {
            str(path): sha256_file(path)
            for path in sorted(file_paths, key=lambda item: str(item))
            if path.exists()
        },
        "coverage": _coverage(rows),
        "eligibility": {
            "eligible": eligible,
            "ineligible": len(rows) - eligible,
        },
    }


def write_manifest(
    path: str | Path,
    dataset_type: str,
    records: Iterable[Mapping[str, Any]],
    files: Iterable[str | Path],
) -> dict[str, Any]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(dataset_type, records, files)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["build_manifest", "sha256_file", "write_manifest"]
