"""Bridge reviewed Studio artifacts into planner and visual-state datasets."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from playmind.planner_data import export_preferences, export_sft
from playmind.studio.annotations import annotation_is_eligible
from playmind.studio.corrections import CorrectionStore, correction_records
from playmind.studio.project_store import DEFAULT_PROJECTS_ROOT, ProjectStore
from playmind.studio.provenance import is_training_eligible


DEFAULT_PLANNER_EXPORT_ROOT = Path("data/playmind/planner")
DEFAULT_VISION_ROOT = Path("data/playmind/vision")


def leakage_violations(
    records: Iterable[Mapping[str, Any]],
    *,
    split_key: str = "split",
) -> list[dict[str, Any]]:
    """Find a project or source represented in more than one data split."""

    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        split = str(record.get(split_key) or "")
        if not split:
            continue
        for key in ("project_id", "source_id", "source_sha256"):
            value = record.get(key)
            if value not in (None, ""):
                groups[(key, str(value))].add(split)
    return [
        {"group_type": key, "group": value, "splits": sorted(splits)}
        for (key, value), splits in sorted(groups.items())
        if len(splits) > 1
    ]


check_leakage = leakage_violations


def assert_no_leakage(records: Iterable[Mapping[str, Any]]) -> None:
    violations = leakage_violations(records)
    if violations:
        raise ValueError(f"dataset leakage detected: {violations}")


def _planner_state(
    analysis: list[dict[str, Any]], timestamp: float
) -> dict[str, Any]:
    by_name: dict[str, dict[str, Any]] = {}
    for item in analysis:
        item_timestamp = item.get("timestamp")
        if item_timestamp is None or abs(float(item_timestamp) - timestamp) > 2.0:
            continue
        current = by_name.get(str(item.get("name")))
        if current is None or abs(float(current.get("timestamp") or 0) - timestamp) > abs(
            float(item_timestamp) - timestamp
        ):
            by_name[str(item.get("name"))] = item
    sensors = {
        name: {
            "value": item.get("value"),
            "known": bool(item.get("known", item.get("value") is not None)),
            "confidence": item.get("confidence"),
        }
        for name, item in by_name.items()
    }
    return {
        "timestamp": timestamp,
        "sensors": sensors,
        "unknown_sensors": sorted(
            name for name, value in sensors.items() if not value["known"]
        ),
    }


def _read_corrections(store: ProjectStore, project_id: str) -> list[dict[str, Any]]:
    path = store.project_dir(project_id) / "corrections.json"
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    return [dict(item) for item in value if isinstance(item, Mapping)]


def export_reviewed_projects(
    project_ids: Iterable[str],
    *,
    projects_root: str | Path = DEFAULT_PROJECTS_ROOT,
    planner_root: str | Path = DEFAULT_PLANNER_EXPORT_ROOT,
    vision_root: str | Path = DEFAULT_VISION_ROOT,
    seed: int = 0,
    allow_unverified_private: bool = False,
) -> dict[str, Any]:
    store = ProjectStore(projects_root)
    sft_rows: list[dict[str, Any]] = []
    preference_rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    rejected_projects: list[dict[str, str]] = []
    for project_id in project_ids:
        project = store.load_project(project_id)
        provenance = project.get("provenance")
        if not isinstance(provenance, Mapping) or not is_training_eligible(
            provenance, allow_unverified_private=allow_unverified_private
        ):
            rejected_projects.append(
                {"project_id": project_id, "reason": "ineligible provenance"}
            )
            continue
        media = project.get("media") if isinstance(project.get("media"), Mapping) else {}
        source_hash = str(media.get("sha256") or project_id)
        episode_id = f"studio-source-{source_hash}"
        analysis = store.load_analysis(project_id)
        annotations = store.load_annotations(project_id)
        for annotation in annotations:
            if not annotation_is_eligible(annotation):
                continue
            if annotation.get("segment_type") != "skill":
                continue
            timestamp = float(annotation.get("start") or 0.0)
            category = str(annotation.get("category") or "")
            common = {
                "episode_id": episode_id,
                "project_id": project_id,
                "source_sha256": source_hash,
                "timestamp": timestamp,
                "planner_state": _planner_state(analysis, timestamp),
                "input_source": "studio_reviewed_recording",
                "training_eligible": True,
            }
            sft_rows.append({**common, "plan": {"skills": [category]}})
        for correction in _read_corrections(store, project_id):
            rows = correction_records(correction, provenance=provenance)
            rows["sft"].update(
                {"episode_id": episode_id, "source_sha256": source_hash}
            )
            rows["preference"].update(
                {"episode_id": episode_id, "source_sha256": source_hash}
            )
            if rows["sft"]["training_eligible"]:
                sft_rows.append(rows["sft"])
                preference_rows.append(rows["preference"])
        reviewed_analysis = [
            item for item in analysis if item.get("review_status") == "reviewed"
        ]
        by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in reviewed_analysis:
            by_frame[str(item.get("source_frame") or "")].append(item)
        for source_frame, detections in by_frame.items():
            visual_rows.append(
                {
                    "project_id": project_id,
                    "source_sha256": source_hash,
                    "source_frame": source_frame,
                    "timestamp": detections[0].get("timestamp"),
                    "detections": detections,
                    "provenance": dict(provenance),
                    "reviewed": True,
                }
            )

    planner_path = Path(planner_root)
    sft_manifest = export_sft(
        sft_rows,
        planner_path / "sft" / "studio",
        manifest_dir=planner_path / "manifests" / "studio",
        seed=seed,
    )
    preference_manifest = export_preferences(
        preference_rows,
        planner_path / "preferences" / "studio",
        manifest_dir=planner_path / "manifests" / "studio",
        seed=seed,
    )
    visual_path = Path(vision_root) / "studio_visual_states.jsonl"
    visual_path.parent.mkdir(parents=True, exist_ok=True)
    with visual_path.open("w", encoding="utf-8") as handle:
        for row in visual_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "sft": sft_manifest,
        "preferences": preference_manifest,
        "visual_state_path": str(visual_path),
        "counts": {
            "sft": len(sft_rows),
            "preferences": len(preference_rows),
            "visual_states": len(visual_rows),
        },
        "rejected_projects": rejected_projects,
        "leakage": leakage_violations(sft_rows + preference_rows),
    }


export_projects = export_reviewed_projects


__all__ = [
    "DEFAULT_PLANNER_EXPORT_ROOT",
    "DEFAULT_VISION_ROOT",
    "assert_no_leakage",
    "check_leakage",
    "export_projects",
    "export_reviewed_projects",
    "leakage_violations",
]
