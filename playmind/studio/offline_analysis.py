"""Re-runnable analysis of already-extracted frame files."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from playmind.observations import Observation
from playmind.studio.project_store import DEFAULT_PROJECTS_ROOT, ProjectStore, utc_now
from playmind.vision import detect_death_dialog, detect_target_bar, read_frame


def _frame_rows(project_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in sorted((project_dir / "frames").glob("*/frames.json")):
        value = json.loads(manifest.read_text(encoding="utf-8"))
        frames = value.get("frames") if isinstance(value, Mapping) else None
        if isinstance(frames, list):
            rows.extend(dict(item) for item in frames if isinstance(item, Mapping))
    return sorted(
        rows,
        key=lambda item: (
            item.get("timestamp") is None,
            float(item.get("timestamp") or 0.0),
            str(item.get("path") or ""),
        ),
    )


def _detection(
    *,
    timestamp: float | None,
    name: str,
    value: Any,
    confidence: float | None,
    detector: str,
    frame: str,
) -> dict[str, Any]:
    return {
        "detection_id": uuid.uuid4().hex,
        "timestamp": timestamp,
        "name": name,
        "sensor": name,
        "value": value,
        "known": value is not None,
        "confidence": confidence,
        "detector": detector,
        "source_frame": frame,
        "review_status": "suggested",
    }


def analyze_project(
    project_id: str,
    *,
    projects_root: str | Path = DEFAULT_PROJECTS_ROOT,
    do_ocr: bool = False,
) -> list[dict[str, Any]]:
    store = ProjectStore(projects_root)
    project_dir = store.project_dir(project_id)
    store.load_project(project_id)
    old = store.load_analysis(project_id)
    reviewed = {
        (item.get("source_frame"), item.get("name")): item
        for item in old
        if item.get("review_status") == "reviewed"
    }
    detections: list[dict[str, Any]] = []
    for frame in _frame_rows(project_dir):
        path = Path(str(frame.get("path") or ""))
        timestamp = (
            float(frame["timestamp"]) if frame.get("timestamp") is not None else None
        )
        reading = read_frame(path, do_ocr=do_ocr)
        observation = Observation.from_legacy_dict(
            {
                "timestamp": timestamp or 0.0,
                **reading.to_obs_patch(),
            }
        )
        values = (
            (
                "player_hp",
                observation.player_hp,
                observation.player_hp_confidence,
                "vision.read_frame",
            ),
            (
                "objective_text",
                reading.quest_text,
                None,
                "vision.read_frame",
            ),
        )
        for name, value, confidence, detector in values:
            detections.append(
                _detection(
                    timestamp=timestamp,
                    name=name,
                    value=value,
                    confidence=confidence,
                    detector=detector,
                    frame=str(path),
                )
            )
        try:
            if "pillow_unavailable_or_failed" in (reading.notes or []):
                raise RuntimeError("image detector unavailable")
            is_dead, is_ghost = detect_death_dialog(path)
            has_target, target_confidence = detect_target_bar(path)
        except Exception:
            is_dead = is_ghost = has_target = None
            target_confidence = None
        for name, value, confidence, detector in (
            ("is_dead", is_dead, None, "vision.detect_death_dialog"),
            ("is_ghost", is_ghost, None, "vision.detect_death_dialog"),
            (
                "has_target",
                has_target,
                target_confidence,
                "vision.detect_target_bar",
            ),
        ):
            detections.append(
                _detection(
                    timestamp=timestamp,
                    name=name,
                    value=value,
                    confidence=confidence,
                    detector=detector,
                    frame=str(path),
                )
            )
    for index, item in enumerate(detections):
        prior = reviewed.get((item["source_frame"], item["name"]))
        if prior is not None:
            item = dict(prior)
            item["timestamp"] = detections[index]["timestamp"]
            detections[index] = item
    store.save_analysis(project_id, detections)
    project = store.load_project(project_id)
    project["last_analysis"] = {
        "run_at": utc_now(),
        "detection_count": len(detections),
        "ocr": bool(do_ocr),
    }
    store.save_project(project)
    return detections


class OfflineAnalyzer:
    def __init__(self, projects_root: str | Path = DEFAULT_PROJECTS_ROOT) -> None:
        self.projects_root = Path(projects_root)

    def analyze(self, project_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        return analyze_project(project_id, projects_root=self.projects_root, **kwargs)


__all__ = ["OfflineAnalyzer", "analyze_project"]
