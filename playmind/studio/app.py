"""Stateful backend shared by future Studio CLI and GUI frontends."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from playmind.studio.annotations import AnnotationStore, TimelineSegment
from playmind.studio.dataset_bridge import export_reviewed_projects
from playmind.studio.eval_index import write_index
from playmind.studio.frame_extractor import extract_frames
from playmind.studio.offline_analysis import analyze_project
from playmind.studio.project_store import DEFAULT_PROJECTS_ROOT, ProjectStore
from playmind.studio.safety import assert_studio_safe, studio_may_not_send_input
from playmind.studio.training_readiness import assess_training_readiness
from playmind.studio.transcripts import import_transcript, suggest_skill_sequences
from playmind.studio.video_import import import_video


class StudioState(str, Enum):
    READY = "ready"
    PROJECT_SELECTED = "project_selected"
    BUSY = "busy"
    ERROR = "error"


class StudioApp:
    """Offline-only coordinator; this class has no live or input methods."""

    def __init__(
        self,
        projects_root: str | Path = DEFAULT_PROJECTS_ROOT,
        *,
        data_root: str | Path = "data/playmind",
    ) -> None:
        assert_studio_safe()
        if not studio_may_not_send_input():
            raise RuntimeError("Studio no-input invariant failed")
        self.projects_root = Path(projects_root)
        self.data_root = Path(data_root)
        self.store = ProjectStore(self.projects_root)
        self.state = StudioState.READY
        self.current_project_id: str | None = None
        self.last_error: str = ""

    def _busy(self) -> None:
        if self.state == StudioState.BUSY:
            raise RuntimeError("Studio is already processing an offline task")
        self.state = StudioState.BUSY
        self.last_error = ""

    def _done(self, project_id: str | None = None) -> None:
        if project_id is not None:
            self.current_project_id = project_id
        self.state = (
            StudioState.PROJECT_SELECTED
            if self.current_project_id
            else StudioState.READY
        )

    def _failed(self, exc: Exception) -> None:
        self.last_error = f"{type(exc).__name__}: {exc}"
        self.state = StudioState.ERROR

    def select_project(self, project_id: str) -> dict[str, Any]:
        project = self.store.load_project(project_id)
        self.current_project_id = project_id
        self.state = StudioState.PROJECT_SELECTED
        self.last_error = ""
        return project

    def list_projects(self) -> list[dict[str, Any]]:
        return self.store.list_projects()

    def import_video(self, source: str | Path, **kwargs: Any) -> dict[str, Any]:
        self._busy()
        try:
            project = import_video(
                source, projects_root=self.projects_root, **kwargs
            )
        except Exception as exc:
            self._failed(exc)
            raise
        self._done(str(project["project_id"]))
        return project

    def extract_frames(self, strategy: str = "overview", **kwargs: Any) -> dict[str, Any]:
        project_id = self._require_project()
        self._busy()
        try:
            result = extract_frames(
                project_id,
                projects_root=self.projects_root,
                strategy=strategy,
                **kwargs,
            )
        except Exception as exc:
            self._failed(exc)
            raise
        self._done(project_id)
        return result

    def analyze(self, **kwargs: Any) -> list[dict[str, Any]]:
        project_id = self._require_project()
        self._busy()
        try:
            result = analyze_project(
                project_id, projects_root=self.projects_root, **kwargs
            )
        except Exception as exc:
            self._failed(exc)
            raise
        self._done(project_id)
        return result

    def annotations(self) -> AnnotationStore:
        return AnnotationStore(self._require_project(), self.projects_root)

    def add_annotation(
        self, segment: TimelineSegment | dict[str, Any]
    ) -> TimelineSegment:
        return self.annotations().add(segment)

    def import_transcript(self, path: str | Path) -> list[dict[str, Any]]:
        return import_transcript(
            path,
            project_id=self._require_project(),
            projects_root=self.projects_root,
        )

    def transcript_suggestions(self) -> list[dict[str, Any]]:
        path = self.store.project_dir(self._require_project()) / "transcript.json"
        if not path.exists():
            return []
        cues = json.loads(path.read_text(encoding="utf-8"))
        return suggest_skill_sequences(cues)

    def export_datasets(self, project_ids: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        selected = project_ids or [self._require_project()]
        return export_reviewed_projects(
            selected,
            projects_root=self.projects_root,
            planner_root=self.data_root / "planner",
            vision_root=self.data_root / "vision",
            **kwargs,
        )

    def readiness(
        self,
        project_ids: list[str] | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        selected = project_ids or (
            [self.current_project_id] if self.current_project_id else []
        )
        reviewed = 0
        preferences = 0
        provenance_ok = True
        license_confirmed = True
        for project_id in selected:
            if project_id is None:
                continue
            project = self.store.load_project(project_id)
            provenance = project.get("provenance") or {}
            provenance_ok = provenance_ok and bool(
                provenance.get("rights_confirmed")
                or provenance.get("source_type") == "synthetic"
            )
            license_confirmed = license_confirmed and bool(
                provenance.get("license_confirmed")
                or provenance.get("source_type") in {"synthetic", "user_owned_recording"}
            )
            reviewed += sum(
                1
                for item in self.store.load_annotations(project_id)
                if item.get("review_status") == "reviewed"
                and item.get("category") not in {"unknown", "unusable"}
            )
            corrections_path = self.store.project_dir(project_id) / "corrections.json"
            if corrections_path.exists():
                corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
                preferences += sum(
                    1
                    for item in corrections
                    if item.get("review_status") == "reviewed"
                )
        frozen = len(
            list((self.data_root / "planner" / "evaluation").glob("*_v*.json"))
        )
        options = {
            "reviewed_examples": reviewed,
            "preference_examples": preferences,
            "frozen_real_benchmarks": frozen,
            "provenance_eligible": provenance_ok,
            "license_confirmed": license_confirmed,
        }
        options.update(overrides)
        return assess_training_readiness(**options).to_dict()

    def evaluations(self) -> dict[str, Any]:
        return write_index(self.data_root / "planner" / "evaluation")

    def _require_project(self) -> str:
        if not self.current_project_id:
            raise RuntimeError("select or import a Studio project first")
        return self.current_project_id


__all__ = ["StudioApp", "StudioState"]
