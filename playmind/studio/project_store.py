"""Filesystem-backed Studio project persistence."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PROJECTS_ROOT = Path("data/playmind/studio/projects")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


class ProjectStore:
    def __init__(self, root: str | Path = DEFAULT_PROJECTS_ROOT) -> None:
        self.root = Path(root)

    @staticmethod
    def validate_project_id(project_id: str) -> str:
        value = str(project_id).strip()
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("project_id must be a safe non-empty filename")
        return value

    def project_dir(self, project_id: str) -> Path:
        return self.root / self.validate_project_id(project_id)

    def create_project(
        self,
        *,
        project_id: str | None = None,
        name: str | None = None,
        profile: str = "retail_wow_offline_only",
        provenance: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        identifier = self.validate_project_id(project_id or uuid.uuid4().hex)
        directory = self.project_dir(identifier)
        directory.mkdir(parents=True, exist_ok=False)
        for child in ("frames", "source", "exports"):
            (directory / child).mkdir()
        now = utc_now()
        project: dict[str, Any] = {
            "schema_version": 1,
            "project_id": identifier,
            "name": str(name or identifier),
            "profile": profile,
            "created_at": now,
            "updated_at": now,
            "provenance": dict(provenance or {}),
            "media": {},
            "frame_sets": [],
            "metadata": dict(metadata or {}),
        }
        self.save_project(project)
        self.save_annotations(identifier, [])
        self.save_analysis(identifier, [])
        return project

    create = create_project

    def load_project(self, project_id: str) -> dict[str, Any]:
        path = self.project_dir(project_id) / "project.json"
        if not path.is_file():
            raise KeyError(f"unknown Studio project: {project_id!r}")
        value = _read_json(path, {})
        if not isinstance(value, dict):
            raise ValueError(f"invalid project metadata: {path}")
        return value

    load = load_project

    def save_project(self, project: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(project)
        identifier = self.validate_project_id(str(value.get("project_id") or ""))
        directory = self.project_dir(identifier)
        directory.mkdir(parents=True, exist_ok=True)
        value["project_id"] = identifier
        value.setdefault("created_at", utc_now())
        value["updated_at"] = utc_now()
        _write_json(directory / "project.json", value)
        return value

    save = save_project

    def update_project(
        self, project_id: str, changes: Mapping[str, Any]
    ) -> dict[str, Any]:
        project = self.load_project(project_id)
        changes = dict(changes)
        if "project_id" in changes and changes["project_id"] != project_id:
            raise ValueError("project_id is immutable")
        project.update(changes)
        return self.save_project(project)

    update = update_project

    def delete_project(self, project_id: str) -> None:
        """Delete only an empty project; media removal must be explicit."""

        directory = self.project_dir(project_id)
        if not directory.exists():
            return
        files = [path for path in directory.rglob("*") if path.is_file()]
        permitted = {"project.json", "annotations.json", "analysis.json"}
        if any(path.name not in permitted for path in files):
            raise ValueError("project contains media; explicit filesystem removal required")
        for path in files:
            path.unlink()
        for path in sorted(directory.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
        directory.rmdir()

    delete = delete_project

    def list_projects(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        projects: list[dict[str, Any]] = []
        for path in self.root.iterdir():
            if not path.is_dir() or not (path / "project.json").is_file():
                continue
            try:
                projects.append(self.load_project(path.name))
            except (ValueError, json.JSONDecodeError):
                continue
        return sorted(
            projects,
            key=lambda item: str(item.get("updated_at") or ""),
            reverse=True,
        )

    list = list_projects

    def load_annotations(self, project_id: str) -> list[dict[str, Any]]:
        value = _read_json(self.project_dir(project_id) / "annotations.json", [])
        if not isinstance(value, list):
            raise ValueError("annotations.json must contain a list")
        return [dict(item) for item in value if isinstance(item, Mapping)]

    def save_annotations(
        self, project_id: str, annotations: list[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        self.project_dir(project_id).mkdir(parents=True, exist_ok=True)
        value = [dict(item) for item in annotations]
        _write_json(self.project_dir(project_id) / "annotations.json", value)
        return value

    def load_analysis(self, project_id: str) -> list[dict[str, Any]]:
        value = _read_json(self.project_dir(project_id) / "analysis.json", [])
        if not isinstance(value, list):
            raise ValueError("analysis.json must contain a list")
        return [dict(item) for item in value if isinstance(item, Mapping)]

    def save_analysis(
        self, project_id: str, analysis: list[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        self.project_dir(project_id).mkdir(parents=True, exist_ok=True)
        value = [dict(item) for item in analysis]
        _write_json(self.project_dir(project_id) / "analysis.json", value)
        return value


__all__ = ["DEFAULT_PROJECTS_ROOT", "ProjectStore", "utc_now"]
