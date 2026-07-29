"""Import user-authorized recordings into offline Studio projects."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from playmind.studio.media_probe import (
    MediaProbe,
    MediaToolUnavailableError,
    probe_media,
)
from playmind.studio.profiles import PROFILE_RETAIL_WOW_OFFLINE_ONLY
from playmind.studio.project_store import DEFAULT_PROJECTS_ROOT, ProjectStore
from playmind.studio.provenance import ProvenanceRecord, coerce_provenance


def import_video(
    source: str | Path,
    *,
    provenance: ProvenanceRecord | Mapping[str, Any],
    projects_root: str | Path = DEFAULT_PROJECTS_ROOT,
    project_id: str | None = None,
    name: str | None = None,
    mode: str = "copy",
    profile: str = PROFILE_RETAIL_WOW_OFFLINE_ONLY,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
) -> dict[str, Any]:
    """Probe and import a recording, failing before project creation on errors."""

    if mode not in {"copy", "reference"}:
        raise ValueError("mode must be 'copy' or 'reference'")
    if not (ffmpeg or shutil.which("ffmpeg")):
        raise MediaToolUnavailableError("ffmpeg")
    source_path = Path(source).expanduser().resolve()
    probe: MediaProbe = probe_media(source_path, ffprobe=ffprobe)
    rights = coerce_provenance(provenance)
    store = ProjectStore(projects_root)
    identifier = project_id or probe.sha256[:16]
    if (store.root / identifier).exists():
        suffix = 2
        while (store.root / f"{identifier}-{suffix}").exists():
            suffix += 1
        identifier = f"{identifier}-{suffix}"
    project = store.create_project(
        project_id=identifier,
        name=name or source_path.stem,
        profile=profile,
        provenance=rights.to_dict(),
    )
    directory = store.project_dir(identifier)
    if mode == "copy":
        destination = directory / "source" / ("recording" + source_path.suffix.lower())
        shutil.copy2(source_path, destination)
        stored_path = destination.resolve()
    else:
        stored_path = source_path
    project["media"] = {
        **probe.to_dict(),
        "original_path": str(source_path),
        "stored_path": str(stored_path),
        "storage_mode": mode,
    }
    return store.save_project(project)


class VideoImporter:
    def __init__(self, projects_root: str | Path = DEFAULT_PROJECTS_ROOT) -> None:
        self.projects_root = Path(projects_root)

    def import_video(self, source: str | Path, **kwargs: Any) -> dict[str, Any]:
        return import_video(source, projects_root=self.projects_root, **kwargs)

    import_file = import_video


__all__ = ["VideoImporter", "import_video"]
