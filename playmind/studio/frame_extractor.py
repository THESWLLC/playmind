"""Offline frame extraction strategies backed by FFmpeg."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from playmind.studio.media_probe import MediaToolUnavailableError
from playmind.studio.project_store import DEFAULT_PROJECTS_ROOT, ProjectStore, utc_now
from playmind.studio.provenance import SOURCE_SYNTHETIC


def _source_for(project: Mapping[str, Any]) -> Path:
    media = project.get("media")
    if not isinstance(media, Mapping) or not media.get("stored_path"):
        raise ValueError("project has no imported media")
    source = Path(str(media["stored_path"]))
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            "FFmpeg frame extraction failed: "
            + (completed.stderr.strip() or "unknown error")
        )


def _timestamps_for_overview(duration: float | None, interval: float) -> list[float]:
    if interval <= 0:
        raise ValueError("interval_seconds must be positive")
    if duration is None or duration <= 0:
        return [0.0]
    count = max(1, int(math.floor(duration / interval)) + 1)
    return [min(duration, index * interval) for index in range(count)]


def _timestamps_for_ranges(
    ranges: Iterable[Sequence[float]], interval: float
) -> list[float]:
    timestamps: set[float] = set()
    for value in ranges:
        if len(value) != 2:
            raise ValueError("each manual range must contain start and end")
        start, end = float(value[0]), float(value[1])
        if start < 0 or end < start:
            raise ValueError("manual ranges require 0 <= start <= end")
        current = start
        while current <= end + 1e-9:
            timestamps.add(round(current, 6))
            current += interval
    return sorted(timestamps)


def _extract_at_timestamps(
    executable: str,
    source: Path,
    destination: Path,
    timestamps: Sequence[float],
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for index, timestamp in enumerate(timestamps, 1):
        path = destination / f"frame_{index:06d}.png"
        _run(
            [
                executable,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-y",
                str(path),
            ]
        )
        if not path.is_file():
            raise RuntimeError(f"FFmpeg did not produce expected frame: {path}")
        frames.append(
            {
                "frame_id": index,
                "timestamp": timestamp,
                "path": str(path.resolve()),
            }
        )
    return frames


def extract_frames(
    project_id: str,
    *,
    projects_root: str | Path = DEFAULT_PROJECTS_ROOT,
    strategy: str = "overview",
    interval_seconds: float = 10.0,
    ranges: Iterable[Sequence[float]] | None = None,
    timestamps: Iterable[float] | None = None,
    ffmpeg: str | None = None,
) -> dict[str, Any]:
    """Extract a re-runnable frame set without reading any live screen."""

    executable = ffmpeg or shutil.which("ffmpeg")
    if not executable:
        raise MediaToolUnavailableError("ffmpeg")
    if strategy not in {"overview", "change_aware", "keyframes", "manual"}:
        raise ValueError("unknown extraction strategy")
    store = ProjectStore(projects_root)
    project = store.load_project(project_id)
    source = _source_for(project)
    frame_set_id = f"{strategy}-{len(project.get('frame_sets') or []) + 1:03d}"
    destination = store.project_dir(project_id) / "frames" / frame_set_id
    destination.mkdir(parents=True, exist_ok=False)
    media = project.get("media") if isinstance(project.get("media"), Mapping) else {}
    if timestamps is not None:
        selected = sorted({max(0.0, float(value)) for value in timestamps})
    elif strategy == "manual":
        selected = _timestamps_for_ranges(ranges or [], interval_seconds)
        if not selected:
            raise ValueError("manual extraction requires at least one range")
    elif strategy == "change_aware":
        # Milestone stub: denser uniform candidates for later visual-delta ranking.
        selected = _timestamps_for_overview(
            float(media["duration"]) if media.get("duration") is not None else None,
            max(0.1, interval_seconds / 2.0),
        )
    elif strategy == "overview":
        selected = _timestamps_for_overview(
            float(media["duration"]) if media.get("duration") is not None else None,
            interval_seconds,
        )
    else:
        selected = []

    if strategy == "keyframes" and timestamps is None:
        output = destination / "frame_%06d.png"
        _run(
            [
                executable,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-skip_frame",
                "nokey",
                "-i",
                str(source),
                "-vsync",
                "vfr",
                "-y",
                str(output),
            ]
        )
        frames = [
            {"frame_id": index, "timestamp": None, "path": str(path.resolve())}
            for index, path in enumerate(sorted(destination.glob("*.png")), 1)
        ]
    else:
        frames = _extract_at_timestamps(executable, source, destination, selected)
    result = {
        "frame_set_id": frame_set_id,
        "strategy": strategy,
        "interval_seconds": interval_seconds,
        "change_aware_stub": strategy == "change_aware",
        "created_at": utc_now(),
        "frames": frames,
    }
    (destination / "frames.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    project.setdefault("frame_sets", []).append(
        {
            key: value
            for key, value in result.items()
            if key != "frames"
        }
        | {"frame_count": len(frames), "manifest": str(destination / "frames.json")}
    )
    store.save_project(project)
    return result


def create_tiny_synthetic_fixture(
    projects_root: str | Path,
    *,
    project_id: str = "synthetic-fixture",
    frame_count: int = 3,
) -> dict[str, Any] | None:
    """Create a tiny no-FFmpeg project fixture, or return None without Pillow."""

    try:
        from PIL import Image, ImageDraw  # type: ignore
    except ImportError:
        return None
    store = ProjectStore(projects_root)
    project = store.create_project(
        project_id=project_id,
        name="Synthetic Studio fixture",
        provenance={
            "source_type": SOURCE_SYNTHETIC,
            "rights_confirmed": True,
            "license_confirmed": True,
        },
    )
    destination = store.project_dir(project_id) / "frames" / "synthetic-001"
    destination.mkdir(parents=True)
    frames: list[dict[str, Any]] = []
    for index in range(max(1, frame_count)):
        path = destination / f"frame_{index + 1:06d}.png"
        image = Image.new("RGB", (160, 90), (20 + index * 20, 30, 50))
        draw = ImageDraw.Draw(image)
        draw.rectangle((5, 5, 105 - index * 20, 14), fill=(190, 35, 35))
        draw.text((8, 40), f"synthetic {index}", fill=(255, 255, 255))
        image.save(path)
        frames.append(
            {
                "frame_id": index + 1,
                "timestamp": float(index),
                "path": str(path.resolve()),
            }
        )
    result = {
        "frame_set_id": "synthetic-001",
        "strategy": "synthetic",
        "created_at": utc_now(),
        "frames": frames,
    }
    manifest = destination / "frames.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    project["media"] = {
        "synthetic": True,
        "duration": float(max(0, frame_count - 1)),
        "width": 160,
        "height": 90,
        "fps": 1.0,
        "has_audio": False,
        "sha256": "synthetic",
    }
    project["frame_sets"] = [
        {
            "frame_set_id": "synthetic-001",
            "strategy": "synthetic",
            "frame_count": len(frames),
            "manifest": str(manifest),
        }
    ]
    store.save_project(project)
    return result


__all__ = ["create_tiny_synthetic_fixture", "extract_frames"]
