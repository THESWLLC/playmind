"""Offline media inspection through ffprobe."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm"})
FFMPEG_INSTALL_INSTRUCTIONS = (
    "Install FFmpeg (including ffprobe), then ensure both executables are on PATH. "
    "Windows: winget install Gyan.FFmpeg; macOS: brew install ffmpeg; "
    "Debian/Ubuntu: sudo apt install ffmpeg."
)


class MediaToolUnavailableError(RuntimeError):
    def __init__(self, tool: str) -> None:
        self.tool = tool
        self.instructions = FFMPEG_INSTALL_INSTRUCTIONS
        super().__init__(f"{tool} was not found. {self.instructions}")


class UnsupportedMediaError(ValueError):
    pass


@dataclass(frozen=True)
class MediaProbe:
    path: str
    sha256: str
    duration: float | None
    width: int | None
    height: int | None
    fps: float | None
    has_audio: bool
    format_name: str = ""
    size_bytes: int = 0

    @property
    def resolution(self) -> str | None:
        if self.width is None or self.height is None:
            return None
        return f"{self.width}x{self.height}"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["resolution"] = self.resolution
        return result


def find_media_tools() -> dict[str, str | None]:
    return {"ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe")}


def media_tools_status() -> dict[str, Any]:
    tools = find_media_tools()
    missing = sorted(name for name, path in tools.items() if path is None)
    return {
        "available": not missing,
        "tools": tools,
        "missing": missing,
        "instructions": "" if not missing else FFMPEG_INSTALL_INSTRUCTIONS,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fps(value: object) -> float | None:
    text = str(value or "")
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            result = float(numerator) / float(denominator)
        else:
            result = float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if result > 0 else None


def probe_media(path: str | Path, *, ffprobe: str | None = None) -> MediaProbe:
    source = Path(path)
    if source.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise UnsupportedMediaError(
            f"unsupported video extension {source.suffix!r}; expected "
            + ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        )
    if not source.is_file():
        raise FileNotFoundError(source)
    executable = ffprobe or shutil.which("ffprobe")
    if not executable:
        raise MediaToolUnavailableError("ffprobe")
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        message = completed.stderr.strip() or "ffprobe failed"
        raise RuntimeError(f"could not inspect {source}: {message}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") if isinstance(payload, dict) else []
    streams = streams if isinstance(streams, list) else []
    video = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        ),
        {},
    )
    if not video:
        raise ValueError(f"media has no video stream: {source}")
    format_data = payload.get("format") if isinstance(payload, dict) else {}
    format_data = format_data if isinstance(format_data, dict) else {}
    duration_value = video.get("duration", format_data.get("duration"))
    try:
        duration = float(duration_value)
    except (TypeError, ValueError):
        duration = None
    return MediaProbe(
        path=str(source.resolve()),
        sha256=sha256_file(source),
        duration=duration,
        width=int(video["width"]) if video.get("width") is not None else None,
        height=int(video["height"]) if video.get("height") is not None else None,
        fps=_fps(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        has_audio=any(
            isinstance(stream, dict) and stream.get("codec_type") == "audio"
            for stream in streams
        ),
        format_name=str(format_data.get("format_name") or ""),
        size_bytes=source.stat().st_size,
    )


__all__ = [
    "FFMPEG_INSTALL_INSTRUCTIONS",
    "MediaProbe",
    "MediaToolUnavailableError",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "UnsupportedMediaError",
    "find_media_tools",
    "media_tools_status",
    "probe_media",
    "sha256_file",
]
