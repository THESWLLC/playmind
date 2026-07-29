"""Import local transcript files and produce review-required skill suggestions."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from playmind.studio.project_store import DEFAULT_PROJECTS_ROOT, ProjectStore


SUPPORTED_TRANSCRIPT_EXTENSIONS = frozenset({".srt", ".vtt", ".txt"})
_TIMING = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{3}|\d{1,2}:\d{2}[,.]\d{3})"
    r"\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{3}|\d{1,2}:\d{2}[,.]\d{3})"
)
_KEYWORDS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("dead", "died", "release spirit"), ("death_recovery", "ghost_runback")),
    (("stuck", "not moving"), ("unstuck",)),
    (("heal", "eat", "recover"), ("recover_health",)),
    (("loot",), ("loot_target",)),
    (("attack", "fight", "combat"), ("acquire_target", "engage_target", "basic_combat_rotation")),
    (("target",), ("acquire_target", "validate_target")),
    (("quest", "objective", "travel", "go to"), ("explore",)),
    (("interact", "talk to", "speak to"), ("interact",)),
    (("wait", "loading"), ("wait",)),
)


@dataclass(frozen=True)
class TranscriptCue:
    start: float
    end: float | None
    text: str
    cue_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _seconds(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    hours, minutes, seconds = parts
    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)


def _parse_timed(text: str) -> list[TranscriptCue]:
    lines = text.replace("\ufeff", "").splitlines()
    cues: list[TranscriptCue] = []
    index = 0
    while index < len(lines):
        match = _TIMING.search(lines[index])
        if match is None:
            index += 1
            continue
        start = _seconds(match.group("start"))
        end = _seconds(match.group("end"))
        index += 1
        content: list[str] = []
        while index < len(lines) and lines[index].strip():
            content.append(lines[index].strip())
            index += 1
        value = " ".join(content).strip()
        if value:
            cues.append(TranscriptCue(start, end, value, uuid.uuid4().hex))
        index += 1
    return cues


def import_transcript(
    path: str | Path,
    *,
    project_id: str | None = None,
    projects_root: str | Path = DEFAULT_PROJECTS_ROOT,
) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_TRANSCRIPT_EXTENSIONS:
        raise ValueError("transcript must be SRT, VTT, or TXT")
    text = source.read_text(encoding="utf-8-sig", errors="replace")
    if suffix == ".txt":
        cues = [
            TranscriptCue(0.0, None, line.strip(), uuid.uuid4().hex)
            for line in text.splitlines()
            if line.strip()
        ]
    else:
        cues = _parse_timed(text)
    result = [cue.to_dict() for cue in cues]
    if project_id is not None:
        project_dir = ProjectStore(projects_root).project_dir(project_id)
        ProjectStore(projects_root).load_project(project_id)
        destination = project_dir / "transcript.json"
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def suggest_skill_sequences(
    cues: list[TranscriptCue | dict[str, Any]],
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    for cue_value in cues:
        cue = (
            cue_value
            if isinstance(cue_value, TranscriptCue)
            else TranscriptCue(**cue_value)
        )
        lowered = cue.text.casefold()
        for keywords, skills in _KEYWORDS:
            matched = [keyword for keyword in keywords if keyword in lowered]
            if not matched:
                continue
            suggestions.append(
                {
                    "suggestion_id": uuid.uuid4().hex,
                    "cue_id": cue.cue_id,
                    "start": cue.start,
                    "end": cue.end,
                    "skills": list(skills),
                    "matched_keywords": matched,
                    "review_status": "suggested",
                    "training_eligible": False,
                }
            )
            break
    return suggestions


__all__ = [
    "SUPPORTED_TRANSCRIPT_EXTENSIONS",
    "TranscriptCue",
    "import_transcript",
    "suggest_skill_sequences",
]
