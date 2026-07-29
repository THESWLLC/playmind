"""Reviewed timeline annotations for offline recordings."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from playmind.skills import list_skills
from playmind.studio.project_store import DEFAULT_PROJECTS_ROOT, ProjectStore, utc_now


SEGMENT_TYPES = frozenset({"skill", "goal", "outcome"})
OUTCOME_CATEGORIES = frozenset({"success", "failure", "unknown", "unusable"})
SPECIAL_CATEGORIES = frozenset({"unknown", "unusable"})
REVIEW_STATUSES = frozenset({"suggested", "reviewed", "rejected"})


def annotation_categories() -> frozenset[str]:
    return frozenset(list_skills()) | OUTCOME_CATEGORIES


@dataclass
class TimelineSegment:
    start: float
    end: float
    category: str
    segment_type: str = "skill"
    review_status: str = "suggested"
    segment_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    label: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.start = float(self.start)
        self.end = float(self.end)
        if self.start < 0 or self.end < self.start:
            raise ValueError("timeline segment requires 0 <= start <= end")
        if self.segment_type not in SEGMENT_TYPES:
            raise ValueError(f"segment_type must be one of {sorted(SEGMENT_TYPES)}")
        if self.category not in annotation_categories():
            raise ValueError(f"unknown annotation category: {self.category!r}")
        if self.review_status not in REVIEW_STATUSES:
            raise ValueError(f"review_status must be one of {sorted(REVIEW_STATUSES)}")

    @property
    def training_eligible(self) -> bool:
        return annotation_is_eligible(self)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["training_eligible"] = self.training_eligible
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TimelineSegment:
        data = dict(value)
        data.pop("training_eligible", None)
        return cls(**data)


def annotation_is_eligible(
    segment: TimelineSegment | Mapping[str, Any],
    *,
    allow_suggested: bool = False,
) -> bool:
    item = (
        segment
        if isinstance(segment, TimelineSegment)
        else TimelineSegment.from_dict(segment)
    )
    reviewed = item.review_status == "reviewed" or (
        allow_suggested and item.review_status == "suggested"
    )
    return (
        reviewed
        and item.category not in SPECIAL_CATEGORIES
        and item.review_status != "rejected"
    )


class AnnotationStore:
    def __init__(
        self,
        project_id: str,
        projects_root: str | Path = DEFAULT_PROJECTS_ROOT,
    ) -> None:
        self.project_id = project_id
        self.projects = ProjectStore(projects_root)
        self._undo: list[list[dict[str, Any]]] = []

    def list(self) -> list[TimelineSegment]:
        return [
            TimelineSegment.from_dict(item)
            for item in self.projects.load_annotations(self.project_id)
        ]

    def _save(self, segments: list[TimelineSegment]) -> list[TimelineSegment]:
        self.projects.save_annotations(
            self.project_id, [segment.to_dict() for segment in segments]
        )
        return segments

    def add(self, segment: TimelineSegment | Mapping[str, Any]) -> TimelineSegment:
        item = (
            segment
            if isinstance(segment, TimelineSegment)
            else TimelineSegment.from_dict(segment)
        )
        current = self.list()
        self._undo.append([value.to_dict() for value in current])
        current.append(item)
        self._save(current)
        return item

    add_segment = add

    def update(self, segment_id: str, **changes: Any) -> TimelineSegment:
        current = self.list()
        self._undo.append([value.to_dict() for value in current])
        for index, segment in enumerate(current):
            if segment.segment_id != segment_id:
                continue
            value = segment.to_dict()
            value.pop("training_eligible", None)
            value.update(changes)
            current[index] = TimelineSegment.from_dict(value)
            self._save(current)
            return current[index]
        self._undo.pop()
        raise KeyError(f"unknown timeline segment: {segment_id!r}")

    update_segment = update

    def review(
        self, segment_id: str, *, accepted: bool = True
    ) -> TimelineSegment:
        return self.update(
            segment_id, review_status="reviewed" if accepted else "rejected"
        )

    def remove(self, segment_id: str) -> None:
        current = self.list()
        remaining = [item for item in current if item.segment_id != segment_id]
        if len(remaining) == len(current):
            raise KeyError(f"unknown timeline segment: {segment_id!r}")
        self._undo.append([value.to_dict() for value in current])
        self._save(remaining)

    def undo(self) -> list[TimelineSegment]:
        if not self._undo:
            return self.list()
        previous = [TimelineSegment.from_dict(item) for item in self._undo.pop()]
        return self._save(previous)


__all__ = [
    "AnnotationStore",
    "OUTCOME_CATEGORIES",
    "REVIEW_STATUSES",
    "SEGMENT_TYPES",
    "SPECIAL_CATEGORIES",
    "TimelineSegment",
    "annotation_categories",
    "annotation_is_eligible",
]
