"""Human corrections of candidate plans for SFT and preference learning."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from playmind.planner_data.schemas import normalize_plan
from playmind.studio.project_store import DEFAULT_PROJECTS_ROOT, ProjectStore, utc_now
from playmind.studio.provenance import is_training_eligible


@dataclass
class PlanCorrection:
    project_id: str
    planner_state: dict[str, Any]
    candidate_plan: dict[str, Any]
    corrected_plan: dict[str, Any]
    timestamp: float | None = None
    review_status: str = "suggested"
    correction_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    notes: str = ""
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.planner_state = dict(self.planner_state)
        self.candidate_plan = normalize_plan(self.candidate_plan)
        self.corrected_plan = normalize_plan(self.corrected_plan)
        if self.review_status not in {"suggested", "reviewed", "rejected"}:
            raise ValueError("invalid correction review_status")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PlanCorrection:
        return cls(**dict(value))


def correction_is_eligible(
    correction: PlanCorrection | Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> bool:
    item = (
        correction
        if isinstance(correction, PlanCorrection)
        else PlanCorrection.from_dict(correction)
    )
    return (
        item.review_status == "reviewed"
        and bool(item.corrected_plan.get("skills"))
        and bool(item.candidate_plan.get("skills"))
        and item.corrected_plan != item.candidate_plan
        and (provenance is None or is_training_eligible(provenance))
    )


def correction_records(
    correction: PlanCorrection | Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    item = (
        correction
        if isinstance(correction, PlanCorrection)
        else PlanCorrection.from_dict(correction)
    )
    eligible = correction_is_eligible(item, provenance=provenance)
    common = {
        "example_id": item.correction_id,
        "episode_id": item.project_id,
        "project_id": item.project_id,
        "timestamp": item.timestamp,
        "planner_state": item.planner_state,
        "input_source": "human_correction",
        "training_eligible": eligible,
    }
    return {
        "sft": {**common, "plan": item.corrected_plan},
        "preference": {
            **common,
            "chosen": item.corrected_plan,
            "rejected": item.candidate_plan,
        },
    }


class CorrectionStore:
    def __init__(
        self,
        project_id: str,
        projects_root: str | Path = DEFAULT_PROJECTS_ROOT,
    ) -> None:
        self.project_id = project_id
        self.projects = ProjectStore(projects_root)
        self.projects.load_project(project_id)
        self.path = self.projects.project_dir(project_id) / "corrections.json"

    def list(self) -> list[PlanCorrection]:
        if not self.path.exists():
            return []
        value = json.loads(self.path.read_text(encoding="utf-8"))
        return [
            PlanCorrection.from_dict(item)
            for item in value
            if isinstance(item, Mapping)
        ]

    def save(self, corrections: list[PlanCorrection]) -> None:
        self.path.write_text(
            json.dumps(
                [item.to_dict() for item in corrections],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def add(self, correction: PlanCorrection | Mapping[str, Any]) -> PlanCorrection:
        item = (
            correction
            if isinstance(correction, PlanCorrection)
            else PlanCorrection.from_dict(correction)
        )
        if item.project_id != self.project_id:
            raise ValueError("correction project_id does not match store")
        current = self.list()
        current.append(item)
        self.save(current)
        return item

    def review(
        self, correction_id: str, *, accepted: bool = True
    ) -> PlanCorrection:
        current = self.list()
        for item in current:
            if item.correction_id == correction_id:
                item.review_status = "reviewed" if accepted else "rejected"
                self.save(current)
                return item
        raise KeyError(f"unknown correction: {correction_id!r}")


__all__ = [
    "CorrectionStore",
    "PlanCorrection",
    "correction_is_eligible",
    "correction_records",
]
