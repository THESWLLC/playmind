from __future__ import annotations

from pathlib import Path

import pytest

from playmind.studio.annotations import (
    AnnotationStore,
    TimelineSegment,
    annotation_is_eligible,
)
from playmind.studio.benchmark_builder import BenchmarkBuilder, StudioScenario
from playmind.studio.project_store import ProjectStore


def test_suggested_annotation_needs_review_and_supports_undo(tmp_path: Path) -> None:
    ProjectStore(tmp_path).create_project(project_id="review")
    annotations = AnnotationStore("review", tmp_path)
    segment = annotations.add(TimelineSegment(1, 2, "wait"))
    assert not annotation_is_eligible(segment)
    reviewed = annotations.review(segment.segment_id)
    assert annotation_is_eligible(reviewed)
    assert annotations.undo()[0].review_status == "suggested"


def test_frozen_real_benchmark_is_versioned_and_immutable(tmp_path: Path) -> None:
    scenario = StudioScenario(
        scenario_id="real-wait",
        category="loading",
        planner_state={"unknown_sensors": ["motion"]},
        expected_plan={"skills": ["wait"]},
        acceptable_alternative_plans=[{"skills": ["clear_modal", "wait"]}],
        project_id="project-1",
        source_id="recording-1",
        reviewed=True,
        provenance_eligible=True,
    )
    builder = BenchmarkBuilder(tmp_path)
    first = builder.freeze([scenario])
    second = builder.freeze([scenario])
    assert first["version"] == 1 and second["version"] == 2
    assert first["immutable"] is True
    with pytest.raises(FileExistsError, match="immutable"):
        builder.freeze([scenario], version=1)
