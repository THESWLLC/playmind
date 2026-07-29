from __future__ import annotations

from pathlib import Path

import pytest

from playmind.planner_v2.model_registry import ModelRegistry
from playmind.studio.corrections import (
    PlanCorrection,
    correction_is_eligible,
    correction_records,
)


def test_reviewed_correction_produces_sft_and_preference_rows() -> None:
    correction = PlanCorrection(
        project_id="p1",
        planner_state={"sensors": {}},
        candidate_plan={"skills": ["engage_target"]},
        corrected_plan={"skills": ["recover_health"]},
        review_status="reviewed",
    )
    provenance = {
        "source_type": "user_owned_recording",
        "rights_confirmed": True,
    }
    assert correction_is_eligible(correction, provenance=provenance)
    rows = correction_records(correction, provenance=provenance)
    assert rows["sft"]["plan"]["skills"] == ["recover_health"]
    assert rows["preference"]["rejected"]["skills"] == ["engage_target"]


def test_registry_enforces_smoke_and_offline_use_restrictions(
    tmp_path: Path,
) -> None:
    registry = ModelRegistry(tmp_path / "registry.sqlite")
    offline = registry.register(
        "offline",
        status="candidate",
        live_use_prohibited=True,
        source_game_profile="retail_wow_offline_only",
        allowed_uses=["offline_evaluation"],
    )
    assert offline["live_use_prohibited"] is True
    assert registry.assert_use_allowed("offline", "offline_evaluation")
    with pytest.raises(ValueError, match="prohibited"):
        registry.assert_use_allowed("offline", "live_gameplay")
    with pytest.raises(ValueError, match="promotion prohibited"):
        registry.promote("offline", manual_override=True)

    registry.register("smoke", status="candidate", smoke=True)
    with pytest.raises(ValueError, match="promotion prohibited"):
        registry.promote("smoke", manual_override=True)
