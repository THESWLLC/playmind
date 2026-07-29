from __future__ import annotations

from playmind.studio.training_readiness import assess_training_readiness


def test_readiness_levels_and_blockers() -> None:
    smoke = assess_training_readiness(
        reviewed_examples=0, disk_free_gb=10, gpu=False
    )
    assert smoke.status == "Ready for smoke"
    insufficient = assess_training_readiness(
        reviewed_examples=3,
        license_confirmed=True,
        disk_free_gb=10,
        gpu=False,
    )
    assert insufficient.status == "Not ready"
    experimental = assess_training_readiness(
        reviewed_examples=10,
        frozen_real_benchmarks=1,
        license_confirmed=True,
        disk_free_gb=10,
        gpu=False,
    )
    assert experimental.status == "Ready for experimental"
    normal = assess_training_readiness(
        reviewed_examples=1000,
        preference_examples=100,
        frozen_real_benchmarks=1,
        license_confirmed=True,
        disk_free_gb=10,
        gpu=True,
    )
    assert normal.status == "Ready for normal"
    blocked = assess_training_readiness(
        reviewed_examples=10,
        license_confirmed=False,
        disk_free_gb=10,
        leakage=[{"project_id": "p"}],
    )
    assert blocked.status.startswith("Blocked:")
    assert len(blocked.blockers) == 2
