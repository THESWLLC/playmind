"""Conservative readiness gates for Studio-produced training data."""

from __future__ import annotations

import importlib.util
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReadinessReport:
    status: str
    checks: dict[str, dict[str, Any]]
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status.startswith("Ready")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ready"] = self.ready
        return result


def gpu_status() -> dict[str, Any]:
    """Best-effort dependency/GPU check; never installs or initializes training."""

    if importlib.util.find_spec("torch") is None:
        return {"available": False, "detail": "PyTorch is not installed", "stub": True}
    try:
        import torch  # type: ignore

        available = bool(torch.cuda.is_available())
        return {
            "available": available,
            "detail": torch.cuda.get_device_name(0) if available else "CUDA unavailable",
            "stub": False,
        }
    except Exception as exc:
        return {"available": False, "detail": str(exc), "stub": True}


def disk_status(path: str | Path = ".") -> dict[str, Any]:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target)
    return {
        "free_bytes": usage.free,
        "free_gb": usage.free / (1024**3),
        "total_bytes": usage.total,
    }


def assess_training_readiness(
    *,
    reviewed_examples: int,
    preference_examples: int = 0,
    frozen_real_benchmarks: int = 0,
    provenance_eligible: bool = True,
    leakage: Sequence[Mapping[str, Any]] | None = None,
    license_confirmed: bool = False,
    gpu: Mapping[str, Any] | bool | None = None,
    disk_free_gb: float | None = None,
    min_experimental_examples: int = 10,
    min_normal_examples: int = 1000,
    min_normal_preferences: int = 100,
    min_disk_gb: float = 2.0,
) -> ReadinessReport:
    reviewed = max(0, int(reviewed_examples))
    preferences = max(0, int(preference_examples))
    benchmarks = max(0, int(frozen_real_benchmarks))
    leak_rows = list(leakage or [])
    gpu_info = (
        dict(gpu)
        if isinstance(gpu, Mapping)
        else {"available": bool(gpu), "detail": "caller supplied", "stub": True}
        if gpu is not None
        else gpu_status()
    )
    free_gb = float(disk_free_gb) if disk_free_gb is not None else disk_status()["free_gb"]
    blockers: list[str] = []
    warnings: list[str] = []
    if reviewed and not license_confirmed:
        blockers.append("license confirmation is required for real-data training")
    if reviewed and not provenance_eligible:
        blockers.append("one or more sources have ineligible provenance")
    if leak_rows:
        blockers.append("project/source leakage was detected across splits")
    if free_gb < min_disk_gb:
        blockers.append(f"only {free_gb:.2f} GiB disk is free")
    if not gpu_info.get("available"):
        warnings.append("GPU unavailable; smoke checks can still run")

    checks = {
        "reviewed_examples": {
            "passed": reviewed >= min_experimental_examples,
            "value": reviewed,
            "experimental_minimum": min_experimental_examples,
            "normal_minimum": min_normal_examples,
        },
        "preferences": {
            "passed": preferences >= min_normal_preferences,
            "value": preferences,
            "normal_minimum": min_normal_preferences,
        },
        "frozen_real_benchmark": {
            "passed": benchmarks > 0,
            "value": benchmarks,
        },
        "provenance": {"passed": bool(provenance_eligible), "value": provenance_eligible},
        "leakage": {"passed": not leak_rows, "violations": leak_rows},
        "gpu": gpu_info,
        "disk": {"passed": free_gb >= min_disk_gb, "free_gb": free_gb},
        "license_confirmation": {
            "passed": bool(license_confirmed),
            "value": bool(license_confirmed),
        },
    }
    if blockers:
        status = "Blocked: " + "; ".join(blockers)
    elif reviewed == 0:
        status = "Ready for smoke"
    elif reviewed < min_experimental_examples or benchmarks == 0:
        status = "Not ready"
    elif (
        reviewed >= min_normal_examples
        and preferences >= min_normal_preferences
        and benchmarks > 0
        and gpu_info.get("available")
    ):
        status = "Ready for normal"
    else:
        status = "Ready for experimental"
    return ReadinessReport(
        status=status,
        checks=checks,
        blockers=blockers,
        warnings=warnings,
        counts={
            "reviewed_examples": reviewed,
            "preference_examples": preferences,
            "frozen_real_benchmarks": benchmarks,
        },
    )


training_readiness = assess_training_readiness


__all__ = [
    "ReadinessReport",
    "assess_training_readiness",
    "disk_status",
    "gpu_status",
    "training_readiness",
]
