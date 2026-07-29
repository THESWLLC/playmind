"""Build versioned offline planner benchmarks from reviewed scenarios."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playmind.planner_data.schemas import normalize_plan


BENCHMARK_TIERS = frozenset({"synthetic", "development", "frozen_real"})
REQUIRED_BENCHMARK_CATEGORIES = frozenset(
    {
        "combat",
        "recovery",
        "multi_enemy",
        "target_loss",
        "death",
        "ghost",
        "loading",
        "inventory",
        "quest",
        "modal",
        "nav",
        "stuck",
        "skill_fail",
        "conflicting_sensors",
        "unknown_sensors",
        "long_horizon",
    }
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass
class StudioScenario:
    scenario_id: str
    category: str
    planner_state: dict[str, Any]
    expected_plan: dict[str, Any]
    acceptable_alternative_plans: list[dict[str, Any]] = field(default_factory=list)
    project_id: str = ""
    source_id: str = ""
    reviewed: bool = False
    provenance_eligible: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        self.expected_plan = normalize_plan(self.expected_plan)
        self.acceptable_alternative_plans = [
            normalize_plan(value) for value in self.acceptable_alternative_plans
        ]
        if not self.scenario_id or not self.category:
            raise ValueError("scenario_id and category are required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def alternative_plans(self) -> list[dict[str, Any]]:
        return self.acceptable_alternative_plans

    @property
    def acceptable_plans(self) -> list[dict[str, Any]]:
        return [self.expected_plan, *self.acceptable_alternative_plans]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StudioScenario:
        data = dict(value)
        alternatives = data.pop(
            "acceptable_alternative_plans",
            data.pop("alternative_plans", data.pop("acceptable_plans", [])),
        )
        data["acceptable_alternative_plans"] = alternatives
        return cls(**data)


def missing_required_categories(
    scenarios: Iterable[StudioScenario | Mapping[str, Any]],
    required: Iterable[str] = REQUIRED_BENCHMARK_CATEGORIES,
) -> list[str]:
    present = {
        (
            value.category
            if isinstance(value, StudioScenario)
            else str(value.get("category") or "")
        )
        for value in scenarios
    }
    return sorted(set(required) - present)


class BenchmarkBuilder:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _versions(self, benchmark_id: str) -> list[int]:
        if not self.root.exists():
            return []
        pattern = re.compile(re.escape(benchmark_id) + r"_v(\d+)\.json$")
        return sorted(
            int(match.group(1))
            for path in self.root.glob(f"{benchmark_id}_v*.json")
            if (match := pattern.fullmatch(path.name))
        )

    def freeze(
        self,
        scenarios: Iterable[StudioScenario | Mapping[str, Any]],
        *,
        benchmark_id: str = "studio_real_benchmark",
        tier: str = "frozen_real",
        version: int | None = None,
        required_categories: Iterable[str] = (),
    ) -> dict[str, Any]:
        if not _SAFE_NAME.fullmatch(benchmark_id):
            raise ValueError("benchmark_id must be filename-safe")
        if tier not in BENCHMARK_TIERS:
            raise ValueError(f"tier must be one of {sorted(BENCHMARK_TIERS)}")
        items = [
            value if isinstance(value, StudioScenario) else StudioScenario.from_dict(value)
            for value in scenarios
        ]
        if not items:
            raise ValueError("cannot freeze an empty benchmark")
        if len({item.scenario_id for item in items}) != len(items):
            raise ValueError("scenario_id values must be unique")
        missing = missing_required_categories(items, required_categories)
        if missing:
            raise ValueError("missing required benchmark categories: " + ", ".join(missing))
        if tier == "frozen_real":
            ineligible = [
                item.scenario_id
                for item in items
                if not item.reviewed or not item.provenance_eligible
            ]
            if ineligible:
                raise ValueError(
                    "frozen_real scenarios require review and eligible provenance: "
                    + ", ".join(ineligible)
                )
        versions = self._versions(benchmark_id)
        selected_version = int(version) if version is not None else (max(versions, default=0) + 1)
        if selected_version <= 0:
            raise ValueError("version must be positive")
        path = self.root / f"{benchmark_id}_v{selected_version}.json"
        if path.exists():
            raise FileExistsError(
                f"benchmark version is immutable: {path}; create a new version"
            )
        scenario_rows = [item.to_dict() for item in items]
        canonical = json.dumps(scenario_rows, sort_keys=True, separators=(",", ":"))
        payload = {
            "schema_version": 1,
            "benchmark_id": benchmark_id,
            "version": selected_version,
            "tier": tier,
            "immutable": tier == "frozen_real",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scenario_count": len(items),
            "categories": sorted({item.category for item in items}),
            "content_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "scenarios": scenario_rows,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        payload["path"] = str(path)
        return payload


def freeze_benchmark(
    scenarios: Iterable[StudioScenario | Mapping[str, Any]],
    output_root: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return BenchmarkBuilder(output_root).freeze(scenarios, **kwargs)


__all__ = [
    "BENCHMARK_TIERS",
    "BenchmarkBuilder",
    "REQUIRED_BENCHMARK_CATEGORIES",
    "StudioScenario",
    "freeze_benchmark",
    "missing_required_categories",
]
