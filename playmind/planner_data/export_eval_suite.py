"""Frozen planner evaluation scenarios spanning lifecycle and uncertainty."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from playmind.planner_data.manifests import write_manifest
from playmind.planner_data.schemas import EvalScenario, build_planner_state, normalize_plan

DEFAULT_EVALUATION_ROOT = Path("data/playmind/planner/evaluation")


def _scenario(
    category: str,
    observation: dict[str, Any],
    skills: list[str],
    *,
    goal: str = "continue safely",
    invalid: dict[str, Any] | None = None,
) -> EvalScenario:
    state = build_planner_state(
        {
            "episode_id": f"eval-{category}",
            "goal": goal,
            "observation": observation,
            "lifecycle_state": observation.get("life_phase"),
        }
    )
    return EvalScenario(
        scenario_id=f"planner-{category}-v1",
        category=category,
        planner_state=state,
        expected_plan=normalize_plan(skills),
        invalid_plan=invalid,
    )


FROZEN_EVAL_SCENARIOS: tuple[EvalScenario, ...] = (
    _scenario("combat", {"player_hp": 0.9, "has_target": True, "in_combat": True, "target_hp": 0.7, "life_phase": "alive"}, ["engage_target", "basic_combat_rotation"]),
    _scenario("recovery", {"player_hp": 0.2, "has_target": False, "in_combat": False, "life_phase": "alive"}, ["recover_health"]),
    _scenario("multi_enemy", {"player_hp": 0.55, "has_target": True, "in_combat": True, "hostile_count": 4, "life_phase": "alive"}, ["disengage", "recover_health"], invalid={"skills": ["basic_combat_rotation"]}),
    _scenario("target_loss", {"player_hp": 0.8, "has_target": False, "in_combat": True, "hostile_count": 1, "life_phase": "alive"}, ["acquire_target", "validate_target"]),
    _scenario("death", {"player_hp": 0.0, "is_dead": True, "is_ghost": False, "life_phase": "dead_dialog"}, ["death_recovery"], invalid={"skills": ["engage_target"]}),
    _scenario("ghost", {"player_hp": 0.0, "is_dead": False, "is_ghost": True, "life_phase": "ghost", "motion": 1.0}, ["ghost_runback"]),
    _scenario("loading", {"life_phase": "loading", "motion": 0.0, "blocking_modal": False}, ["wait"]),
    _scenario("inventory", {"player_hp": 0.7, "blocking_modal": True, "life_phase": "alive", "ui_detections": ["inventory"]}, ["clear_modal"]),
    _scenario("quest", {"player_hp": 0.9, "has_target": False, "objective_text": "Speak to the quartermaster", "objective_progress": 0.8, "life_phase": "alive"}, ["interact"]),
    _scenario("modal", {"blocking_modal": True, "life_phase": "alive", "ui_detections": ["confirmation dialog"]}, ["clear_modal"]),
    _scenario("nav", {"player_hp": 1.0, "has_target": False, "motion": 2.0, "life_phase": "alive"}, ["explore"], goal="navigate to quest objective"),
    _scenario("stuck", {"player_hp": 0.8, "motion": 0.0, "stagnation_count": 8, "life_phase": "alive"}, ["unstuck"]),
    _scenario("skill_fail", {"player_hp": 0.75, "has_target": True, "failed_action_streak": 4, "recent_action": "engage_target", "recent_action_outcome": "failure", "life_phase": "alive"}, ["validate_target", "approach_target"]),
    _scenario("conflicting_sensors", {"player_hp": 0.9, "is_dead": True, "is_ghost": False, "life_phase": "dead_dialog", "sensor_warnings": ["player_hp conflicts with dead state"]}, ["death_recovery"]),
    _scenario("unknown_sensors", {"life_phase": "unknown", "sensor_warnings": ["capture unavailable"]}, ["wait"]),
    _scenario("long_horizon", {"player_hp": 0.85, "has_target": False, "objective_text": "Defeat 10 wolves", "objective_progress": 0.2, "life_phase": "alive"}, ["explore", "acquire_target", "approach_target", "engage_target", "basic_combat_rotation", "loot_target"], goal="complete the wolf quest"),
    _scenario("invalid_temptation", {"player_hp": 0.05, "has_target": True, "in_combat": True, "hostile_count": 3, "life_phase": "alive"}, ["disengage", "recover_health"], invalid={"skills": ["basic_combat_rotation"], "rationale": "keep attacking"}),
    _scenario("malformed_recovery", {"player_hp": 0.15, "has_target": False, "in_combat": False, "life_phase": "alive"}, ["recover_health"], invalid={"skillz": "recover_health", "then": object.__name__}),
)


def export_eval_suite(
    output_dir: str | Path = DEFAULT_EVALUATION_ROOT,
    *,
    manifest_dir: str | Path | None = None,
    scenarios: Iterable[EvalScenario] = FROZEN_EVAL_SCENARIOS,
) -> dict[str, Any]:
    rows = []
    for scenario in scenarios:
        row = scenario.to_dict()
        row["split"] = "evaluation"
        row["eligible"] = True
        rows.append(row)
    root = Path(output_dir)
    path = root / "eval_suite.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifests = (
        Path(manifest_dir)
        if manifest_dir is not None
        else root.parent / "manifests"
    )
    manifest_path = manifests / "evaluation.manifest.json"
    manifest = write_manifest(manifest_path, "evaluation", rows, [path])
    manifest["frozen_suite_version"] = 1
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


__all__ = [
    "DEFAULT_EVALUATION_ROOT",
    "FROZEN_EVAL_SCENARIOS",
    "export_eval_suite",
]
