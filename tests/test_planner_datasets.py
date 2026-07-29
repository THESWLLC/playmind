from __future__ import annotations

import json
from pathlib import Path

from playmind.planner_data import (
    FROZEN_EVAL_SCENARIOS,
    assert_episode_safe_splits,
    export_eval_suite,
    export_preferences,
    export_sft,
)


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_sft_export_episode_safe_manifest_and_unknown_sensors(tmp_path: Path) -> None:
    records = [
        {
            "sample_id": f"sample-{episode}-{index}",
            "episode_id": f"episode-{episode}",
            "observation": (
                {"has_target": False}
                if episode == 0 and index == 0
                else {"player_hp": 0.8, "has_target": False, "life_phase": "alive"}
            ),
            "plan": {"skills": ["explore"], "rationale": "seek objective"},
            "input_source": "human",
        }
        for episode in range(12)
        for index in range(2)
    ]
    output = tmp_path / "planner" / "sft"
    manifest = export_sft(records, output, seed=7)
    splits = {name: _jsonl(output / f"{name}.jsonl") for name in ("train", "val", "test")}
    assert_episode_safe_splits(splits)
    locations: dict[str, set[str]] = {}
    for split, rows in splits.items():
        for row in rows:
            locations.setdefault(row["episode_id"], set()).add(split)
    assert all(len(values) == 1 for values in locations.values())
    assert manifest["counts"]["total"] == len(records)
    assert manifest["hashes"]
    assert manifest["eligibility"]["eligible"] == len(records)
    assert Path(manifest["manifest_path"]).exists()

    unknown = next(
        row
        for rows in splits.values()
        for row in rows
        if row["example_id"] == "sample-0-0"
    )
    user = json.loads(unknown["messages"][1]["content"])
    state = user["planner_state"]
    assert state["sensors"]["player_hp"]["known"] is False
    assert "player_hp" in state["unknown_sensors"]
    assert state["sensors"]["has_target"]["known"] is True


def test_preference_export_pairs_outcomes_and_manifest(tmp_path: Path) -> None:
    records = [
        {
            "episode_id": f"pref-{index}",
            "observation": {"player_hp": 0.2, "in_combat": False},
            "chosen": {"skills": ["recover_health"]},
            "rejected": {"skills": ["engage_target"]},
            "outcomes": {"chosen": "success", "rejected": "death"},
            "input_source": "human",
        }
        for index in range(8)
    ]
    output = tmp_path / "planner" / "preferences"
    manifest = export_preferences(records, output, seed=2)
    rows = sum(
        (_jsonl(output / f"{split}.jsonl") for split in ("train", "val", "test")),
        [],
    )
    assert len(rows) == len(records)
    assert rows[0]["chosen"]["skills"] == ["recover_health"]
    assert rows[0]["rejected"]["skills"] == ["engage_target"]
    assert rows[0]["outcomes"]["chosen"] == "success"
    assert manifest["eligibility"]["ineligible"] == 0
    assert manifest["hashes"]


def test_generated_rows_are_excluded_and_eval_coverage_is_frozen(tmp_path: Path) -> None:
    manifest = export_sft(
        [
            {
                "episode_id": "generated",
                "input_source": "playmind_generated",
                "skill": "explore",
                "training_eligible": True,
            }
        ],
        tmp_path / "sft",
    )
    assert manifest["counts"]["total"] == 0
    assert manifest["eligibility"]["ineligible"] == 1

    eval_manifest = export_eval_suite(tmp_path / "evaluation")
    expected = {
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
        "invalid_temptation",
        "malformed_recovery",
    }
    assert {scenario.category for scenario in FROZEN_EVAL_SCENARIOS} == expected
    assert set(eval_manifest["coverage"]["categories"]) == expected
