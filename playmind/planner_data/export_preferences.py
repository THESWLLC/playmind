"""Export episode-safe planner preference pairs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from playmind.planner_data.manifests import write_manifest
from playmind.planner_data.schemas import PreferenceExample, build_planner_state, normalize_plan
from playmind.planner_data.splits import (
    assert_episode_safe_splits,
    split_records_by_episode,
)

DEFAULT_PREFERENCES_ROOT = Path("data/playmind/planner/preferences")


def _id(record: Mapping[str, Any], index: int) -> str:
    if record.get("example_id") or record.get("preference_id"):
        return str(record.get("example_id") or record.get("preference_id"))
    payload = f"{record.get('episode_id', 'unknown')}:{record.get('timestamp')}:{index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _outcomes(record: Mapping[str, Any]) -> dict[str, Any]:
    outcomes = record.get("outcomes")
    result = dict(outcomes) if isinstance(outcomes, Mapping) else {}
    if "chosen" not in result and "chosen_outcome" in record:
        result["chosen"] = record.get("chosen_outcome")
    if "rejected" not in result and "rejected_outcome" in record:
        result["rejected"] = record.get("rejected_outcome")
    return result


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def export_preferences(
    records: Iterable[Mapping[str, Any]],
    output_dir: str | Path = DEFAULT_PREFERENCES_ROOT,
    *,
    manifest_dir: str | Path | None = None,
    seed: int = 0,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    include_ineligible: bool = False,
) -> dict[str, Any]:
    source_rows = [dict(record) for record in records]
    prepared: list[dict[str, Any]] = []
    eligible_count = 0
    for index, record in enumerate(source_rows):
        chosen = normalize_plan(
            record.get("chosen") or record.get("chosen_plan") or record.get("preferred"),
            record,
        )
        rejected = normalize_plan(
            record.get("rejected") or record.get("rejected_plan") or record.get("dispreferred")
        )
        eligible = (
            bool(record.get("training_eligible", True))
            and str(record.get("input_source") or "unknown") != "playmind_generated"
            and bool(chosen.get("skills"))
            and bool(rejected.get("skills"))
            and chosen != rejected
        )
        eligible_count += int(eligible)
        if not eligible and not include_ineligible:
            continue
        example = PreferenceExample(
            example_id=_id(record, index),
            episode_id=str(record.get("episode_id") or "unknown"),
            planner_state=build_planner_state(record),
            chosen=chosen,
            rejected=rejected,
            outcomes=_outcomes(record),
            eligible=eligible,
        ).to_dict()
        example["input_source"] = record.get("input_source", "unknown")
        prepared.append(example)

    splits = split_records_by_episode(
        prepared,
        ratios=split_ratios,
        seed=seed,
    )
    assert_episode_safe_splits(splits)
    root = Path(output_dir)
    files: list[Path] = []
    all_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        path = root / f"{split}.jsonl"
        rows = splits[split]  # type: ignore[index]
        _write_jsonl(path, rows)
        files.append(path)
        all_rows.extend(rows)

    manifests = (
        Path(manifest_dir)
        if manifest_dir is not None
        else root.parent / "manifests"
    )
    manifest_path = manifests / "preferences.manifest.json"
    manifest = write_manifest(manifest_path, "preferences", all_rows, files)
    manifest["counts"]["source_total"] = len(source_rows)
    manifest["eligibility"] = {
        "eligible": eligible_count,
        "ineligible": len(source_rows) - eligible_count,
        "exported": len(all_rows),
    }
    manifest["split_seed"] = seed
    manifest["split_ratios"] = list(split_ratios)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


export_preference_dataset = export_preferences

__all__ = [
    "DEFAULT_PREFERENCES_ROOT",
    "export_preference_dataset",
    "export_preferences",
]
