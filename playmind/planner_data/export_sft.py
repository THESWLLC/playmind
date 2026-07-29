"""Export demonstration/planner records as episode-safe chat SFT JSONL."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from playmind.demonstrations import list_sessions, load_session_samples
from playmind.planner_data.manifests import write_manifest
from playmind.planner_data.schemas import (
    SFTExample,
    build_planner_state,
    normalize_plan,
    planner_messages,
)
from playmind.planner_data.splits import (
    assert_episode_safe_splits,
    split_records_by_episode,
)

DEFAULT_SFT_ROOT = Path("data/playmind/planner/sft")


def load_demonstration_records(root: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for session in list_sessions(root):
        records.extend(load_session_samples(session))
    return records


def _eligible(record: Mapping[str, Any], plan: Mapping[str, Any]) -> bool:
    source = str(record.get("input_source") or "unknown")
    segmentation = record.get("segmentation_meta")
    segment_eligible = not (
        isinstance(segmentation, Mapping)
        and segmentation.get("training_eligible") is False
    )
    return (
        bool(record.get("training_eligible", True))
        and source != "playmind_generated"
        and segment_eligible
        and bool(plan.get("skills"))
    )


def _example_id(record: Mapping[str, Any], index: int) -> str:
    if record.get("example_id") or record.get("sample_id"):
        return str(record.get("example_id") or record.get("sample_id"))
    payload = f"{record.get('episode_id', 'unknown')}:{record.get('timestamp')}:{index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def export_sft(
    records: Iterable[Mapping[str, Any]],
    output_dir: str | Path = DEFAULT_SFT_ROOT,
    *,
    manifest_dir: str | Path | None = None,
    seed: int = 0,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    include_ineligible: bool = False,
) -> dict[str, Any]:
    """Write train/val/test SFT files and return their integrity manifest."""
    source_rows = [dict(record) for record in records]
    prepared: list[dict[str, Any]] = []
    source_eligible = 0
    for index, record in enumerate(source_rows):
        state = build_planner_state(record)
        plan = normalize_plan(
            record.get("plan") or record.get("target_plan") or record.get("assistant_plan"),
            record,
        )
        eligible = _eligible(record, plan)
        source_eligible += int(eligible)
        if not eligible and not include_ineligible:
            continue
        episode_id = str(record.get("episode_id") or "unknown")
        first_skill = (plan.get("skills") or [None])[0]
        if isinstance(first_skill, Mapping):
            first_skill = first_skill.get("name") or first_skill.get("skill")
        example = SFTExample(
            example_id=_example_id(record, index),
            episode_id=episode_id,
            messages=planner_messages(state, plan),
            eligible=eligible,
            metadata={
                "sample_id": record.get("sample_id"),
                "session_id": record.get("session_id"),
                "input_source": record.get("input_source", "unknown"),
                "skill": first_skill,
            },
        ).to_dict()
        # Retained outside messages for audits/manifests; trainers consume messages.
        example["planner_state"] = state
        example["plan"] = plan
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
    manifest_path = manifests / "sft.manifest.json"
    manifest = write_manifest(manifest_path, "sft", all_rows, files)
    manifest["counts"]["source_total"] = len(source_rows)
    manifest["eligibility"] = {
        "eligible": source_eligible,
        "ineligible": len(source_rows) - source_eligible,
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


export_sft_dataset = export_sft

__all__ = [
    "DEFAULT_SFT_ROOT",
    "export_sft",
    "export_sft_dataset",
    "load_demonstration_records",
]
