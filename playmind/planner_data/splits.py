"""Deterministic, episode-safe planner dataset splits."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any, Literal

SplitName = Literal["train", "val", "test"]


def episode_bucket(episode_id: str, *, seed: int = 0) -> float:
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def assign_episode_split(
    episode_id: str,
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> SplitName:
    if any(float(value) < 0 for value in ratios) or sum(ratios) <= 0:
        raise ValueError("ratios must be non-negative with a positive sum")
    train_ratio, val_ratio, _ = ratios
    total = float(sum(ratios))
    value = episode_bucket(str(episode_id), seed=seed)
    if value < train_ratio / total:
        return "train"
    if value < (train_ratio + val_ratio) / total:
        return "val"
    return "test"


def split_records_by_episode(
    records: Iterable[Mapping[str, Any]],
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[SplitName, list[dict[str, Any]]]:
    result: dict[SplitName, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for raw in records:
        record = dict(raw)
        episode_id = str(record.get("episode_id") or "unknown")
        split = assign_episode_split(episode_id, ratios=ratios, seed=seed)
        record["episode_id"] = episode_id
        record["split"] = split
        result[split].append(record)
    return result


def assert_episode_safe_splits(
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
) -> None:
    seen: dict[str, str] = {}
    for split, records in splits.items():
        for record in records:
            episode_id = str(record.get("episode_id") or "unknown")
            previous = seen.get(episode_id)
            if previous is not None and previous != split:
                raise ValueError(
                    f"episode leakage: {episode_id!r} occurs in {previous} and {split}"
                )
            seen[episode_id] = str(split)


__all__ = [
    "SplitName",
    "assert_episode_safe_splits",
    "assign_episode_split",
    "episode_bucket",
    "split_records_by_episode",
]
