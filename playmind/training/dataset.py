"""Episode-safe recurrent behavior-cloning dataset (PyTorch optional)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

from playmind.demonstrations import DEFAULT_ROOT, list_sessions, load_session_samples
from playmind.models.feature_schema import (
    FEATURE_DIM,
    FeatureNormalizer,
    structured_feature_vector_v2,
)
from playmind.observations import Observation

SplitName = Literal["train", "val", "test", "all"]
EpisodeKey = tuple[str, str]


def _episode_bucket(group_id: str, seed: int = 0) -> float:
    """Deterministic [0, 1) hash for episode/session grouping."""
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _assign_split(
    group_id: str,
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> SplitName:
    if any(float(x) < 0 for x in ratios) or sum(ratios) <= 0:
        raise ValueError("split_ratios must be non-negative with a positive sum")
    train_r, val_r, _test_r = ratios
    total = float(sum(ratios))
    value = _episode_bucket(group_id, seed=seed)
    if value < train_r / total:
        return "train"
    if value < (train_r + val_r) / total:
        return "val"
    return "test"


def _session_metadata(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "session.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return dict(value) if isinstance(value, Mapping) else {}
    except (OSError, ValueError):
        return {}


def _skill_label(row: Mapping[str, Any]) -> str | None:
    skill = row.get("skill") or row.get("skill_label")
    if skill:
        return str(skill)
    for event in row.get("key_events") or []:
        if isinstance(event, Mapping) and event.get("skill"):
            return str(event["skill"])
    return None


def _known_float(observation: Mapping[str, Any], name: str) -> float | None:
    raw = dict(observation)
    if name == "player_hp" and "player_hp" in raw and "vision_player_hp" not in raw:
        raw["vision_player_hp"] = raw["player_hp"]
    obs = Observation.from_legacy_dict(raw)
    value = getattr(obs, name, None)
    return float(value) if value is not None else None


def _delta(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    name: str,
) -> float | None:
    if previous is None:
        return None
    now = _known_float(current, name)
    before = _known_float(previous, name)
    return now - before if now is not None and before is not None else None


def _binary(observation: Mapping[str, Any], name: str) -> float | None:
    obs = Observation.from_legacy_dict(dict(observation))
    value = getattr(obs, name, None)
    return None if value is None else (1.0 if bool(value) else 0.0)


def _aux_targets(
    target: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> dict[str, float | None]:
    observation = target.get("observation") or {}
    previous_obs = previous.get("observation") if previous is not None else None
    outcome = str(target.get("_session_outcome") or target.get("label") or "").lower()
    success: float | None = None
    if outcome == "success":
        success = 1.0
    elif outcome in {"failure", "bad"}:
        success = 0.0
    return {
        "target_valid": _binary(observation, "has_target"),
        "combat": _binary(observation, "in_combat"),
        "death": _binary(observation, "is_dead"),
        "player_hp_delta": _delta(observation, previous_obs, "player_hp"),
        "target_hp_delta": _delta(observation, previous_obs, "target_hp"),
        "progress_delta": _delta(observation, previous_obs, "objective_progress"),
        "skill_success": success,
    }


class DemonstrationDataset:
    """Plain-Python, left-padded sequence windows.

    ``padding_mask`` is ``True`` exactly where a timestep is valid. Episodes
    are keyed by ``(session_id, episode_id)`` so repeated episode labels in
    separate recording sessions can never be merged into one sequence.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        history_length: int = 16,
        window_size: int | None = None,
        stride: int = 1,
        min_sequence_length: int = 1,
        split: SplitName = "train",
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        split_unit: Literal["episode", "session"] = "episode",
        seed: int = 0,
        session_dirs: Sequence[str | Path] | None = None,
        include_unlabeled: bool = False,
        include_bad: bool = False,
        outcome_filter: str | Sequence[str] | None = None,
        success_only: bool = False,
        failure_only: bool = False,
        balance_classes: bool = False,
    ) -> None:
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        selected_length = window_size if window_size is not None else history_length
        self.history_length = max(1, int(selected_length))
        self.window_size = self.history_length  # historical public name
        self.stride = max(1, int(stride))
        self.min_sequence_length = max(1, int(min_sequence_length))
        if self.min_sequence_length > self.history_length:
            raise ValueError("min_sequence_length cannot exceed history_length")
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"unknown split: {split}")
        if split_unit not in {"episode", "session"}:
            raise ValueError("split_unit must be 'episode' or 'session'")
        if success_only and failure_only:
            raise ValueError("success_only and failure_only are mutually exclusive")
        self.split: SplitName = split
        self.split_ratios = split_ratios
        self.split_unit = split_unit
        self.seed = int(seed)
        self.include_unlabeled = bool(include_unlabeled)
        self.include_bad = bool(include_bad)
        self.balance_classes = bool(balance_classes)

        requested_outcomes: set[str] | None = None
        if outcome_filter is not None:
            raw = [outcome_filter] if isinstance(outcome_filter, str) else outcome_filter
            requested_outcomes = {str(x).lower() for x in raw}
        if success_only:
            requested_outcomes = {"success"}
        elif failure_only:
            requested_outcomes = {"failure"}

        sessions = (
            [Path(path) for path in session_dirs]
            if session_dirs is not None
            else list_sessions(self.root)
        )
        by_episode: dict[EpisodeKey, list[dict[str, Any]]] = defaultdict(list)
        self.sessions_loaded: list[str] = []
        self.duplicate_count = 0
        seen_fingerprints: set[str] = set()
        for session_dir in sessions:
            rows = load_session_samples(session_dir)
            if not rows:
                continue
            session_meta = _session_metadata(session_dir)
            outcome = str(session_meta.get("outcome") or "").lower() or None
            if outcome == "bad" and not self.include_bad:
                continue
            if requested_outcomes is not None and outcome not in requested_outcomes:
                continue
            session_id_default = str(session_meta.get("session_id") or session_dir.name)
            accepted = False
            for raw_row in rows:
                row = dict(raw_row)
                if str(row.get("label") or "").lower() == "bad" and not self.include_bad:
                    continue
                skill = _skill_label(row)
                if skill is None and not self.include_unlabeled:
                    continue
                session_id = str(row.get("session_id") or session_id_default)
                episode_id = str(row.get("episode_id") or session_meta.get("episode_id") or "unknown")
                row["session_id"] = session_id
                row["episode_id"] = episode_id
                row["_session_dir"] = str(session_dir)
                row["_session_outcome"] = outcome
                row["_skill"] = skill
                fingerprint_payload = {
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "sample_id": row.get("sample_id"),
                    "timestamp": row.get("timestamp"),
                    "skill": skill,
                    "observation": row.get("observation"),
                }
                fingerprint = hashlib.sha256(
                    json.dumps(fingerprint_payload, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
                if fingerprint in seen_fingerprints:
                    self.duplicate_count += 1
                    continue
                seen_fingerprints.add(fingerprint)
                by_episode[(session_id, episode_id)].append(row)
                accepted = True
            if accepted:
                self.sessions_loaded.append(str(session_dir))

        for rows in by_episode.values():
            rows.sort(
                key=lambda row: (
                    int(row.get("index") or 0),
                    float(row.get("timestamp") or 0.0),
                )
            )

        self.episode_keys: list[EpisodeKey] = sorted(by_episode)
        self.episode_ids: list[str] = sorted({key[1] for key in self.episode_keys})
        self._episode_split: dict[EpisodeKey, SplitName] = {}
        for session_id, episode_id in self.episode_keys:
            # Episode ids are the split group by default. The composite key is
            # still used for storage/windowing so same-named episodes in
            # different sessions never concatenate.
            group_id = session_id if split_unit == "session" else episode_id
            self._episode_split[(session_id, episode_id)] = _assign_split(
                group_id, ratios=split_ratios, seed=self.seed
            )

        zero = [0.0] * FEATURE_DIM
        self._windows: list[dict[str, Any]] = []
        for episode_key, rows in by_episode.items():
            episode_split = self._episode_split[episode_key]
            if split != "all" and episode_split != split:
                continue
            for end in range(self.min_sequence_length - 1, len(rows), self.stride):
                start = max(0, end - self.history_length + 1)
                valid_rows = rows[start : end + 1]
                length = len(valid_rows)
                if length < self.min_sequence_length:
                    continue
                pad_count = self.history_length - length
                valid_features = [
                    structured_feature_vector_v2(
                        row.get("observation") or {},
                        row.get("temporal_summary") or row.get("summary"),
                    )
                    for row in valid_rows
                ]
                features = [list(zero) for _ in range(pad_count)] + valid_features
                padding_mask = [False] * pad_count + [True] * length
                target = rows[end]
                previous = rows[end - 1] if end > 0 else None
                skill = target.get("_skill")
                self._windows.append(
                    {
                        "features": features,
                        "feature": features[-1],
                        "length": length,
                        "padding_mask": padding_mask,
                        "skill_target": skill,
                        "aux_targets": _aux_targets(target, previous),
                        "episode_id": episode_key[1],
                        "session_id": episode_key[0],
                        "timestamp": float(target.get("timestamp") or 0.0),
                        "sample_weight": 1.0,
                        "skill": skill,
                        "observation": dict(target.get("observation") or {}),
                        # Backward-compatible inspection fields.
                        "observations": ([{}] * pad_count)
                        + [dict(row.get("observation") or {}) for row in valid_rows],
                        "split": episode_split,
                        "window_size": self.history_length,
                        "frame_path": target.get("frame_path"),
                        "sample_id": target.get("sample_id"),
                        "label": target.get("label"),
                        "schema_version": target.get("schema_version", 1),
                        "goal": target.get("goal"),
                        "key_events": list(target.get("key_events") or []),
                    }
                )

        self.skill_counts: dict[str, int] = dict(
            Counter(str(item["skill"]) for item in self._windows if item.get("skill"))
        )
        if self.balance_classes and self.skill_counts:
            total = float(sum(self.skill_counts.values()))
            classes = float(len(self.skill_counts))
            for item in self._windows:
                skill = item.get("skill")
                if skill in self.skill_counts:
                    item["sample_weight"] = total / (classes * self.skill_counts[str(skill)])
        self.diagnostic_warnings = self._diagnostic_warnings()

    def _diagnostic_warnings(self) -> list[str]:
        messages: list[str] = []
        total = sum(self.skill_counts.values())
        if total and self.skill_counts:
            skill, count = max(self.skill_counts.items(), key=lambda pair: pair[1])
            if count / float(total) >= 0.75:
                messages.append(
                    f"dominant skill {skill!r} accounts for {count / float(total):.1%} of samples"
                )
        episodes_per_skill: dict[str, set[EpisodeKey]] = defaultdict(set)
        for item in self._windows:
            if item.get("skill"):
                episodes_per_skill[str(item["skill"])].add(
                    (str(item["session_id"]), str(item["episode_id"]))
                )
        for skill, episodes in sorted(episodes_per_skill.items()):
            if len(episodes) < 2:
                messages.append(f"skill {skill!r} appears in only {len(episodes)} episode(s)")
        if self.duplicate_count:
            messages.append(f"excluded {self.duplicate_count} duplicate sample(s)")
        return messages

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self._windows[index])

    def episode_split_map(self) -> dict[str, str]:
        """Compatibility map; qualify duplicate episode ids with session ids."""
        counts = Counter(episode_id for _session_id, episode_id in self.episode_keys)
        return {
            (
                episode_id
                if counts[episode_id] == 1
                else f"{session_id}::{episode_id}"
            ): str(split)
            for (session_id, episode_id), split in self._episode_split.items()
        }

    def split_episode_keys(self) -> set[EpisodeKey]:
        return {
            (str(item["session_id"]), str(item["episode_id"]))
            for item in self._windows
        }

    def valid_feature_vectors(self) -> list[list[float]]:
        vectors: list[list[float]] = []
        for item in self._windows:
            vectors.extend(
                row
                for row, valid in zip(item["features"], item["padding_mask"])
                if valid
            )
        return vectors

    def fit_normalizer(self) -> FeatureNormalizer:
        """Fit only on a training split, preventing validation/test leakage."""
        if self.split != "train":
            raise ValueError(
                "FeatureNormalizer may only be fit from the training split (split='train')"
            )
        return FeatureNormalizer.fit(self.valid_feature_vectors())

    def validate(self) -> dict[str, Any]:
        skills = sorted(self.skill_counts)
        lengths = [int(item["length"]) for item in self._windows]
        return {
            "root": str(self.root),
            "split": self.split,
            "sessions": len(self.sessions_loaded),
            "episodes": len(self.episode_keys),
            "windows": len(self._windows),
            "window_size": self.history_length,
            "history_length": self.history_length,
            "stride": self.stride,
            "min_sequence_length": self.min_sequence_length,
            "feature_dim": FEATURE_DIM,
            "skills": skills,
            "per_skill_counts": dict(self.skill_counts),
            "duplicate_count": self.duplicate_count,
            "sequence_lengths": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "mean": sum(lengths) / float(len(lengths)) if lengths else 0.0,
            },
            "warnings": list(self.diagnostic_warnings),
            "episode_splits": {
                name: sum(1 for value in self._episode_split.values() if value == name)
                for name in ("train", "val", "test")
            },
        }

    def iter_batches(self, batch_size: int = 8) -> Iterator[dict[str, Any]]:
        """Yield list-valued batch dictionaries without requiring torch."""
        size = max(1, int(batch_size))
        for start in range(0, len(self._windows), size):
            chunk = self._windows[start : start + size]
            keys = (
                "features",
                "feature",
                "length",
                "padding_mask",
                "skill_target",
                "aux_targets",
                "skill",
                "episode_id",
                "session_id",
                "sample_id",
                "sample_weight",
                "observation",
            )
            yield {key: [item.get(key) for item in chunk] for key in keys}


def assert_no_split_leakage(*datasets: DemonstrationDataset) -> None:
    """Raise when any composite episode key occurs in two named splits."""
    seen: dict[EpisodeKey, str] = {}
    for dataset in datasets:
        if dataset.split == "all":
            continue
        for key in dataset.split_episode_keys():
            previous = seen.get(key)
            if previous is not None and previous != dataset.split:
                raise ValueError(
                    f"Episode leakage: session={key[0]!r} episode={key[1]!r} "
                    f"is shared by {previous} and {dataset.split}"
                )
            seen[key] = str(dataset.split)


def load_all_meta_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a standalone meta.jsonl file."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


__all__ = [
    "DemonstrationDataset",
    "assert_no_split_leakage",
    "load_all_meta_jsonl",
]
