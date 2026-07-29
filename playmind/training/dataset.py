"""Demonstration dataset for behavior-cloning (works without torch).

Reads JSONL demos from demonstration sessions, builds temporal windows, and
splits train/val/test by episode id (never by individual timesteps).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Sequence

from playmind.demonstrations import DEFAULT_ROOT, list_sessions, load_session_samples

SplitName = Literal["train", "val", "test", "all"]


def _episode_bucket(episode_id: str, seed: int = 0) -> float:
    """Deterministic [0, 1) hash of episode_id for split assignment."""
    h = hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / float(0xFFFFFFFF)


def _assign_split(
    episode_id: str,
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> SplitName:
    train_r, val_r, _test_r = ratios
    total = sum(ratios) or 1.0
    t, v = train_r / total, val_r / total
    x = _episode_bucket(episode_id, seed=seed)
    if x < t:
        return "train"
    if x < t + v:
        return "val"
    return "test"


def _feature_vector(obs: dict[str, Any]) -> list[float]:
    """Compact numeric features from an observation dict (no torch)."""

    def _f(key: str, default: float = 0.0) -> float:
        if key not in obs or obs[key] is None:
            return default
        val = obs[key]
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
    hp = obs.get("vision_player_hp")
    if hp is None:
        hp = obs.get("player_hp")
    if hp is None and player:
        hp = player.get("hp")
    try:
        hp_f = float(hp) if hp is not None else 0.5
    except (TypeError, ValueError):
        hp_f = 0.5

    return [
        hp_f,
        _f("target_hp", _f("target_hp_est", 0.0)),
        _f("has_target"),
        _f("in_combat"),
        _f("is_dead"),
        _f("is_ghost"),
        _f("motion", 0.0),
        _f("hostiles_near"),
        float(obs.get("hostile_count") or 0),
        _f("blocking_modal", _f("modal_menu")),
        float(obs.get("stagnation_count") or 0),
        float(obs.get("failed_action_streak") or 0),
        _f("objective_progress", 0.0),
    ]


class DemonstrationDataset:
    """Episode-split temporal windows over demonstration JSONL sessions.

    ``__getitem__`` returns plain dicts (no torch tensors) so training scripts
    can dry-validate without CUDA / torch.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        window_size: int = 4,
        split: SplitName = "train",
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 0,
        session_dirs: Sequence[str | Path] | None = None,
        include_unlabeled: bool = True,
    ) -> None:
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        self.window_size = max(1, int(window_size))
        self.split = split
        self.split_ratios = split_ratios
        self.seed = int(seed)
        self.include_unlabeled = include_unlabeled

        if session_dirs is not None:
            sessions = [Path(p) for p in session_dirs]
        else:
            sessions = list_sessions(self.root)

        # episode_id -> ordered samples
        by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.sessions_loaded: list[str] = []
        for session_dir in sessions:
            rows = load_session_samples(session_dir)
            if not rows:
                continue
            self.sessions_loaded.append(str(session_dir))
            for row in rows:
                if row.get("label") == "bad" and not include_unlabeled:
                    continue
                ep = str(row.get("episode_id") or session_dir.name)
                enriched = dict(row)
                enriched["_session_dir"] = str(session_dir)
                by_episode[ep].append(enriched)

        # Sort each episode by index/timestamp.
        for ep, rows in by_episode.items():
            rows.sort(key=lambda r: (r.get("index", 0), r.get("timestamp", 0.0)))

        self.episode_ids = sorted(by_episode.keys())
        self._episode_split: dict[str, SplitName] = {
            ep: _assign_split(ep, ratios=split_ratios, seed=seed) for ep in self.episode_ids
        }

        self._windows: list[dict[str, Any]] = []
        for ep, rows in by_episode.items():
            ep_split = self._episode_split[ep]
            if split != "all" and ep_split != split:
                continue
            for end in range(len(rows)):
                start = max(0, end - self.window_size + 1)
                window = rows[start : end + 1]
                # Left-pad conceptually by repeating first obs if short.
                while len(window) < self.window_size:
                    window = [window[0]] + window if window else window
                target = rows[end]
                skill = target.get("skill") or target.get("skill_label")
                if skill is None and target.get("key_events"):
                    # Best-effort: first string key event as pseudo-label.
                    for ev in target["key_events"]:
                        if isinstance(ev, str) and ev:
                            skill = ev
                            break
                        if isinstance(ev, dict) and ev.get("skill"):
                            skill = str(ev["skill"])
                            break
                if skill is None and not include_unlabeled:
                    continue
                feats = [_feature_vector(w.get("observation") or {}) for w in window]
                self._windows.append(
                    {
                        "episode_id": ep,
                        "split": ep_split,
                        "window_size": self.window_size,
                        "features": feats,
                        "feature": feats[-1],
                        "skill": skill,
                        "observation": target.get("observation") or {},
                        "observations": [w.get("observation") or {} for w in window],
                        "frame_path": target.get("frame_path"),
                        "sample_id": target.get("sample_id"),
                        "session_id": target.get("session_id"),
                        "label": target.get("label"),
                        "schema_version": target.get("schema_version", 1),
                        "timestamp": target.get("timestamp"),
                        "goal": target.get("goal"),
                        "key_events": list(target.get("key_events") or []),
                    }
                )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self._windows[index])

    def episode_split_map(self) -> dict[str, str]:
        return {k: str(v) for k, v in self._episode_split.items()}

    def validate(self) -> dict[str, Any]:
        """Dry-validate dataset contents; returns a summary dict."""
        skills = sorted({w["skill"] for w in self._windows if w.get("skill")})
        return {
            "root": str(self.root),
            "split": self.split,
            "sessions": len(self.sessions_loaded),
            "episodes": len(self.episode_ids),
            "windows": len(self._windows),
            "window_size": self.window_size,
            "skills": skills,
            "episode_splits": {
                s: sum(1 for v in self._episode_split.values() if v == s)
                for s in ("train", "val", "test")
            },
        }

    def iter_batches(self, batch_size: int = 8) -> Iterator[dict[str, Any]]:
        """Yield dict batches (lists of fields) — no torch required."""
        bs = max(1, int(batch_size))
        for i in range(0, len(self._windows), bs):
            chunk = self._windows[i : i + bs]
            yield {
                "features": [c["features"] for c in chunk],
                "feature": [c["feature"] for c in chunk],
                "skill": [c["skill"] for c in chunk],
                "episode_id": [c["episode_id"] for c in chunk],
                "sample_id": [c["sample_id"] for c in chunk],
                "observation": [c["observation"] for c in chunk],
            }


def load_all_meta_jsonl(path: Path) -> list[dict[str, Any]]:
    """Helper used by tests / CLI to read a single meta.jsonl file."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
