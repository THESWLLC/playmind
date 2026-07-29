"""Small episodic memory stores for planner context."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_MEMORY_ROOT = Path("data/playmind/planner/memory")


def _entry(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        item = dict(value)
    elif hasattr(value, "to_dict") and callable(value.to_dict):
        raw = value.to_dict()
        item = dict(raw) if isinstance(raw, Mapping) else {"value": raw}
    else:
        item = {"value": value}
    item.setdefault("timestamp", time.time())
    return item


class EpisodicMemory:
    """Bounded in-process ring buffer."""

    def __init__(self, capacity: int = 50) -> None:
        self.capacity = max(1, int(capacity))
        self._items: deque[dict[str, Any]] = deque(maxlen=self.capacity)

    def append(self, episode: Mapping[str, Any] | Any) -> dict[str, Any]:
        item = _entry(episode)
        self._items.append(item)
        return item

    add = append

    def recent(self, limit: int | None = None) -> list[dict[str, Any]]:
        count = self.capacity if limit is None else max(0, int(limit))
        return [dict(item) for item in list(self._items)[-count:]]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


ShortTermMemory = EpisodicMemory
ShortTermEpisodicMemory = EpisodicMemory


class JsonlMemoryStore:
    """Append-only long-term JSONL storage."""

    def __init__(self, path: str | Path = DEFAULT_MEMORY_ROOT / "episodes.jsonl") -> None:
        candidate = Path(path)
        self.path = (
            candidate / "episodes.jsonl"
            if candidate.exists() and candidate.is_dir()
            else candidate
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, episode: Mapping[str, Any] | Any) -> dict[str, Any]:
        item = _entry(episode)
        line = json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return item

    add = append

    def load(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self._lock:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        records.append(value)
        if limit is not None:
            records = records[-max(0, int(limit)) :]
        return records

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        needle = str(query or "").strip().lower()
        records = self.load()
        if needle:
            records = [
                item
                for item in records
                if needle in json.dumps(item, sort_keys=True, default=str).lower()
            ]
        return records[-max(0, int(limit)) :]


LongTermMemory = JsonlMemoryStore
LongTermJsonlMemory = JsonlMemoryStore


class PlannerMemory:
    """Combined short-term ring and optional persisted long-term store."""

    def __init__(
        self,
        *,
        capacity: int = 50,
        root: str | Path = DEFAULT_MEMORY_ROOT,
        filename: str = "episodes.jsonl",
        load_recent: bool = True,
    ) -> None:
        self.short_term = EpisodicMemory(capacity)
        self.long_term = JsonlMemoryStore(Path(root) / filename)
        if load_recent:
            for item in self.long_term.load(limit=capacity):
                self.short_term.append(item)

    def remember(
        self,
        episode: Mapping[str, Any] | Any,
        *,
        persist: bool = True,
    ) -> dict[str, Any]:
        item = _entry(episode)
        self.short_term.append(item)
        if persist:
            self.long_term.append(item)
        return item

    add_episode = remember
    append = remember

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.short_term.recent(limit)

    def recall(self, query: str = "", *, limit: int = 10) -> list[dict[str, Any]]:
        needle = str(query or "").strip().lower()
        local = self.short_term.recent()
        if needle:
            local = [
                item
                for item in local
                if needle in json.dumps(item, sort_keys=True, default=str).lower()
            ]
        if len(local) >= limit:
            return local[-limit:]
        persisted = self.long_term.search(query, limit=limit)
        seen = {
            json.dumps(item, sort_keys=True, default=str)
            for item in local
        }
        for item in persisted:
            key = json.dumps(item, sort_keys=True, default=str)
            if key not in seen:
                local.append(item)
                seen.add(key)
        return local[-max(0, int(limit)) :]

    def snapshot(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.recent(limit)


__all__ = [
    "DEFAULT_MEMORY_ROOT",
    "EpisodicMemory",
    "JsonlMemoryStore",
    "LongTermMemory",
    "LongTermJsonlMemory",
    "PlannerMemory",
    "ShortTermEpisodicMemory",
    "ShortTermMemory",
]
