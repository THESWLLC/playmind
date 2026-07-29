"""Episode boundaries and JSONL persistence for Learning Architecture V2."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 1

StartReason = Literal[
    "new_run",
    "controllable",
    "resurrected",
    "new_objective",
    "manual_reset",
    "unknown",
]

EndReason = Literal[
    "death",
    "goal_complete",
    "session_end",
    "logout",
    "loading_timeout",
    "manual_reset",
    "sensor_failure",
    "max_duration",
    "truncated",
    "unknown",
]


@dataclass
class EpisodeRecord:
    """One contiguous controllable segment with an explicit terminal/truncate end."""

    episode_id: str
    start_reason: str
    end_reason: str | None = None
    terminal: bool = False
    truncated: bool = False
    total_reward: float = 0.0
    goal_progress: float = 0.0
    death_count: int = 0
    skill_attempts: int = 0
    skill_successes: int = 0
    skill_failures: int = 0
    duration_s: float = 0.0
    model_version: str | None = None
    configuration_version: str | None = None
    schema_version: int = SCHEMA_VERSION
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    steps: int = 0
    done: bool = False  # True only after end/truncate — never always False by design
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Terminal end reasons (episode truly finished, not just cut short).
_TERMINAL_REASONS = frozenset({"death", "goal_complete", "logout", "session_end"})
_TRUNCATE_REASONS = frozenset(
    {"loading_timeout", "manual_reset", "sensor_failure", "max_duration", "truncated"}
)


class EpisodeManager:
    """Tracks start/end/truncate of episodes and appends JSONL records."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        *,
        model_version: str | None = None,
        configuration_version: str | None = None,
        max_duration_s: float | None = None,
    ) -> None:
        root = Path(persist_dir) if persist_dir else Path("data/playmind/episodes")
        self.persist_dir = root
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.model_version = model_version
        self.configuration_version = configuration_version
        self.max_duration_s = max_duration_s
        self.current: EpisodeRecord | None = None
        self._path = self.persist_dir / "episodes.jsonl"

    @property
    def done(self) -> bool:
        """Whether the current episode has ended (False only while active)."""
        if self.current is None:
            return True
        return bool(self.current.done)

    def start(
        self,
        reason: StartReason | str = "new_run",
        *,
        episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EpisodeRecord:
        if self.current is not None and not self.current.done:
            # Auto-truncate previous open episode before starting a new one.
            self.truncate("truncated", note="superseded_by_new_start")
        ep = EpisodeRecord(
            episode_id=episode_id or str(uuid.uuid4()),
            start_reason=str(reason),
            model_version=self.model_version,
            configuration_version=self.configuration_version,
            metadata=dict(metadata or {}),
            done=False,
        )
        self.current = ep
        return ep

    def note_reward(self, reward: float) -> None:
        if self.current is None or self.current.done:
            return
        self.current.total_reward += float(reward)
        self.current.steps += 1
        self.current.duration_s = max(0.0, time.time() - self.current.started_at)
        if self.max_duration_s is not None and self.current.duration_s >= self.max_duration_s:
            self.truncate("max_duration")

    def note_skill_attempt(self, *, success: bool | None = None) -> None:
        if self.current is None or self.current.done:
            return
        self.current.skill_attempts += 1
        if success is True:
            self.current.skill_successes += 1
        elif success is False:
            self.current.skill_failures += 1

    def note_goal_progress(self, progress: float) -> None:
        if self.current is None or self.current.done:
            return
        self.current.goal_progress = float(progress)

    def note_death(self) -> None:
        if self.current is None or self.current.done:
            return
        self.current.death_count += 1

    def should_end_for_obs(self, obs: dict[str, Any]) -> EndReason | None:
        """Heuristic terminal detection from an observation dict."""
        if not obs:
            return None
        if obs.get("is_dead") or str(obs.get("life_phase") or "") in {
            "dead_dialog",
            "confirm",
            "rez_picker",
        }:
            # Only end when transitioning into death is handled by caller;
            # if already dead at start of episode this still signals terminal.
            return "death"
        if obs.get("goal_complete") or obs.get("objective_completed"):
            return "goal_complete"
        if obs.get("logout") or obs.get("session_end"):
            return "logout"
        if obs.get("sensor_failure"):
            return "sensor_failure"
        return None

    def end(self, reason: EndReason | str, *, note: str | None = None) -> EpisodeRecord:
        return self._finish(reason, terminal=True, truncated=False, note=note)

    def truncate(self, reason: EndReason | str = "truncated", *, note: str | None = None) -> EpisodeRecord:
        return self._finish(reason, terminal=False, truncated=True, note=note)

    def end_or_truncate(self, reason: EndReason | str, *, note: str | None = None) -> EpisodeRecord:
        r = str(reason)
        if r in _TERMINAL_REASONS or r == "death":
            return self.end(r, note=note)
        return self.truncate(r, note=note)

    def _finish(
        self,
        reason: EndReason | str,
        *,
        terminal: bool,
        truncated: bool,
        note: str | None,
    ) -> EpisodeRecord:
        if self.current is None:
            raise RuntimeError("No active episode to end")
        if self.current.done:
            return self.current
        r = str(reason)
        # Normalize terminal vs truncate flags from reason when caller is ambiguous.
        if r in _TRUNCATE_REASONS:
            terminal, truncated = False, True
        elif r in _TERMINAL_REASONS:
            terminal, truncated = True, False
        ep = self.current
        ep.end_reason = r
        ep.terminal = terminal
        ep.truncated = truncated
        ep.ended_at = time.time()
        ep.duration_s = max(0.0, ep.ended_at - ep.started_at)
        ep.done = True  # Explicit — never leave done stuck at False after finish
        if note:
            ep.metadata["end_note"] = note
        if r == "death":
            ep.death_count = max(ep.death_count, 1)
        self._append(ep)
        return ep

    def _append(self, ep: EpisodeRecord) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        row = ep.to_dict()
        row["schema_version"] = SCHEMA_VERSION
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
