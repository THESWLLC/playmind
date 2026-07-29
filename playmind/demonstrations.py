"""Human demonstration recording for Learning Architecture V2 behavior cloning.

Stores session metadata as JSONL under ``data/playmind/demonstrations/<session>/``
with optional frame file references. Schema version 2; writes are atomic-ish
(temp file + ``os.replace`` for session status; flushed appends for samples).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence

from playmind.observations import Observation

SCHEMA_VERSION = 2

Outcome = Literal["success", "failure", "bad"]
InputSource = Literal["human", "playmind_generated", "unknown"]
DEFAULT_ROOT = Path("data/playmind/demonstrations")


def _obs_to_dict(observation: Any) -> dict[str, Any]:
    if observation is None:
        return {}
    if isinstance(observation, Observation):
        return observation.to_legacy_dict()
    if isinstance(observation, Mapping):
        return dict(observation)
    raise TypeError(f"observation must be dict or Observation, got {type(observation)!r}")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(dict(payload), f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


@dataclass
class DemonstrationSample:
    """One recorded demonstration timestep."""

    sample_id: str
    session_id: str
    episode_id: str
    timestamp: float
    frame_path: Optional[str] = None
    observation: dict[str, Any] = field(default_factory=dict)
    key_events: list[Any] = field(default_factory=list)
    physical_events: list[Any] = field(default_factory=list)
    input_source: InputSource = "unknown"
    lifecycle_state: Any = None
    sensor_confidence: dict[str, Any] = field(default_factory=dict)
    inferred_skill: Optional[str] = None
    segmentation_meta: dict[str, Any] = field(default_factory=dict)
    training_eligible: bool = True
    is_human_demonstration: bool = False
    human_training_eligible: bool = False
    goal: Any = None
    profile: Any = None
    notes: Optional[str] = None
    skill: Optional[str] = None
    label: Optional[str] = None
    schema_version: int = SCHEMA_VERSION
    index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "frame_path": self.frame_path,
            "observation": dict(self.observation),
            "key_events": list(self.key_events),
            "physical_events": list(self.physical_events),
            "input_source": self.input_source,
            "lifecycle_state": self.lifecycle_state,
            "sensor_confidence": dict(self.sensor_confidence),
            "inferred_skill": self.inferred_skill,
            "segmentation_meta": dict(self.segmentation_meta),
            "training_eligible": self.training_eligible,
            "is_human_demonstration": self.is_human_demonstration,
            "human_training_eligible": self.human_training_eligible,
            "goal": self.goal,
            "profile": self.profile,
            "notes": self.notes,
            "skill": self.skill,
            "label": self.label,
            "index": self.index,
        }


class DemonstrationRecorder:
    """Record human (or scripted) demos for offline BC training / replay."""

    SCHEMA_VERSION = SCHEMA_VERSION

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        session_id: str | None = None,
        input_source: InputSource = "unknown",
    ) -> None:
        if input_source not in {"human", "playmind_generated", "unknown"}:
            raise ValueError(f"unknown input_source: {input_source!r}")
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        self.session_id: str | None = session_id
        self.default_input_source: InputSource = input_source
        self.session_dir: Path | None = None
        self.recording: bool = False
        self.sample_count: int = 0
        self.episode_id: str | None = None
        self.default_goal: Any = None
        self.default_profile: Any = None
        self.outcome: Outcome | None = None
        self.outcome_notes: str | None = None
        self.started_at: float | None = None
        self.stopped_at: float | None = None
        self._meta_path: Path | None = None

    @property
    def meta_path(self) -> Path | None:
        return self._meta_path

    def start(
        self,
        *,
        session_id: str | None = None,
        episode_id: str | None = None,
        goal: Any = None,
        profile: Any = None,
        input_source: InputSource | None = None,
    ) -> str:
        """Begin a recording session. Returns the session id."""
        if self.recording:
            raise RuntimeError("Already recording; call stop() first")
        sid = session_id or self.session_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.session_id = sid
        self.session_dir = self.root / sid
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "frames").mkdir(exist_ok=True)
        self._meta_path = self.session_dir / "meta.jsonl"
        self.recording = True
        self.sample_count = 0
        self.episode_id = episode_id or str(uuid.uuid4())
        self.default_goal = goal
        self.default_profile = profile
        if input_source is not None:
            if input_source not in {"human", "playmind_generated", "unknown"}:
                raise ValueError(f"unknown input_source: {input_source!r}")
            self.default_input_source = input_source
        self.outcome = None
        self.outcome_notes = None
        self.started_at = time.time()
        self.stopped_at = None
        self._write_session_status()
        return sid

    def stop(self) -> Path:
        """Flush and end the session. Returns the session directory."""
        if not self.recording or self.session_dir is None:
            raise RuntimeError("Not recording")
        self.recording = False
        self.stopped_at = time.time()
        self._write_session_status()
        return self.session_dir

    def append(
        self,
        *,
        frame_path: str | Path | None = None,
        observation: dict[str, Any] | Observation | None = None,
        key_events: Sequence[Any] | None = None,
        physical_events: Sequence[Any] | None = None,
        input_source: InputSource | None = None,
        lifecycle_state: Any = None,
        sensor_confidence: Mapping[str, Any] | None = None,
        sensor_confidence_blob: Mapping[str, Any] | None = None,
        inferred_skill: str | None = None,
        segmentation_meta: Mapping[str, Any] | None = None,
        training_eligible: bool | None = None,
        goal: Any = None,
        profile: Any = None,
        notes: str | None = None,
        timestamp: float | None = None,
        episode_id: str | None = None,
        skill: str | None = None,
        label: str | None = None,
        sample_id: str | None = None,
    ) -> dict[str, Any]:
        """Append one sample to ``meta.jsonl`` and return the row dict."""
        if not self.recording or self.session_dir is None or self._meta_path is None:
            raise RuntimeError("Not recording; call start() first")

        obs_dict = _obs_to_dict(observation)
        ts = float(timestamp if timestamp is not None else time.time())
        ep = str(episode_id or self.episode_id or "unknown")
        self.episode_id = ep
        source = input_source or self.default_input_source
        if source not in {"human", "playmind_generated", "unknown"}:
            raise ValueError(f"unknown input_source: {source!r}")
        segmentation = dict(segmentation_meta or {})
        segment_eligible = bool(segmentation.get("training_eligible", True))
        requested_eligible = (
            bool(training_eligible) if training_eligible is not None else segment_eligible
        )
        eligible = source != "playmind_generated" and requested_eligible
        is_human = source == "human"

        rel_frame: Optional[str] = None
        if frame_path is not None:
            fp = Path(frame_path)
            if fp.is_absolute():
                try:
                    rel_frame = str(fp.relative_to(self.session_dir))
                except ValueError:
                    # Copy reference as-is when outside session dir.
                    rel_frame = str(fp)
            else:
                rel_frame = str(fp)

        sample = DemonstrationSample(
            sample_id=sample_id or str(uuid.uuid4()),
            session_id=str(self.session_id),
            episode_id=ep,
            timestamp=ts,
            frame_path=rel_frame,
            observation=obs_dict,
            key_events=list(key_events or []),
            physical_events=list(physical_events or []),
            input_source=source,
            lifecycle_state=lifecycle_state,
            sensor_confidence=dict(sensor_confidence or sensor_confidence_blob or {}),
            inferred_skill=inferred_skill,
            segmentation_meta=segmentation,
            training_eligible=eligible,
            is_human_demonstration=is_human,
            human_training_eligible=is_human and eligible,
            goal=goal if goal is not None else self.default_goal,
            profile=profile if profile is not None else self.default_profile,
            notes=notes,
            skill=skill,
            label=label,
            schema_version=SCHEMA_VERSION,
            index=self.sample_count,
        )
        row = sample.to_dict()
        self._append_jsonl(row)
        self.sample_count += 1
        self._write_session_status()
        return row

    def mark(
        self,
        outcome: Outcome,
        *,
        notes: str | None = None,
        sample_id: str | None = None,
    ) -> None:
        """Mark the session (or a specific sample via ``sample_id``) success/failure/bad."""
        if outcome not in ("success", "failure", "bad"):
            raise ValueError(f"outcome must be success|failure|bad, got {outcome!r}")
        if sample_id is not None:
            self._relabel_sample(sample_id, outcome, notes=notes)
            return
        if self.session_dir is None:
            raise RuntimeError("No session; call start() first")
        self.outcome = outcome
        self.outcome_notes = notes
        self._write_session_status()

    def _append_jsonl(self, row: Mapping[str, Any]) -> None:
        assert self._meta_path is not None
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(dict(row), sort_keys=True) + "\n"
        with self._meta_path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def _write_session_status(self) -> None:
        if self.session_dir is None:
            return
        status = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "recording": self.recording,
            "sample_count": self.sample_count,
            "episode_id": self.episode_id,
            "goal": self.default_goal,
            "profile": self.default_profile,
            "input_source": self.default_input_source,
            "outcome": self.outcome,
            "outcome_notes": self.outcome_notes,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "meta_jsonl": "meta.jsonl",
        }
        _atomic_write_json(self.session_dir / "session.json", status)

    def _relabel_sample(
        self,
        sample_id: str,
        outcome: Outcome,
        *,
        notes: str | None = None,
    ) -> None:
        """Rewrite meta.jsonl with one sample's label updated (atomic replace)."""
        if self._meta_path is None or not self._meta_path.exists():
            raise RuntimeError("No meta.jsonl to relabel")
        rows: list[dict[str, Any]] = []
        found = False
        with self._meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("sample_id") == sample_id:
                    row["label"] = outcome
                    if notes is not None:
                        row["notes"] = notes
                    found = True
                rows.append(row)
        if not found:
            raise KeyError(f"sample_id not found: {sample_id}")
        tmp = self._meta_path.with_suffix(".jsonl.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._meta_path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass


def load_session_samples(session_dir: str | Path) -> list[dict[str, Any]]:
    """Load samples, normalizing absent v2 fields on legacy v1 rows."""
    path = Path(session_dir) / "meta.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                continue
            normalized = dict(row)
            source = str(normalized.get("input_source") or "unknown")
            if source not in {"human", "playmind_generated", "unknown"}:
                source = "unknown"
            normalized.setdefault("physical_events", [])
            normalized["input_source"] = source
            normalized.setdefault("lifecycle_state", None)
            normalized.setdefault("sensor_confidence", {})
            normalized.setdefault("inferred_skill", None)
            normalized.setdefault("segmentation_meta", {})
            default_eligible = source != "playmind_generated"
            normalized.setdefault("training_eligible", default_eligible)
            normalized.setdefault("is_human_demonstration", source == "human")
            normalized.setdefault(
                "human_training_eligible",
                source == "human" and bool(normalized["training_eligible"]),
            )
            rows.append(normalized)
    return rows


def list_sessions(root: str | Path | None = None) -> list[Path]:
    """List demonstration session directories under ``root``."""
    base = Path(root) if root is not None else DEFAULT_ROOT
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir() and (p / "meta.jsonl").exists())
