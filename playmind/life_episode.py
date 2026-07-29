"""Life-state gating and episode lifecycle for PlayMind.

This module deliberately does not drive recovery actions.  It turns noisy
life/UI observations and confirmed events into safe gameplay/recovery episode
boundaries that a controller can consume.
"""

from __future__ import annotations

import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from playmind.episodes import EpisodeManager, EpisodeRecord
from playmind.events import EventType


class LifecycleState(str, Enum):
    UNKNOWN = "unknown"
    ALIVE_CONTROLLABLE = "alive_controllable"
    COMBAT = "combat"
    DEAD_DIALOG = "dead_dialog"
    RELEASE_CONFIRM = "release_confirm"
    GHOST = "ghost"
    RUNBACK = "runback"
    RESURRECTION_PENDING = "resurrection_pending"
    LOADING = "loading"
    ALIVE_NOT_CONTROLLABLE = "alive_not_controllable"
    ALIVE_CONTROLLABLE_AFTER_RESURRECTION = "alive_controllable_after_resurrection"


NON_GAMEPLAY_STATES = frozenset(
    {
        LifecycleState.DEAD_DIALOG,
        LifecycleState.RELEASE_CONFIRM,
        LifecycleState.GHOST,
        LifecycleState.RUNBACK,
        LifecycleState.RESURRECTION_PENDING,
        LifecycleState.LOADING,
        LifecycleState.ALIVE_NOT_CONTROLLABLE,
        LifecycleState.UNKNOWN,
    }
)


def _phase(obs: Mapping[str, Any]) -> str:
    return str(obs.get("life_phase") or "unknown").strip().lower()


def is_loading(obs: Mapping[str, Any]) -> bool:
    """Return whether the observation is inside a loading transition."""
    return bool(obs.get("loading") or obs.get("is_loading")) or _phase(obs) == "loading"


def is_ghost(obs: Mapping[str, Any]) -> bool:
    """Return whether the player is in ghost/runback form."""
    return bool(obs.get("is_ghost")) or _phase(obs) in {"ghost", "runback"}


def is_alive(obs: Mapping[str, Any]) -> bool:
    """Return conservative alive, non-ghost evidence."""
    phase = _phase(obs)
    return (
        not bool(obs.get("is_dead"))
        and not is_ghost(obs)
        and not is_loading(obs)
        and phase
        not in {
            "unknown",
            "dead",
            "dead_dialog",
            "confirm",
            "release_confirm",
            "rez_picker",
            "resurrection_pending",
            "alive_not_controllable",
        }
    )


def controls_responsive(obs: Mapping[str, Any]) -> bool:
    """Return explicit evidence that gameplay controls are responsive."""
    return bool(
        obs.get("controls_responsive")
        or obs.get("control_responsive")
        or obs.get("controllable")
        or obs.get("can_control")
        or obs.get("input_responsive")
    )


def ui_stable(obs: Mapping[str, Any]) -> bool:
    """Return whether no known unstable/blocking UI is present."""
    if obs.get("ui_stable") is False:
        return False
    return not bool(
        obs.get("blocking_modal")
        or obs.get("modal_menu")
        or obs.get("confirm_pending")
        or obs.get("ui_transitioning")
    )


def classify_lifecycle_state(
    obs: Mapping[str, Any],
    *,
    controllable: bool | None = None,
) -> LifecycleState:
    """Classify one observation without applying temporal hysteresis."""
    phase = _phase(obs)
    if is_loading(obs):
        return LifecycleState.LOADING
    if phase in {"confirm", "release_confirm", "rez_picker"} or obs.get("confirm_pending"):
        return LifecycleState.RELEASE_CONFIRM
    if phase in {"dead_dialog", "dead"} or bool(obs.get("is_dead")):
        return LifecycleState.DEAD_DIALOG
    if phase == "runback":
        return LifecycleState.RUNBACK
    if is_ghost(obs):
        return LifecycleState.GHOST
    if phase == "resurrection_pending":
        return LifecycleState.RESURRECTION_PENDING
    if not is_alive(obs):
        return LifecycleState.UNKNOWN
    ready = controls_responsive(obs) if controllable is None else controllable
    if not ready or not ui_stable(obs):
        return LifecycleState.ALIVE_NOT_CONTROLLABLE
    if bool(obs.get("in_combat")):
        return LifecycleState.COMBAT
    return LifecycleState.ALIVE_CONTROLLABLE


def _event_name(event: Any) -> str:
    if isinstance(event, str):
        return event
    if isinstance(event, Mapping):
        raw = event.get("type") or event.get("name")
    else:
        raw = getattr(event, "type", None) or getattr(event, "name", None)
    return raw.value if isinstance(raw, Enum) else str(raw or "")


def _event_metadata(event: Any) -> Mapping[str, Any]:
    if isinstance(event, Mapping):
        metadata = event.get("metadata") or event.get("payload") or {}
    else:
        metadata = getattr(event, "metadata", None) or getattr(event, "payload", None) or {}
    return metadata if isinstance(metadata, Mapping) else {}


class EpisodeLifecycleController:
    """Gate gameplay episodes across death, loading, and recovery.

    ``update`` may both close and open an episode. Its returned dictionary
    reports those edges explicitly and always describes the active episode
    after the update.
    """

    def __init__(
        self,
        episode_manager: EpisodeManager | None = None,
        *,
        persist_dir: str | Path | None = None,
        frames_alive_controllable: int = 3,
        max_duration_s: float | None = None,
    ) -> None:
        if frames_alive_controllable < 1:
            raise ValueError("frames_alive_controllable must be >= 1")
        self.episode_manager = episode_manager or EpisodeManager(
            persist_dir=persist_dir,
            max_duration_s=None,
        )
        # Short alias for controller integrations.
        self.episodes = self.episode_manager
        self.frames_alive_controllable = int(frames_alive_controllable)
        self.max_duration_s = max_duration_s
        self.state = LifecycleState.UNKNOWN

        self._controllable_frames = 0
        self._controls_confirmed = False
        self._active_started_monotonic: float | None = None
        self._previous_state = LifecycleState.UNKNOWN
        self._previous_gameplay_episode_id: str | None = None
        self._death_event_id: str | None = None
        self._death_ts: float | None = None
        self._recovery_segment_id: str | None = None
        self._resurrection_ts: float | None = None
        self._controllable_ts: float | None = None
        self._recovery_result: str | None = None
        self._goal_latched = False
        self._session_ended = False

    @property
    def current_episode(self) -> EpisodeRecord | None:
        current = self.episode_manager.current
        return current if current is not None and not current.done else None

    def update(
        self,
        obs_dict: Mapping[str, Any],
        events: Iterable[Any] | None,
        now_monotonic: float,
    ) -> dict[str, Any]:
        """Advance lifecycle state and return episode/status edges."""
        obs = dict(obs_dict or {})
        event_list = list(events or [])
        event_names = {_event_name(event) for event in event_list}
        now = float(now_monotonic)
        status: dict[str, Any] = {
            "episode_started": False,
            "episode_ended": False,
            "started_episode_id": None,
            "ended_episode_id": None,
            "end_reason": None,
            "terminal": False,
            "truncated": False,
        }

        objective_flag = bool(
            obs.get("goal_complete") or obs.get("objective_completed") or obs.get("quest_complete")
        )
        objective_event = EventType.OBJECTIVE_COMPLETED.value in event_names
        goal_complete = objective_event or (objective_flag and not self._goal_latched)
        self._goal_latched = objective_flag

        death_event = next(
            (event for event in event_list if _event_name(event) == EventType.DEATH_CONFIRMED.value),
            None,
        )
        observed_state = classify_lifecycle_state(obs)
        death_confirmed = death_event is not None or observed_state in {
            LifecycleState.DEAD_DIALOG,
            LifecycleState.RELEASE_CONFIRM,
        }
        manual_reset = bool(obs.get("manual_reset")) or "ManualReset" in event_names
        fatal_sensor = bool(obs.get("fatal_sensor") or obs.get("sensor_fatal")) or (
            "FatalSensor" in event_names
        )
        session_end = bool(obs.get("session_end") or obs.get("logout")) or bool(
            {"SessionEnd", "Logout"} & event_names
        )

        active = self.current_episode
        max_duration = (
            active is not None
            and self.max_duration_s is not None
            and self._active_started_monotonic is not None
            and now - self._active_started_monotonic >= self.max_duration_s
        )

        allow_start = not (goal_complete or manual_reset or fatal_sensor or session_end)
        if active is not None and death_confirmed and active.episode_kind == "gameplay":
            self._on_death(death_event, now, status)
            observed_state = LifecycleState.DEAD_DIALOG
            allow_start = False
        elif active is not None and goal_complete:
            self._finish_active("goal_complete", terminal=True, status=status)
            allow_start = False
        elif active is not None and manual_reset:
            self._mark_recovery_result("manual_reset")
            self._finish_active("manual_reset", terminal=False, status=status)
            allow_start = False
        elif active is not None and fatal_sensor:
            self._mark_recovery_result("fatal_sensor")
            self._finish_active("sensor_failure", terminal=False, status=status)
            allow_start = False
        elif active is not None and max_duration:
            self._mark_recovery_result("max_duration")
            self._finish_active("max_duration", terminal=False, status=status)
            allow_start = False
        elif active is not None and session_end:
            self._mark_recovery_result("session_end")
            self._finish_active("session_end", terminal=True, status=status)
            self._session_ended = True
            allow_start = False
        elif session_end:
            self._session_ended = True
            allow_start = False

        active = self.current_episode
        if death_confirmed and active is None and not manual_reset and not self._session_ended:
            self._death_event_id = self._id_for_death_event(death_event)
            self._death_ts = now
            self._recovery_segment_id = None
            self._resurrection_ts = None
            self._controllable_ts = None
            self._recovery_result = None
            self._begin_recovery(death_event, now, status)
            active = self.current_episode
            observed_state = LifecycleState.DEAD_DIALOG
            allow_start = False

        resurrection_event = EventType.RESURRECTION_CONFIRMED.value in event_names
        in_recovery = active is not None and active.episode_kind == "recovery"
        if in_recovery and self._resurrection_ts is None and (
            resurrection_event
            or (
                is_alive(obs)
                and self._previous_state
                in {
                    LifecycleState.DEAD_DIALOG,
                    LifecycleState.RELEASE_CONFIRM,
                    LifecycleState.GHOST,
                    LifecycleState.RUNBACK,
                    LifecycleState.LOADING,
                    LifecycleState.RESURRECTION_PENDING,
                }
            )
        ):
            self._resurrection_ts = now
            active.resurrection_ts = now
            active.time_death_to_rez = self._elapsed(self._death_ts, now)

        event_controllable = EventType.BECAME_CONTROLLABLE.value in event_names
        control_keys = (
            "controls_responsive",
            "control_responsive",
            "controllable",
            "can_control",
            "input_responsive",
        )
        if event_controllable:
            self._controls_confirmed = True
        elif any(key in obs for key in control_keys):
            self._controls_confirmed = controls_responsive(obs)
        if not is_alive(obs):
            self._controls_confirmed = False
        frame_ready = (
            is_alive(obs)
            and ui_stable(obs)
            and self._controls_confirmed
            and not death_confirmed
            and not objective_flag
            and not manual_reset
            and not fatal_sensor
            and not self._session_ended
        )
        if frame_ready:
            self._controllable_frames += 1
        else:
            self._controllable_frames = 0

        gate_open = self._controllable_frames >= self.frames_alive_controllable
        if observed_state in {LifecycleState.ALIVE_CONTROLLABLE, LifecycleState.COMBAT} and not gate_open:
            observed_state = LifecycleState.ALIVE_NOT_CONTROLLABLE

        if allow_start and gate_open:
            active = self.current_episode
            if active is not None and active.episode_kind == "recovery":
                self._complete_recovery_and_start_gameplay(now, status)
                observed_state = LifecycleState.ALIVE_CONTROLLABLE_AFTER_RESURRECTION
            elif active is None:
                self._start_gameplay(now, status)
                observed_state = (
                    LifecycleState.COMBAT
                    if bool(obs.get("in_combat"))
                    else LifecycleState.ALIVE_CONTROLLABLE
                )

        self.state = observed_state
        # Keep the just-observed state for transition detection on the next call.
        self._previous_state = observed_state
        return self._status(status)

    def _on_death(
        self,
        event: Any,
        now: float,
        status: dict[str, Any],
    ) -> None:
        active = self.current_episode
        if active is None:
            return
        self._previous_gameplay_episode_id = active.episode_id
        self._death_event_id = self._id_for_death_event(event)
        self._death_ts = now
        self._recovery_segment_id = str(uuid.uuid4())
        self._resurrection_ts = None
        self._controllable_ts = None
        self._recovery_result = None
        active.death_event_id = self._death_event_id
        active.recovery_segment_id = self._recovery_segment_id
        active.death_count = max(1, active.death_count)
        self._finish_active("death", terminal=True, status=status)
        self._begin_recovery(event, now, status)

    def _begin_recovery(
        self,
        event: Any,
        now: float,
        status: dict[str, Any],
    ) -> EpisodeRecord:
        if self._death_event_id is None:
            self._death_event_id = self._id_for_death_event(event)
        if self._death_ts is None:
            self._death_ts = now
        recovery_id = self._recovery_segment_id or str(uuid.uuid4())
        recovery = self.episode_manager.start(
            "death_recovery",
            episode_id=recovery_id,
            episode_kind="recovery",
            previous_episode_id=self._previous_gameplay_episode_id,
            death_event_id=self._death_event_id,
            recovery_segment_id=recovery_id,
            metadata={"death_ts": self._death_ts},
        )
        self._recovery_segment_id = recovery_id
        self._active_started_monotonic = now
        self._record_started(recovery, status)
        return recovery

    def _complete_recovery_and_start_gameplay(
        self,
        now: float,
        status: dict[str, Any],
    ) -> None:
        recovery = self.current_episode
        if recovery is None or recovery.episode_kind != "recovery":
            return
        if self._resurrection_ts is None:
            self._resurrection_ts = now
        self._controllable_ts = now
        self._recovery_result = "success"
        recovery.resurrection_ts = self._resurrection_ts
        recovery.controllable_ts = now
        recovery.time_death_to_rez = self._elapsed(self._death_ts, self._resurrection_ts)
        recovery.time_rez_to_controllable = self._elapsed(self._resurrection_ts, now)
        recovery.recovery_result = self._recovery_result
        self._finish_active("recovery_complete", terminal=True, status=status)
        self._start_gameplay(now, status, after_recovery=True)

    def _start_gameplay(
        self,
        now: float,
        status: dict[str, Any],
        *,
        after_recovery: bool = False,
    ) -> EpisodeRecord:
        gameplay = self.episode_manager.start(
            "controllable" if after_recovery else "new_run",
            episode_kind="gameplay",
            previous_episode_id=self._previous_gameplay_episode_id if after_recovery else None,
            death_event_id=self._death_event_id if after_recovery else None,
            recovery_segment_id=self._recovery_segment_id if after_recovery else None,
            metadata={"after_recovery": after_recovery},
        )
        if after_recovery:
            gameplay.resurrection_ts = self._resurrection_ts
            gameplay.controllable_ts = self._controllable_ts
            gameplay.time_death_to_rez = self._elapsed(self._death_ts, self._resurrection_ts)
            gameplay.time_rez_to_controllable = self._elapsed(
                self._resurrection_ts, self._controllable_ts
            )
            gameplay.recovery_result = self._recovery_result
        self._active_started_monotonic = now
        self._record_started(gameplay, status)
        return gameplay

    def _finish_active(
        self,
        reason: str,
        *,
        terminal: bool,
        status: dict[str, Any],
    ) -> EpisodeRecord | None:
        active = self.current_episode
        if active is None:
            return None
        if terminal:
            finished = self.episode_manager.end(reason)
        else:
            finished = self.episode_manager.truncate(reason)
        status.update(
            {
                "episode_ended": True,
                "ended_episode_id": finished.episode_id,
                "end_reason": reason,
                "terminal": finished.terminal,
                "truncated": finished.truncated,
            }
        )
        self._active_started_monotonic = None
        return finished

    def _mark_recovery_result(self, result: str) -> None:
        active = self.current_episode
        if active is not None and active.episode_kind == "recovery":
            self._recovery_result = result
            active.recovery_result = result

    @staticmethod
    def _record_started(episode: EpisodeRecord, status: dict[str, Any]) -> None:
        status["episode_started"] = True
        status["started_episode_id"] = episode.episode_id

    @staticmethod
    def _elapsed(start: float | None, end: float | None) -> float | None:
        if start is None or end is None:
            return None
        return max(0.0, float(end) - float(start))

    @staticmethod
    def _id_for_death_event(event: Any) -> str:
        metadata = _event_metadata(event)
        event_id = metadata.get("event_id") or metadata.get("id")
        return str(event_id) if event_id else str(uuid.uuid4())

    def _status(self, edges: dict[str, Any]) -> dict[str, Any]:
        active = self.current_episode
        result = dict(edges)
        result.update(
            {
                "state": self.state.value,
                "lifecycle_state": self.state.value,
                "episode_active": active is not None,
                "episode_id": active.episode_id if active else None,
                "episode_kind": active.episode_kind if active else None,
                "active_episode_kind": active.episode_kind if active else None,
                "previous_episode_id": (
                    active.previous_episode_id if active else self._previous_gameplay_episode_id
                ),
                "death_event_id": active.death_event_id if active else self._death_event_id,
                "recovery_segment_id": (
                    active.recovery_segment_id if active else self._recovery_segment_id
                ),
                "resurrection_ts": self._resurrection_ts,
                "controllable_ts": self._controllable_ts,
                "time_death_to_rez": self._elapsed(self._death_ts, self._resurrection_ts),
                "time_rez_to_controllable": self._elapsed(
                    self._resurrection_ts, self._controllable_ts
                ),
                "recovery_result": self._recovery_result,
                "controllable_frames": self._controllable_frames,
                "controllable_frames_required": self.frames_alive_controllable,
            }
        )
        return result


__all__ = [
    "EpisodeLifecycleController",
    "LifecycleState",
    "NON_GAMEPLAY_STATES",
    "classify_lifecycle_state",
    "controls_responsive",
    "is_alive",
    "is_ghost",
    "is_loading",
    "ui_stable",
]
