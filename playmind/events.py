"""Evidence-based event detection for Learning Architecture V2."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class EventType(str, Enum):
    TARGET_ACQUIRED = "TargetAcquired"
    TARGET_VALIDATED = "TargetValidated"
    DAMAGE_DEALT = "DamageDealt"
    DAMAGE_RECEIVED = "DamageReceived"
    TARGET_LOST = "TargetLost"
    SUSPECTED_KILL = "SuspectedKill"
    KILL_CONFIRMED = "KillConfirmed"
    OBJECTIVE_PROGRESSED = "ObjectiveProgressed"
    OBJECTIVE_COMPLETED = "ObjectiveCompleted"
    LOOT_CONFIRMED = "LootConfirmed"
    DEATH_CONFIRMED = "DeathConfirmed"
    RESURRECTION_CONFIRMED = "ResurrectionConfirmed"
    BECAME_CONTROLLABLE = "BecameControllable"
    LOADING_STARTED = "LoadingStarted"
    LOADING_ENDED = "LoadingEnded"
    SKILL_SUCCESS = "SkillSuccess"
    SKILL_FAILURE = "SkillFailure"
    # Legacy event names remain readable by old logs and callers.
    SKILL_SUCCEEDED = "SkillSucceeded"
    SKILL_FAILED = "SkillFailed"
    MODAL_CLEARED = "ModalCleared"
    MOVEMENT_BLOCKED = "MovementBlocked"
    SENSOR_CONFLICT = "SensorConflict"


@dataclass
class ConfirmedEvent:
    """Evidence-bearing event shared by detectors, lifecycle, and rewards."""

    type: EventType | str
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    evidence: list[str] = field(default_factory=list)
    conflicting_evidence: list[str] = field(default_factory=list)
    source_frames: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.type.value if isinstance(self.type, EventType) else str(self.type)


@dataclass
class Event(ConfirmedEvent):
    """Backward-compatible base event name."""


@dataclass
class TargetAcquired(Event):
    def __post_init__(self) -> None:
        self.type = EventType.TARGET_ACQUIRED


@dataclass
class TargetValidated(Event):
    def __post_init__(self) -> None:
        self.type = EventType.TARGET_VALIDATED


@dataclass
class DamageDealt(Event):
    def __post_init__(self) -> None:
        self.type = EventType.DAMAGE_DEALT


@dataclass
class DamageReceived(Event):
    def __post_init__(self) -> None:
        self.type = EventType.DAMAGE_RECEIVED


@dataclass
class TargetLost(Event):
    def __post_init__(self) -> None:
        self.type = EventType.TARGET_LOST


@dataclass
class SuspectedKill(Event):
    """A weak combat outcome which must never receive confirmed-kill reward."""

    def __post_init__(self) -> None:
        self.type = EventType.SUSPECTED_KILL


@dataclass
class KillConfirmed(Event):
    """Requires multiple independent pieces of evidence — never target-loss alone."""

    def __post_init__(self) -> None:
        self.type = EventType.KILL_CONFIRMED


@dataclass
class ObjectiveProgressed(Event):
    def __post_init__(self) -> None:
        self.type = EventType.OBJECTIVE_PROGRESSED


@dataclass
class ObjectiveCompleted(Event):
    def __post_init__(self) -> None:
        self.type = EventType.OBJECTIVE_COMPLETED


@dataclass
class LootConfirmed(Event):
    def __post_init__(self) -> None:
        self.type = EventType.LOOT_CONFIRMED


@dataclass
class DeathConfirmed(Event):
    def __post_init__(self) -> None:
        self.type = EventType.DEATH_CONFIRMED


@dataclass
class ResurrectionConfirmed(Event):
    def __post_init__(self) -> None:
        self.type = EventType.RESURRECTION_CONFIRMED


@dataclass
class BecameControllable(Event):
    def __post_init__(self) -> None:
        self.type = EventType.BECAME_CONTROLLABLE


@dataclass
class LoadingStarted(Event):
    def __post_init__(self) -> None:
        self.type = EventType.LOADING_STARTED


@dataclass
class LoadingEnded(Event):
    def __post_init__(self) -> None:
        self.type = EventType.LOADING_ENDED


@dataclass
class SkillSuccess(Event):
    def __post_init__(self) -> None:
        self.type = EventType.SKILL_SUCCESS


@dataclass
class SkillFailure(Event):
    def __post_init__(self) -> None:
        self.type = EventType.SKILL_FAILURE


@dataclass
class SkillSucceeded(Event):
    """Legacy spelling; new detectors emit :class:`SkillSuccess`."""

    def __post_init__(self) -> None:
        self.type = EventType.SKILL_SUCCEEDED


@dataclass
class SkillFailed(Event):
    """Legacy spelling; new detectors emit :class:`SkillFailure`."""

    def __post_init__(self) -> None:
        self.type = EventType.SKILL_FAILED


@dataclass
class ModalCleared(Event):
    def __post_init__(self) -> None:
        self.type = EventType.MODAL_CLEARED


@dataclass
class MovementBlocked(Event):
    def __post_init__(self) -> None:
        self.type = EventType.MOVEMENT_BLOCKED


@dataclass
class SensorConflict(Event):
    def __post_init__(self) -> None:
        self.type = EventType.SENSOR_CONFLICT


KILL_ORTHOGONAL_EVIDENCE_CLASSES = frozenset(
    {"hp_zero", "loot_or_xp", "objective_kill", "explicit_flag"}
)

_KILL_EVIDENCE_CLASS_BY_SIGNAL: dict[str, str] = {
    "target_hp_near_zero": "hp_zero",
    "target_hp_zero": "hp_zero",
    "loot_or_xp_text": "loot_or_xp",
    "loot_or_xp_flag": "loot_or_xp",
    "objective_kill_count_increased": "objective_kill",
    "objective_kill": "objective_kill",
    "explicit_kill_flag": "explicit_flag",
}


def kill_evidence_classes(evidence: Iterable[str]) -> set[str]:
    """Return independent kill-evidence classes represented by signals."""
    classes: set[str] = set()
    for raw in evidence:
        signal = str(raw).strip().lower()
        mapped = _KILL_EVIDENCE_CLASS_BY_SIGNAL.get(signal)
        if mapped:
            classes.add(mapped)
        elif signal in KILL_ORTHOGONAL_EVIDENCE_CLASSES:
            classes.add(signal)
    return classes


def _hp(obs: Mapping[str, Any], key: str = "vision_player_hp") -> float | None:
    raw = obs.get(key)
    if raw is None and key == "vision_player_hp":
        raw = obs.get("player_hp")
        if raw is None and isinstance(obs.get("player"), Mapping):
            raw = obs["player"].get("hp")  # type: ignore[index]
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _ocr(obs: Mapping[str, Any]) -> str:
    return str(obs.get("screen_ocr") or "").lower()


def _collect_kill_evidence(prev: Mapping[str, Any], nxt: Mapping[str, Any]) -> list[str]:
    """Gather independent kill signals — target loss alone is insufficient."""
    evidence: list[str] = []
    prev_thp = _hp(prev, "target_hp_est")
    next_thp = _hp(nxt, "target_hp_est")
    had_target = bool(prev.get("has_target"))
    has_target = bool(nxt.get("has_target"))
    was_combat = bool(prev.get("in_combat"))
    now_combat = bool(nxt.get("in_combat"))

    if prev_thp is not None and prev_thp > 0 and (next_thp is not None and next_thp <= 0.05):
        evidence.append("target_hp_near_zero")
    elif prev_thp is not None and next_thp is not None and next_thp <= 0.02 < prev_thp:
        evidence.append("target_hp_near_zero")

    if had_target and not has_target and was_combat:
        evidence.append("target_lost_after_combat")

    if was_combat and not now_combat and had_target:
        evidence.append("combat_ended")

    ocr1 = _ocr(nxt)
    ocr0 = _ocr(prev)
    loot_xp_markers = ("experience", " you receive", "loot", "killed", " slain", "xp")
    if any(m in ocr1 for m in loot_xp_markers) and not any(m in ocr0 for m in loot_xp_markers):
        evidence.append("loot_or_xp_text")
    if nxt.get("loot_confirmed") or nxt.get("xp_gained"):
        evidence.append("loot_or_xp_flag")

    prev_kills = prev.get("quest_kills")
    next_kills = nxt.get("quest_kills")
    try:
        if prev_kills is not None and next_kills is not None and int(next_kills) > int(prev_kills):
            evidence.append("objective_kill_count_increased")
    except (TypeError, ValueError):
        pass

    if nxt.get("kill_confirmed"):
        evidence.append("explicit_kill_flag")

    return evidence


def _is_loading(obs: Mapping[str, Any]) -> bool:
    return bool(obs.get("loading") or obs.get("is_loading")) or str(
        obs.get("life_phase") or ""
    ).lower() == "loading"


def _is_controllable(obs: Mapping[str, Any]) -> bool:
    phase = str(obs.get("life_phase") or "").lower()
    alive = (
        not bool(obs.get("is_dead"))
        and not bool(obs.get("is_ghost"))
        and phase not in {"dead_dialog", "confirm", "release_confirm", "rez_picker", "ghost", "loading"}
    )
    responsive = bool(
        obs.get("controls_responsive")
        or obs.get("controllable")
        or obs.get("can_control")
        or obs.get("input_responsive")
    )
    return alive and responsive and not _is_loading(obs)


def detect_events(
    prev: Mapping[str, Any],
    action: str,
    nxt: Mapping[str, Any],
) -> list[Event]:
    """Detect confirmed events from a prev → action → next transition."""
    events: list[Event] = []
    a = (action or "").lower()

    # Target acquired / validated
    if not bool(prev.get("has_target")) and bool(nxt.get("has_target")):
        events.append(
            TargetAcquired(
                type=EventType.TARGET_ACQUIRED,
                confidence=0.85,
                evidence=["has_target_false_to_true"],
            )
        )
    if bool(nxt.get("target_validated") or nxt.get("valid_target")) and not bool(
        prev.get("target_validated") or prev.get("valid_target")
    ):
        events.append(
            TargetValidated(
                type=EventType.TARGET_VALIDATED,
                confidence=0.8,
                evidence=["valid_target_edge"],
            )
        )

    # Damage dealt / received
    prev_thp = _hp(prev, "target_hp_est")
    next_thp = _hp(nxt, "target_hp_est")
    if (
        prev_thp is not None
        and next_thp is not None
        and prev_thp > 0
        and next_thp < prev_thp - 0.02
        and bool(prev.get("has_target"))
    ):
        events.append(
            DamageDealt(
                type=EventType.DAMAGE_DEALT,
                confidence=0.75,
                evidence=["target_hp_drop"],
                payload={"delta": prev_thp - next_thp},
            )
        )

    prev_hp = _hp(prev)
    next_hp = _hp(nxt)
    if prev_hp is not None and next_hp is not None and next_hp < prev_hp - 0.02:
        events.append(
            DamageReceived(
                type=EventType.DAMAGE_RECEIVED,
                confidence=0.7,
                evidence=["player_hp_drop"],
                payload={"delta": prev_hp - next_hp},
            )
        )

    # Kill — a combat transition alone is never confirmation. At least one
    # orthogonal class (HP zero, loot/XP, objective count, or explicit flag)
    # must corroborate the outcome.
    kill_ev = _collect_kill_evidence(prev, nxt)
    kill_classes = kill_evidence_classes(kill_ev)
    contextual = {
        item for item in kill_ev if item in {"target_lost_after_combat", "combat_ended"}
    }
    explicitly_confirmed = "explicit_flag" in kill_classes
    kill_confirmed = bool(kill_classes) and (
        explicitly_confirmed or bool(contextual) or len(kill_classes) >= 2
    )
    if kill_confirmed:
        conf = min(1.0, 0.55 + 0.15 * len(kill_classes) + 0.05 * len(contextual))
        events.append(
            KillConfirmed(
                type=EventType.KILL_CONFIRMED,
                confidence=conf,
                evidence=list(kill_ev),
                metadata={"evidence_classes": sorted(kill_classes)},
            )
        )
    elif contextual:
        if "target_lost_after_combat" in contextual:
            events.append(
                TargetLost(
                    type=EventType.TARGET_LOST,
                    confidence=0.65,
                    evidence=["target_lost_after_combat"],
                )
            )
        if len(contextual) >= 2:
            events.append(
                SuspectedKill(
                    type=EventType.SUSPECTED_KILL,
                    confidence=0.4,
                    evidence=sorted(contextual),
                    metadata={"rewardable_as_kill": False},
                )
            )

    # Objectives
    try:
        pg0 = float(prev.get("goal_progress") if prev.get("goal_progress") is not None else -1)
        pg1 = float(nxt.get("goal_progress") if nxt.get("goal_progress") is not None else -1)
        if pg1 > pg0 >= 0:
            events.append(
                ObjectiveProgressed(
                    type=EventType.OBJECTIVE_PROGRESSED,
                    confidence=0.9,
                    evidence=["goal_progress_increased"],
                    payload={"from": pg0, "to": pg1},
                )
            )
    except (TypeError, ValueError):
        pass
    prev_complete = bool(
        prev.get("goal_complete") or prev.get("objective_completed") or prev.get("quest_complete")
    )
    next_complete = bool(
        nxt.get("goal_complete") or nxt.get("objective_completed") or nxt.get("quest_complete")
    )
    if next_complete and not prev_complete:
        events.append(
            ObjectiveCompleted(
                type=EventType.OBJECTIVE_COMPLETED,
                confidence=0.95,
                evidence=["objective_completed_flag"],
            )
        )

    # Loot
    if nxt.get("loot_confirmed") or (
        a == "loot" and bool(prev.get("has_target")) and not bool(nxt.get("has_target"))
        and "loot" in _ocr(nxt)
    ):
        events.append(
            LootConfirmed(
                type=EventType.LOOT_CONFIRMED,
                confidence=0.7,
                evidence=["loot_signal"],
            )
        )

    # Death / resurrection
    prev_dead = bool(prev.get("is_dead")) or str(prev.get("life_phase") or "") in {
        "dead_dialog",
        "confirm",
        "rez_picker",
    }
    next_dead = bool(nxt.get("is_dead")) or str(nxt.get("life_phase") or "") in {
        "dead_dialog",
        "confirm",
        "rez_picker",
    }
    if next_dead and not prev_dead:
        events.append(
            DeathConfirmed(
                type=EventType.DEATH_CONFIRMED,
                confidence=0.95,
                evidence=["life_to_dead_transition"],
            )
        )
    prev_ghost = bool(prev.get("is_ghost")) or str(prev.get("life_phase") or "") == "ghost"
    next_alive = (
        not bool(nxt.get("is_dead"))
        and not bool(nxt.get("is_ghost"))
        and str(nxt.get("life_phase") or "alive") == "alive"
    )
    if (prev_dead or prev_ghost) and next_alive:
        events.append(
            ResurrectionConfirmed(
                type=EventType.RESURRECTION_CONFIRMED,
                confidence=0.9,
                evidence=["returned_to_alive"],
            )
        )

    prev_loading = _is_loading(prev)
    next_loading = _is_loading(nxt)
    if next_loading and not prev_loading:
        events.append(
            LoadingStarted(
                type=EventType.LOADING_STARTED,
                confidence=0.95,
                evidence=["loading_false_to_true"],
            )
        )
    if prev_loading and not next_loading:
        events.append(
            LoadingEnded(
                type=EventType.LOADING_ENDED,
                confidence=0.95,
                evidence=["loading_true_to_false"],
            )
        )

    if _is_controllable(nxt) and not _is_controllable(prev):
        events.append(
            BecameControllable(
                type=EventType.BECAME_CONTROLLABLE,
                confidence=0.9,
                evidence=["controls_responsive_edge"],
            )
        )

    # Skill outcomes (caller may set flags)
    if nxt.get("skill_succeeded"):
        events.append(
            SkillSuccess(
                type=EventType.SKILL_SUCCESS,
                confidence=float(nxt.get("skill_success_confidence") or 0.8),
                evidence=["skill_succeeded_flag"],
                payload={"skill": nxt.get("skill_name")},
            )
        )
    if nxt.get("skill_failed"):
        reason = str(nxt.get("skill_fail_reason") or "failed")
        events.append(
            SkillFailure(
                type=EventType.SKILL_FAILURE,
                confidence=0.85,
                evidence=[reason],
                payload={"skill": nxt.get("skill_name"), "reason": reason},
            )
        )
    if nxt.get("skill_timeout"):
        events.append(
            SkillFailure(
                type=EventType.SKILL_FAILURE,
                confidence=0.9,
                evidence=["timeout"],
                payload={"skill": nxt.get("skill_name"), "reason": "timeout"},
            )
        )

    # Modal cleared
    if bool(prev.get("modal_menu") or prev.get("blocking_modal")) and not bool(
        nxt.get("modal_menu") or nxt.get("blocking_modal")
    ):
        events.append(
            ModalCleared(
                type=EventType.MODAL_CLEARED,
                confidence=0.85,
                evidence=["modal_false_edge"],
            )
        )

    # Movement blocked
    if (
        (a.startswith("move_") or a.startswith("hold:"))
        and float(nxt.get("motion") or 0) < 1.0
        and float(prev.get("motion") or 0) < 1.0
    ) or bool(nxt.get("movement_blocked") or nxt.get("stuck")):
        if a.startswith("move_") or a.startswith("hold:") or nxt.get("movement_blocked"):
            events.append(
                MovementBlocked(
                    type=EventType.MOVEMENT_BLOCKED,
                    confidence=0.65,
                    evidence=["low_motion_while_moving"],
                )
            )

    # Sensor conflict
    if nxt.get("sensor_conflict") or (
        bool(nxt.get("is_dead")) and float(nxt.get("vision_player_hp") or 0) > 0.5
    ):
        events.append(
            SensorConflict(
                type=EventType.SENSOR_CONFLICT,
                confidence=0.6,
                evidence=["conflicting_sensors"],
            )
        )

    return events
