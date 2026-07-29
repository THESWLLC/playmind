"""Evidence-based event detection for Learning Architecture V2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class EventType(str, Enum):
    TARGET_ACQUIRED = "TargetAcquired"
    TARGET_VALIDATED = "TargetValidated"
    DAMAGE_DEALT = "DamageDealt"
    DAMAGE_RECEIVED = "DamageReceived"
    KILL_CONFIRMED = "KillConfirmed"
    OBJECTIVE_PROGRESSED = "ObjectiveProgressed"
    OBJECTIVE_COMPLETED = "ObjectiveCompleted"
    LOOT_CONFIRMED = "LootConfirmed"
    DEATH_CONFIRMED = "DeathConfirmed"
    RESURRECTION_CONFIRMED = "ResurrectionConfirmed"
    SKILL_SUCCEEDED = "SkillSucceeded"
    SKILL_FAILED = "SkillFailed"
    MODAL_CLEARED = "ModalCleared"
    MOVEMENT_BLOCKED = "MovementBlocked"
    SENSOR_CONFLICT = "SensorConflict"


@dataclass
class Event:
    """Base confirmed/candidate event with confidence in [0, 1]."""

    type: EventType
    confidence: float = 1.0
    evidence: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.type.value


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
class SkillSucceeded(Event):
    def __post_init__(self) -> None:
        self.type = EventType.SKILL_SUCCEEDED


@dataclass
class SkillFailed(Event):
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


_KILL_MIN_EVIDENCE = 2


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

    # Kill — multi-evidence only
    kill_ev = _collect_kill_evidence(prev, nxt)
    if len(kill_ev) >= _KILL_MIN_EVIDENCE:
        conf = min(1.0, 0.45 + 0.2 * len(kill_ev))
        events.append(
            KillConfirmed(
                type=EventType.KILL_CONFIRMED,
                confidence=conf,
                evidence=list(kill_ev),
            )
        )
    # Explicitly do NOT emit KillConfirmed on mere target loss (single evidence).

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
    if nxt.get("goal_complete") or nxt.get("objective_completed"):
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

    # Skill outcomes (caller may set flags)
    if nxt.get("skill_succeeded"):
        events.append(
            SkillSucceeded(
                type=EventType.SKILL_SUCCEEDED,
                confidence=float(nxt.get("skill_success_confidence") or 0.8),
                evidence=["skill_succeeded_flag"],
                payload={"skill": nxt.get("skill_name")},
            )
        )
    if nxt.get("skill_failed"):
        reason = str(nxt.get("skill_fail_reason") or "failed")
        events.append(
            SkillFailed(
                type=EventType.SKILL_FAILED,
                confidence=0.85,
                evidence=[reason],
                payload={"skill": nxt.get("skill_name"), "reason": reason},
            )
        )
    if nxt.get("skill_timeout"):
        events.append(
            SkillFailed(
                type=EventType.SKILL_FAILED,
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
