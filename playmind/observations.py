"""Typed observation model for Learning Architecture V2.

Legacy owned-loop code still passes mutable dicts. Observation wraps those
with explicit Optional/confidence semantics so missing sensors stay unknown
instead of silently becoming False / 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class SensorValue(Generic[T]):
    """Optional reading paired with an optional confidence in [0, 1]."""

    value: Optional[T] = None
    confidence: Optional[float] = None

    @property
    def known(self) -> bool:
        return self.value is not None


def _copy_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, (tuple, set)):
        return [str(x) for x in value]
    return [str(value)]


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


@dataclass
class Observation:
    """Frame-level world state with sensor confidence and migration passthrough."""

    timestamp: float = 0.0
    frame_id: int = 0

    player_hp: Optional[float] = None
    player_hp_confidence: Optional[float] = None

    target_hp: Optional[float] = None
    target_hp_confidence: Optional[float] = None
    has_target: Optional[bool] = None
    has_target_confidence: Optional[float] = None

    in_combat: Optional[bool] = None
    in_combat_confidence: Optional[float] = None

    is_dead: Optional[bool] = None
    is_ghost: Optional[bool] = None
    life_phase: str = "unknown"

    motion: Optional[float] = None
    motion_confidence: Optional[float] = None

    hostiles_near: Optional[bool] = None
    hostile_count: Optional[int] = None
    hostile_count_confidence: Optional[float] = None

    blocking_modal: Optional[bool] = None

    objective_text: Optional[str] = None
    objective_progress: Optional[float] = None

    ocr_text: str = ""
    ui_detections: list[str] = field(default_factory=list)

    known_abilities: list[str] = field(default_factory=list)

    stagnation_count: int = 0
    failed_action_streak: int = 0

    recent_action: Optional[str] = None
    recent_action_outcome: Optional[str] = None

    goal_kind: Optional[str] = None
    goal_summary: Optional[str] = None

    sensor_warnings: list[str] = field(default_factory=list)
    progress_stage: Optional[str] = None

    # Raw dict passthrough for migration / owned_loop compatibility.
    legacy: dict[str, Any] = field(default_factory=dict)

    def sensor(self, name: str) -> SensorValue[Any]:
        """Convenience accessor for common confidence-paired sensors."""
        mapping: dict[str, tuple[Optional[Any], Optional[float]]] = {
            "player_hp": (self.player_hp, self.player_hp_confidence),
            "target_hp": (self.target_hp, self.target_hp_confidence),
            "has_target": (self.has_target, self.has_target_confidence),
            "in_combat": (self.in_combat, self.in_combat_confidence),
            "motion": (self.motion, self.motion_confidence),
            "hostile_count": (self.hostile_count, self.hostile_count_confidence),
        }
        if name not in mapping:
            raise KeyError(name)
        value, conf = mapping[name]
        return SensorValue(value=value, confidence=conf)

    @classmethod
    def from_legacy_dict(cls, obs: dict[str, Any]) -> Observation:
        """Build Observation from an owned-loop dict.

        Missing sensors stay None/unknown — never invent confident False.
        """
        raw = dict(obs) if obs is not None else {}
        warnings: list[str] = []

        # --- player HP ---
        player_hp: Optional[float] = None
        player_hp_confidence: Optional[float] = None
        if "vision_player_hp" in raw and raw["vision_player_hp"] is not None:
            player_hp = _as_optional_float(raw["vision_player_hp"])
            player_hp_confidence = _as_optional_float(raw.get("player_hp_confidence"))
        elif isinstance(raw.get("player"), dict) and "hp" in raw["player"]:
            player_hp = _as_optional_float(raw["player"].get("hp"))
            player_hp_confidence = _as_optional_float(raw.get("player_hp_confidence"))
            warnings.append("player_hp: derived from player.hp (no vision_player_hp)")
        else:
            warnings.append("player_hp: unknown (missing)")

        # --- target HP ---
        target_hp: Optional[float] = None
        target_hp_confidence: Optional[float] = None
        if "target_hp" in raw:
            target_hp = _as_optional_float(raw.get("target_hp"))
            target_hp_confidence = _as_optional_float(raw.get("target_hp_confidence"))
        elif "target_hp_est" in raw:
            target_hp = _as_optional_float(raw.get("target_hp_est"))
            target_hp_confidence = _as_optional_float(raw.get("target_hp_confidence"))
        else:
            warnings.append("target_hp: unknown (missing)")

        # --- boolean sensors: absent key => None, never invent False ---
        has_target: Optional[bool]
        has_target_confidence: Optional[float]
        if "has_target" in raw:
            has_target = _as_optional_bool(raw.get("has_target"))
            has_target_confidence = _as_optional_float(raw.get("has_target_confidence"))
        else:
            has_target = None
            has_target_confidence = None
            warnings.append("has_target: unknown (missing)")

        in_combat: Optional[bool]
        in_combat_confidence: Optional[float]
        if "in_combat" in raw:
            in_combat = _as_optional_bool(raw.get("in_combat"))
            in_combat_confidence = _as_optional_float(raw.get("in_combat_confidence"))
        else:
            in_combat = None
            in_combat_confidence = None
            warnings.append("in_combat: unknown (missing)")

        is_dead: Optional[bool]
        if "is_dead" in raw:
            is_dead = _as_optional_bool(raw.get("is_dead"))
        else:
            is_dead = None
            warnings.append("is_dead: unknown (missing)")

        is_ghost: Optional[bool]
        if "is_ghost" in raw:
            is_ghost = _as_optional_bool(raw.get("is_ghost"))
        else:
            is_ghost = None
            warnings.append("is_ghost: unknown (missing)")

        if "life_phase" in raw and raw.get("life_phase") not in (None, ""):
            life_phase = str(raw.get("life_phase"))
        else:
            life_phase = "unknown"
            warnings.append("life_phase: unknown (missing)")

        motion: Optional[float]
        motion_confidence: Optional[float]
        if "motion" in raw:
            motion = _as_optional_float(raw.get("motion"))
            motion_confidence = _as_optional_float(raw.get("motion_confidence"))
        else:
            motion = None
            motion_confidence = None
            warnings.append("motion: unknown (missing)")

        hostiles_near: Optional[bool]
        if "hostiles_near" in raw:
            hostiles_near = _as_optional_bool(raw.get("hostiles_near"))
        else:
            hostiles_near = None
            warnings.append("hostiles_near: unknown (missing)")

        hostile_count: Optional[int]
        hostile_count_confidence: Optional[float]
        if "hostile_count" in raw:
            hostile_count = _as_optional_int(raw.get("hostile_count"))
            hostile_count_confidence = _as_optional_float(
                raw.get("hostile_count_confidence")
            )
        elif hostiles_near is True:
            hostile_count = 1
            hostile_count_confidence = None
            warnings.append("hostile_count: inferred as 1 from hostiles_near")
        elif hostiles_near is False:
            hostile_count = 0
            hostile_count_confidence = None
        else:
            hostile_count = None
            hostile_count_confidence = None
            if "hostile_count" not in raw:
                warnings.append("hostile_count: unknown (missing)")

        blocking_modal: Optional[bool]
        if "blocking_modal" in raw:
            blocking_modal = _as_optional_bool(raw.get("blocking_modal"))
        elif "modal_menu" in raw:
            blocking_modal = _as_optional_bool(raw.get("modal_menu"))
        else:
            blocking_modal = None
            warnings.append("blocking_modal: unknown (missing)")

        objective_text: Optional[str] = None
        if "objective_text" in raw and raw.get("objective_text") is not None:
            objective_text = str(raw.get("objective_text"))
        elif "quest_text" in raw and raw.get("quest_text") is not None:
            objective_text = str(raw.get("quest_text"))

        objective_progress = _as_optional_float(raw.get("objective_progress"))

        ocr_text = ""
        if "ocr_text" in raw and raw.get("ocr_text") is not None:
            ocr_text = str(raw.get("ocr_text"))
        elif raw.get("screen_ocr") is not None:
            ocr_text = str(raw.get("screen_ocr"))

        ui_detections = _copy_list(
            raw.get("ui_detections")
            if "ui_detections" in raw
            else raw.get("ui_hits")
        )
        known_abilities = _copy_list(raw.get("known_abilities") or raw.get("abilities"))

        stagnation_count = int(
            raw.get("stagnation_count")
            if raw.get("stagnation_count") is not None
            else raw.get("stagnant") or 0
        )
        failed_action_streak = int(raw.get("failed_action_streak") or 0)

        recent_action = (
            str(raw["recent_action"]) if raw.get("recent_action") is not None else None
        )
        recent_action_outcome = (
            str(raw["recent_action_outcome"])
            if raw.get("recent_action_outcome") is not None
            else None
        )

        goal_kind = str(raw["goal_kind"]) if raw.get("goal_kind") is not None else None
        goal_summary = (
            str(raw["goal_summary"]) if raw.get("goal_summary") is not None else None
        )

        progress_stage = (
            str(raw["progress_stage"]) if raw.get("progress_stage") is not None else None
        )

        existing_warnings = _copy_list(raw.get("sensor_warnings"))
        for w in existing_warnings:
            if w not in warnings:
                warnings.append(w)

        timestamp = float(raw.get("timestamp") or 0.0)
        frame_id = int(raw.get("frame_id") or raw.get("steps") or 0)

        return cls(
            timestamp=timestamp,
            frame_id=frame_id,
            player_hp=player_hp,
            player_hp_confidence=player_hp_confidence,
            target_hp=target_hp,
            target_hp_confidence=target_hp_confidence,
            has_target=has_target,
            has_target_confidence=has_target_confidence,
            in_combat=in_combat,
            in_combat_confidence=in_combat_confidence,
            is_dead=is_dead,
            is_ghost=is_ghost,
            life_phase=life_phase,
            motion=motion,
            motion_confidence=motion_confidence,
            hostiles_near=hostiles_near,
            hostile_count=hostile_count,
            hostile_count_confidence=hostile_count_confidence,
            blocking_modal=blocking_modal,
            objective_text=objective_text,
            objective_progress=objective_progress,
            ocr_text=ocr_text,
            ui_detections=ui_detections,
            known_abilities=known_abilities,
            stagnation_count=stagnation_count,
            failed_action_streak=failed_action_streak,
            recent_action=recent_action,
            recent_action_outcome=recent_action_outcome,
            goal_kind=goal_kind,
            goal_summary=goal_summary,
            sensor_warnings=warnings,
            progress_stage=progress_stage,
            legacy=raw,
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        """Export a dict compatible with existing owned_loop helpers."""
        out: dict[str, Any] = dict(self.legacy)

        out["timestamp"] = self.timestamp
        out["frame_id"] = self.frame_id
        out["steps"] = self.frame_id

        if self.player_hp is not None:
            out["vision_player_hp"] = self.player_hp
            player = dict(out.get("player") or {"x": 0, "y": 0, "hp": self.player_hp})
            player["hp"] = self.player_hp
            out["player"] = player

        if self.target_hp is not None:
            out["target_hp_est"] = self.target_hp
            out["target_hp"] = self.target_hp

        # Only write bools when known — never invent False for None.
        if self.has_target is not None:
            out["has_target"] = self.has_target
        if self.in_combat is not None:
            out["in_combat"] = self.in_combat
        if self.is_dead is not None:
            out["is_dead"] = self.is_dead
        if self.is_ghost is not None:
            out["is_ghost"] = self.is_ghost
        if self.life_phase != "unknown":
            out["life_phase"] = self.life_phase
        elif "life_phase" not in out:
            out["life_phase"] = self.life_phase

        if self.motion is not None:
            out["motion"] = self.motion
        if self.hostiles_near is not None:
            out["hostiles_near"] = self.hostiles_near
        if self.hostile_count is not None:
            out["hostile_count"] = self.hostile_count

        if self.blocking_modal is not None:
            out["blocking_modal"] = self.blocking_modal
            out["modal_menu"] = self.blocking_modal

        if self.objective_text is not None:
            out["objective_text"] = self.objective_text
            out["quest_text"] = self.objective_text
        if self.objective_progress is not None:
            out["objective_progress"] = self.objective_progress

        out["ocr_text"] = self.ocr_text
        out["screen_ocr"] = self.ocr_text
        out["ui_detections"] = list(self.ui_detections)
        out["ui_hits"] = list(self.ui_detections)
        out["known_abilities"] = list(self.known_abilities)

        out["stagnation_count"] = self.stagnation_count
        out["stagnant"] = self.stagnation_count
        out["failed_action_streak"] = self.failed_action_streak

        if self.recent_action is not None:
            out["recent_action"] = self.recent_action
        if self.recent_action_outcome is not None:
            out["recent_action_outcome"] = self.recent_action_outcome
        if self.goal_kind is not None:
            out["goal_kind"] = self.goal_kind
        if self.goal_summary is not None:
            out["goal_summary"] = self.goal_summary
        if self.progress_stage is not None:
            out["progress_stage"] = self.progress_stage

        out["sensor_warnings"] = list(self.sensor_warnings)

        if self.player_hp_confidence is not None:
            out["player_hp_confidence"] = self.player_hp_confidence
        if self.target_hp_confidence is not None:
            out["target_hp_confidence"] = self.target_hp_confidence
        if self.has_target_confidence is not None:
            out["has_target_confidence"] = self.has_target_confidence
        if self.in_combat_confidence is not None:
            out["in_combat_confidence"] = self.in_combat_confidence
        if self.motion_confidence is not None:
            out["motion_confidence"] = self.motion_confidence
        if self.hostile_count_confidence is not None:
            out["hostile_count_confidence"] = self.hostile_count_confidence

        return out
