"""Versioned data contract for the PlayMind Planner V2 boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

PLANNER_STATE_SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1

ALLOWED_REPLAN_EVENTS: tuple[str, ...] = (
    "new_objective",
    "plan_complete",
    "skill_fail",
    "health_critical",
    "death",
    "ghost",
    "controllable_after_recovery",
    "severe_stuck",
    "target_invalid",
    "objective_progress_change",
    "plan_expiry",
    "modal",
    "periodic_interval",
)


def sensor_payload(value: Any, confidence: Any = None) -> dict[str, Any]:
    """Return a JSON sensor triple without conflating unknown and false."""
    known = value is not None
    conf: float | None
    try:
        conf = None if confidence is None else float(confidence)
    except (TypeError, ValueError):
        conf = None
    if conf is not None:
        conf = max(0.0, min(1.0, conf))
    return {"value": value, "known": known, "confidence": conf}


def _normalise_sensor(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        value = raw.get("value")
        known = bool(raw["known"]) if "known" in raw else value is not None
        # A false reading is known false. Only an explicitly unknown reading
        # has its value discarded.
        if not known:
            value = None
        payload = sensor_payload(value, raw.get("confidence"))
        payload["known"] = known
        return payload
    return sensor_payload(raw)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class PlannerState:
    """Compact state sent to a planner model.

    Sensor entries use ``{"value", "known", "confidence"}``. This makes an
    absent reading different from a confident ``False`` or zero reading.
    """

    schema_version: int = PLANNER_STATE_SCHEMA_VERSION
    game_id: str = ""
    timestamp: float = 0.0
    goal: str = ""
    profile: dict[str, Any] = field(default_factory=dict)
    available_skills: list[str] = field(default_factory=list)
    current_skill: str | None = None
    recent_skills: list[Any] = field(default_factory=list)
    previous_plan: dict[str, Any] | None = None
    memory: Any = field(default_factory=list)
    sensors: dict[str, dict[str, Any]] = field(default_factory=dict)
    life_phase: str = "unknown"
    loading: dict[str, Any] = field(default_factory=lambda: sensor_payload(None))
    modal: dict[str, Any] = field(default_factory=lambda: sensor_payload(None))
    stuck: dict[str, Any] = field(default_factory=lambda: sensor_payload(None))
    objective_progress: dict[str, Any] = field(
        default_factory=lambda: sensor_payload(None)
    )
    objective_text: str | None = None
    ocr_text: str = ""
    sensor_warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sensors = {
            str(name): _normalise_sensor(payload)
            for name, payload in self.sensors.items()
        }
        self.loading = _normalise_sensor(self.loading)
        self.modal = _normalise_sensor(self.modal)
        self.stuck = _normalise_sensor(self.stuck)
        self.objective_progress = _normalise_sensor(self.objective_progress)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "game_id": self.game_id,
            "timestamp": float(self.timestamp),
            "goal": self.goal,
            "profile": _json_safe(self.profile),
            "available_skills": list(self.available_skills),
            "current_skill": self.current_skill,
            "recent_skills": _json_safe(self.recent_skills),
            "previous_plan": _json_safe(self.previous_plan),
            "memory": _json_safe(self.memory),
            "sensors": _json_safe(self.sensors),
            "life_phase": self.life_phase,
            "loading": dict(self.loading),
            "modal": dict(self.modal),
            "stuck": dict(self.stuck),
            "objective_progress": dict(self.objective_progress),
            "objective_text": self.objective_text,
            "ocr_text": self.ocr_text,
            "sensor_warnings": list(self.sensor_warnings),
        }

    def to_json(self, **kwargs: Any) -> str:
        options = {"sort_keys": True, "separators": (",", ":")}
        options.update(kwargs)
        return json.dumps(self.to_dict(), **options)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PlannerState":
        version = int(raw.get("schema_version", PLANNER_STATE_SCHEMA_VERSION))
        if version != PLANNER_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported planner state schema_version={version}; "
                f"expected {PLANNER_STATE_SCHEMA_VERSION}"
            )
        previous = raw.get("previous_plan")
        if isinstance(previous, Plan):
            previous = previous.to_dict()
        return cls(
            schema_version=version,
            game_id=str(raw.get("game_id") or ""),
            timestamp=float(raw.get("timestamp") or 0.0),
            goal=str(raw.get("goal") or ""),
            profile=dict(raw.get("profile") or {}),
            available_skills=[str(x) for x in raw.get("available_skills") or []],
            current_skill=(
                str(raw["current_skill"])
                if raw.get("current_skill") is not None
                else None
            ),
            recent_skills=list(raw.get("recent_skills") or []),
            previous_plan=dict(previous) if isinstance(previous, Mapping) else None,
            memory=raw.get("memory", []),
            sensors={
                str(k): _normalise_sensor(v)
                for k, v in dict(raw.get("sensors") or {}).items()
            },
            life_phase=str(raw.get("life_phase") or "unknown"),
            loading=_normalise_sensor(raw.get("loading")),
            modal=_normalise_sensor(raw.get("modal")),
            stuck=_normalise_sensor(raw.get("stuck")),
            objective_progress=_normalise_sensor(raw.get("objective_progress")),
            objective_text=(
                str(raw["objective_text"])
                if raw.get("objective_text") is not None
                else None
            ),
            ocr_text=str(raw.get("ocr_text") or ""),
            sensor_warnings=[str(x) for x in raw.get("sensor_warnings") or []],
        )

    @classmethod
    def from_json(cls, text: str) -> "PlannerState":
        raw = json.loads(text)
        if not isinstance(raw, Mapping):
            raise ValueError("planner state JSON must be an object")
        return cls.from_dict(raw)


@dataclass(frozen=True)
class SkillStep:
    name: str
    until: str | None = None
    max_seconds: int = 30
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "until": self.until,
            "max_seconds": int(self.max_seconds),
            "constraints": _json_safe(self.constraints),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SkillStep":
        until = raw.get("until")
        return cls(
            name=str(raw.get("name") or ""),
            until=str(until) if until is not None else None,
            max_seconds=int(raw.get("max_seconds", 30)),
            constraints=dict(raw.get("constraints") or {}),
        )


@dataclass(frozen=True)
class Plan:
    schema_version: int = PLAN_SCHEMA_VERSION
    goal: str = ""
    skills: list[SkillStep] = field(default_factory=list)
    replan_on: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason_code: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "goal": self.goal,
            "skills": [step.to_dict() for step in self.skills],
            "replan_on": list(self.replan_on),
            "confidence": float(self.confidence),
            "reason_code": self.reason_code,
            "summary": self.summary,
        }

    def to_json(self, **kwargs: Any) -> str:
        options = {"sort_keys": True, "separators": (",", ":")}
        options.update(kwargs)
        return json.dumps(self.to_dict(), **options)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Plan":
        skills_raw = raw.get("skills")
        if not isinstance(skills_raw, Sequence) or isinstance(
            skills_raw, (str, bytes)
        ):
            raise ValueError("plan.skills must be an array")
        steps: list[SkillStep] = []
        for item in skills_raw:
            if not isinstance(item, Mapping):
                raise ValueError("each plan skill must be an object")
            steps.append(SkillStep.from_dict(item))
        replan_raw = raw.get("replan_on", [])
        if not isinstance(replan_raw, Sequence) or isinstance(
            replan_raw, (str, bytes)
        ):
            raise ValueError("plan.replan_on must be an array")
        return cls(
            schema_version=int(raw.get("schema_version", PLAN_SCHEMA_VERSION)),
            goal=str(raw.get("goal") or ""),
            skills=steps,
            replan_on=[str(x) for x in replan_raw],
            confidence=float(raw.get("confidence", 0.0)),
            reason_code=str(raw.get("reason_code") or ""),
            summary=str(raw.get("summary") or ""),
        )

    @classmethod
    def from_json(cls, text: str) -> "Plan":
        raw = json.loads(text)
        if not isinstance(raw, Mapping):
            raise ValueError("plan JSON must be an object")
        return cls.from_dict(raw)


@dataclass
class PlanValidationResult:
    ok: bool
    plan: Plan | None = None
    errors: list[str] = field(default_factory=list)
    repaired: bool = False


def serialize_planner_state(state: PlannerState) -> str:
    return state.to_json()


def deserialize_planner_state(text: str) -> PlannerState:
    return PlannerState.from_json(text)


def serialize_plan(plan: Plan) -> str:
    return plan.to_json()


def deserialize_plan(text: str) -> Plan:
    return Plan.from_json(text)


# Concise aliases for callers that treat the contract as a codec module.
state_to_json = serialize_planner_state
state_from_json = deserialize_planner_state
plan_to_json = serialize_plan
plan_from_json = deserialize_plan


__all__ = [
    "ALLOWED_REPLAN_EVENTS",
    "PLAN_SCHEMA_VERSION",
    "PLANNER_STATE_SCHEMA_VERSION",
    "Plan",
    "PlannerState",
    "PlanValidationResult",
    "SkillStep",
    "deserialize_plan",
    "deserialize_planner_state",
    "plan_from_json",
    "plan_to_json",
    "sensor_payload",
    "serialize_plan",
    "serialize_planner_state",
    "state_from_json",
    "state_to_json",
]
