"""Canonical schemas for planner SFT, preference, and evaluation data."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

PLANNER_DATA_SCHEMA_VERSION = 1
PLANNER_SYSTEM_PROMPT = (
    "You are PlayMind's high-level MMO planner. Given planner_state, return "
    "only a valid JSON plan containing a skills list and concise rationale. "
    "Respect lifecycle, uncertainty, and safety constraints."
)

_SENSORS: dict[str, tuple[str, ...]] = {
    "player_hp": ("player_hp", "vision_player_hp"),
    "target_hp": ("target_hp", "target_hp_est"),
    "has_target": ("has_target",),
    "in_combat": ("in_combat",),
    "is_dead": ("is_dead",),
    "is_ghost": ("is_ghost",),
    "motion": ("motion",),
    "hostile_count": ("hostile_count",),
    "blocking_modal": ("blocking_modal", "modal_menu"),
    "objective_progress": ("objective_progress",),
}


def _sensor_value(
    observation: Mapping[str, Any],
    aliases: Sequence[str],
    confidence_blob: Mapping[str, Any],
) -> dict[str, Any]:
    present = next((name for name in aliases if name in observation), None)
    value = observation.get(present) if present is not None else None
    confidence = None
    for name in aliases:
        confidence_key = f"{name}_confidence"
        if confidence_key in confidence_blob:
            confidence = confidence_blob[confidence_key]
            break
        if name in confidence_blob:
            candidate = confidence_blob[name]
            confidence = (
                candidate.get("confidence")
                if isinstance(candidate, Mapping)
                else candidate
            )
            break
        if confidence_key in observation:
            confidence = observation[confidence_key]
            break
    return {
        "value": value,
        # Explicit knownness keeps absent sensors distinct from known false/zero.
        "known": present is not None and value is not None,
        "confidence": confidence,
    }


def build_planner_state(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build an uncertainty-preserving planner input from a demo/planner row."""
    provided = record.get("planner_state")
    if hasattr(provided, "to_dict") and callable(provided.to_dict):
        provided = provided.to_dict()
    if isinstance(provided, Mapping):
        state = dict(provided)
        if "unknown_sensors" not in state:
            sensors = state.get("sensors")
            state["unknown_sensors"] = sorted(
                str(name)
                for name, payload in (
                    sensors.items() if isinstance(sensors, Mapping) else ()
                )
                if not isinstance(payload, Mapping)
                or payload.get("known", payload.get("value") is not None) is False
            )
        return state

    observation = record.get("observation")
    raw = dict(observation) if isinstance(observation, Mapping) else {}
    confidence = record.get("sensor_confidence")
    confidence_blob = dict(confidence) if isinstance(confidence, Mapping) else {}
    sensors = {
        name: _sensor_value(raw, aliases, confidence_blob)
        for name, aliases in _SENSORS.items()
    }
    unknown = sorted(name for name, value in sensors.items() if not value["known"])
    lifecycle = (
        record.get("lifecycle_state")
        or raw.get("life_phase")
        or "unknown"
    )
    return {
        "episode_id": str(record.get("episode_id") or "unknown"),
        "timestamp": record.get("timestamp"),
        "goal": record.get("goal") or raw.get("goal_summary") or raw.get("goal_kind"),
        "lifecycle_state": lifecycle,
        "sensors": sensors,
        "unknown_sensors": unknown,
        "sensor_warnings": list(raw.get("sensor_warnings") or []),
        "recent_action": raw.get("recent_action"),
        "recent_action_outcome": raw.get("recent_action_outcome"),
        "objective_text": raw.get("objective_text") or raw.get("quest_text"),
        "known_abilities": list(raw.get("known_abilities") or raw.get("abilities") or []),
    }


def normalize_plan(value: Any, record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if isinstance(value, Mapping):
        plan = dict(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        plan = {"skills": [str(item) for item in value]}
    elif value not in (None, ""):
        plan = {"skills": [str(value)]}
    else:
        source = record or {}
        skill = source.get("inferred_skill") or source.get("skill")
        plan = {"skills": [str(skill)]} if skill else {"skills": []}
    if "skills" not in plan:
        skill = plan.pop("skill", None)
        plan["skills"] = [skill] if skill else []
    plan["skills"] = [
        dict(skill) if isinstance(skill, Mapping) else str(skill)
        for skill in plan.get("skills") or []
    ]
    plan.setdefault("rationale", "")
    return plan


def planner_messages(
    planner_state: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    system_prompt: str = PLANNER_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                {"planner_state": dict(planner_state)},
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(dict(plan), sort_keys=True, separators=(",", ":")),
        },
    ]


@dataclass
class SFTExample:
    example_id: str
    episode_id: str
    messages: list[dict[str, str]]
    split: str = ""
    eligible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = PLANNER_DATA_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreferenceExample:
    example_id: str
    episode_id: str
    planner_state: dict[str, Any]
    chosen: dict[str, Any]
    rejected: dict[str, Any]
    outcomes: dict[str, Any] = field(default_factory=dict)
    split: str = ""
    eligible: bool = True
    schema_version: int = PLANNER_DATA_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalScenario:
    scenario_id: str
    category: str
    planner_state: dict[str, Any]
    expected_plan: dict[str, Any]
    invalid_plan: dict[str, Any] | None = None
    schema_version: int = PLANNER_DATA_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "EvalScenario",
    "PLANNER_DATA_SCHEMA_VERSION",
    "PLANNER_SYSTEM_PROMPT",
    "PreferenceExample",
    "SFTExample",
    "build_planner_state",
    "normalize_plan",
    "planner_messages",
]
