"""Build compact, uncertainty-aware state for Planner V2."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from playmind.models.feature_schema import TRIPLE_SENSORS, extract_sensor_triples
from playmind.observations import Observation

from .contract import Plan, PlannerState, sensor_payload

_BOOL_SENSORS = frozenset(
    {
        "has_target",
        "in_combat",
        "is_dead",
        "is_ghost",
        "hostiles_near",
        "blocking_modal",
    }
)


def _profile_dict(profile: Any) -> dict[str, Any]:
    if isinstance(profile, Mapping):
        return dict(profile)
    if profile is None:
        return {}
    if hasattr(profile, "to_dict") and callable(profile.to_dict):
        result = profile.to_dict()
        return dict(result) if isinstance(result, Mapping) else {"value": result}
    if hasattr(profile, "__dict__"):
        return dict(vars(profile))
    return {"name": str(profile)}


def _skill_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


def _memory_snapshot(memory: Any) -> Any:
    if memory is None:
        return []
    for method in ("snapshot", "recent", "recall"):
        fn = getattr(memory, method, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                try:
                    return fn(limit=10)
                except TypeError:
                    continue
    if isinstance(memory, Mapping):
        return dict(memory)
    if isinstance(memory, Sequence) and not isinstance(memory, (str, bytes)):
        return list(memory)
    return str(memory)


def _derived_sensor(
    raw: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    derive: Any = None,
    derive_known: bool = False,
) -> dict[str, Any]:
    for key in keys:
        if key in raw:
            return sensor_payload(raw.get(key), raw.get(f"{key}_confidence"))
    return sensor_payload(derive if derive_known else None)


def build_planner_state(
    obs: Mapping[str, Any] | Observation,
    *,
    goal: str,
    profile: Any,
    available_skills: Sequence[str],
    current_skill: Any,
    recent_skills: Sequence[Any],
    previous_plan: Plan | Mapping[str, Any] | None,
    memory: Any,
    game_id: str,
) -> PlannerState:
    """Convert a legacy or typed observation into the planner contract."""
    if isinstance(obs, Observation):
        typed = obs
        raw = obs.to_legacy_dict()
    else:
        raw = dict(obs or {})
        typed = Observation.from_legacy_dict(raw)

    triples = extract_sensor_triples(typed)
    sensors: dict[str, dict[str, Any]] = {}
    for name in TRIPLE_SENSORS:
        value, known, confidence = triples[name]
        if not known:
            actual: Any = None
            conf: float | None = None
        else:
            actual = bool(value) if name in _BOOL_SENSORS else value
            conf = float(confidence)
        sensors[name] = {
            "value": actual,
            "known": bool(known),
            "confidence": conf,
        }

    life_phase = str(typed.life_phase or "unknown")
    loading = _derived_sensor(
        raw,
        ("loading", "is_loading"),
        derive=life_phase == "loading",
        derive_known=life_phase != "unknown",
    )
    modal = sensors["blocking_modal"]

    stuck = _derived_sensor(raw, ("severe_stuck", "stuck"))
    if not stuck["known"] and "stuck_hint" in raw:
        hint = str(raw.get("stuck_hint") or "").strip().lower()
        stuck = sensor_payload(hint not in {"", "none", "false", "0"})
    if not stuck["known"] and any(
        key in raw for key in ("stagnant", "stagnation_count")
    ):
        count = int(raw.get("stagnation_count") or raw.get("stagnant") or 0)
        stuck = sensor_payload(count >= 8)

    objective_progress = sensors["objective_progress"]
    previous: dict[str, Any] | None
    if isinstance(previous_plan, Plan):
        previous = previous_plan.to_dict()
    elif isinstance(previous_plan, Mapping):
        previous = dict(previous_plan)
    else:
        previous = None

    recent: list[Any] = []
    for item in recent_skills:
        if isinstance(item, Mapping):
            recent.append(dict(item))
        else:
            recent.append(_skill_name(item))

    timestamp = (
        float(typed.timestamp)
        if isinstance(obs, Observation) or "timestamp" in raw
        else time.time()
    )
    return PlannerState(
        game_id=str(game_id),
        timestamp=timestamp,
        goal=str(goal),
        profile=_profile_dict(profile),
        available_skills=list(dict.fromkeys(str(s) for s in available_skills)),
        current_skill=_skill_name(current_skill),
        recent_skills=recent,
        previous_plan=previous,
        memory=_memory_snapshot(memory),
        sensors=sensors,
        life_phase=life_phase,
        loading=loading,
        modal=modal,
        stuck=stuck,
        objective_progress=objective_progress,
        objective_text=typed.objective_text,
        ocr_text=typed.ocr_text,
        sensor_warnings=list(typed.sensor_warnings),
    )


__all__ = ["build_planner_state"]
