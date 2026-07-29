"""Strict parsing and safety validation for model-generated plans."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .contract import (
    ALLOWED_REPLAN_EVENTS,
    PLAN_SCHEMA_VERSION,
    Plan,
    PlanValidationResult,
)

_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "goal",
        "skills",
        "replan_on",
        "confidence",
        "reason_code",
        "summary",
    }
)
_DEATH_SKILLS = frozenset({"death_recovery", "ghost_runback"})
_COMBAT_SKILLS = frozenset(
    {
        "acquire_target",
        "validate_target",
        "approach_target",
        "engage_target",
        "basic_combat_rotation",
        "loot_target",
    }
)


def _known_runtime_skills() -> set[str]:
    try:
        from playmind.skills import list_skills

        return set(list_skills())
    except ImportError:
        return set()


def _clean_json_text(text: str) -> tuple[str, bool]:
    clean = str(text or "").strip()
    if clean.startswith("```") and clean.endswith("```"):
        lines = clean.splitlines()
        if len(lines) >= 3:
            clean = "\n".join(lines[1:-1]).strip()
            return clean, True
    return clean, False


class PlanValidator:
    def __init__(
        self,
        *,
        max_plan_length: int = 5,
        min_seconds: int = 1,
        max_seconds: int = 120,
    ) -> None:
        self.max_plan_length = max(1, int(max_plan_length))
        self.min_seconds = int(min_seconds)
        self.max_seconds = int(max_seconds)

    def validate(
        self,
        plan: Plan,
        allowed_skills: Sequence[str],
        *,
        repaired: bool = False,
    ) -> PlanValidationResult:
        errors: list[str] = []
        allowed = set(str(x) for x in allowed_skills)
        known = _known_runtime_skills() | allowed

        if plan.schema_version != PLAN_SCHEMA_VERSION:
            errors.append(
                f"schema_version must be {PLAN_SCHEMA_VERSION}, got {plan.schema_version}"
            )
        if not plan.goal.strip():
            errors.append("goal must be a non-empty string")
        if not plan.skills:
            errors.append("skills must contain at least one skill")
        if len(plan.skills) > self.max_plan_length:
            errors.append(
                f"plan has {len(plan.skills)} skills; maximum is {self.max_plan_length}"
            )

        names: list[str] = []
        for index, step in enumerate(plan.skills):
            name = step.name.strip()
            names.append(name)
            if not name:
                errors.append(f"skills[{index}].name must be non-empty")
            elif name not in known:
                errors.append(f"skills[{index}] invented skill: {name!r}")
            elif name not in allowed:
                errors.append(f"skills[{index}] unavailable skill: {name!r}")
            if isinstance(step.max_seconds, bool) or not isinstance(
                step.max_seconds, int
            ):
                errors.append(f"skills[{index}].max_seconds must be an integer")
            elif not self.min_seconds <= step.max_seconds <= self.max_seconds:
                errors.append(
                    f"skills[{index}].max_seconds must be in "
                    f"[{self.min_seconds}, {self.max_seconds}]"
                )
            if not isinstance(step.constraints, dict):
                errors.append(f"skills[{index}].constraints must be an object")

        invalid_events = sorted(
            {event for event in plan.replan_on if event not in ALLOWED_REPLAN_EVENTS}
        )
        if invalid_events:
            errors.append(f"invalid replan_on events: {', '.join(invalid_events)}")
        if len(plan.replan_on) != len(set(plan.replan_on)):
            errors.append("replan_on events must not contain duplicates")
        if not 0.0 <= plan.confidence <= 1.0:
            errors.append("confidence must be in [0, 1]")
        if set(names) & _DEATH_SKILLS and set(names) & _COMBAT_SKILLS:
            errors.append(
                "plan mixes death/ghost recovery with combat skills; replan after recovery"
            )

        return PlanValidationResult(
            ok=not errors,
            plan=plan if not errors else None,
            errors=errors,
            repaired=bool(repaired),
        )

    def validate_or_parse(
        self,
        text: str | Mapping[str, Any] | Plan,
        allowed_skills: Sequence[str],
    ) -> PlanValidationResult:
        if isinstance(text, Plan):
            return self.validate(text, allowed_skills)

        repaired = False
        if isinstance(text, Mapping):
            raw: Any = dict(text)
        else:
            clean, repaired = _clean_json_text(str(text))
            try:
                raw = json.loads(clean)
            except (json.JSONDecodeError, TypeError) as exc:
                return PlanValidationResult(
                    False,
                    errors=[f"malformed plan JSON: {exc}"],
                    repaired=repaired,
                )
        if not isinstance(raw, Mapping):
            return PlanValidationResult(
                False,
                errors=["plan JSON must be an object"],
                repaired=repaired,
            )

        missing = sorted(_REQUIRED_FIELDS - set(raw))
        structural_errors: list[str] = []
        if missing:
            structural_errors.append(f"missing required fields: {', '.join(missing)}")
        if not isinstance(raw.get("skills"), list):
            structural_errors.append("skills must be an array")
        else:
            for index, item in enumerate(raw["skills"]):
                if not isinstance(item, Mapping):
                    structural_errors.append(f"skills[{index}] must be an object")
                    continue
                for field_name in ("name", "until", "max_seconds", "constraints"):
                    if field_name not in item:
                        structural_errors.append(
                            f"skills[{index}] missing field {field_name!r}"
                        )
                if "name" in item and not isinstance(item["name"], str):
                    structural_errors.append(f"skills[{index}].name must be a string")
                if "until" in item and item["until"] is not None and not isinstance(
                    item["until"], str
                ):
                    structural_errors.append(
                        f"skills[{index}].until must be a string or null"
                    )
                if "constraints" in item and not isinstance(
                    item["constraints"], Mapping
                ):
                    structural_errors.append(
                        f"skills[{index}].constraints must be an object"
                    )
                if "max_seconds" in item and (
                    isinstance(item["max_seconds"], bool)
                    or not isinstance(item["max_seconds"], int)
                ):
                    structural_errors.append(
                        f"skills[{index}].max_seconds must be an integer"
                    )
        if not isinstance(raw.get("replan_on"), list):
            structural_errors.append("replan_on must be an array")
        elif any(not isinstance(event, str) for event in raw["replan_on"]):
            structural_errors.append("replan_on events must be strings")
        if not isinstance(raw.get("goal"), str):
            structural_errors.append("goal must be a string")
        if isinstance(raw.get("confidence"), bool) or not isinstance(
            raw.get("confidence"), (int, float)
        ):
            structural_errors.append("confidence must be numeric")
        for field_name in ("reason_code", "summary"):
            if not isinstance(raw.get(field_name), str):
                structural_errors.append(f"{field_name} must be a string")
        if structural_errors:
            return PlanValidationResult(
                False, errors=structural_errors, repaired=repaired
            )
        try:
            plan = Plan.from_dict(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            return PlanValidationResult(
                False,
                errors=[f"invalid plan fields: {exc}"],
                repaired=repaired,
            )
        return self.validate(plan, allowed_skills, repaired=repaired)


def validate_or_parse(
    text: str | Mapping[str, Any] | Plan,
    allowed_skills: Sequence[str],
    *,
    max_plan_length: int = 5,
) -> PlanValidationResult:
    return PlanValidator(max_plan_length=max_plan_length).validate_or_parse(
        text, allowed_skills
    )


def retry_correction_prompt(errors: Sequence[str]) -> str:
    problems = "\n".join(f"- {str(error)}" for error in errors)
    return (
        "Your previous output was rejected for these reasons:\n"
        f"{problems}\n"
        "Return ONLY a corrected JSON plan object using the required schema. "
        "Do not include markdown or chain-of-thought."
    )


__all__ = ["PlanValidator", "retry_correction_prompt", "validate_or_parse"]
