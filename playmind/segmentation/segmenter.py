"""Skill segmenter interface and deterministic rule-based implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from playmind.segmentation.rules import (
    RuleMatch,
    combat_sequence_rule,
    lifecycle_rule,
    low_hp_recovery_rule,
    stagnation_rule,
)


@dataclass
class Segment:
    """A labeled inclusive span within an input/observation window."""

    skill_label: str
    start_index: int
    end_index: int
    confidence: float
    rule_ids: list[str] = field(default_factory=list)
    training_eligible: bool = True
    manual_override: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def skill(self) -> str:
        return self.skill_label

    @property
    def label(self) -> str:
        return self.skill_label

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill_label,
            "skill_label": self.skill_label,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "confidence": self.confidence,
            "rule_ids": list(self.rule_ids),
            "training_eligible": self.training_eligible,
            "manual_override": self.manual_override,
            "metadata": dict(self.metadata),
        }


class SkillSegmenter(Protocol):
    """Interface implemented by demonstration skill segmenters."""

    def segment(self, window: Any) -> list[Segment]: ...


def _event_from_legacy(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    text = str(value)
    if text.startswith("key:"):
        return {"type": "key_down", "key": text.split(":", 1)[1]}
    if text.startswith("hold:"):
        parts = text.split(":")
        return {"type": "key_down", "key": parts[1] if len(parts) > 1 else text}
    return {"type": "action", "action": text}


def _window_parts(window: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    if isinstance(window, Mapping):
        nested = window.get("samples") or window.get("rows")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            return _window_parts(nested)
        raw_events = (
            window.get("physical_events")
            or window.get("events")
            or window.get("key_events")
            or []
        )
        if isinstance(raw_events, Sequence) and not isinstance(raw_events, (str, bytes)):
            events.extend(_event_from_legacy(event) for event in raw_events)
        raw_observations = window.get("observations")
        if isinstance(raw_observations, Sequence) and not isinstance(
            raw_observations, (str, bytes)
        ):
            observations.extend(
                dict(observation)
                for observation in raw_observations
                if isinstance(observation, Mapping)
            )
        elif isinstance(window.get("observation"), Mapping):
            observations.append(dict(window["observation"]))
        else:
            sensor_keys = {
                "player_hp",
                "vision_player_hp",
                "motion",
                "stagnation_count",
                "is_dead",
                "is_ghost",
                "life_phase",
                "blocking_modal",
            }
            if sensor_keys.intersection(window):
                observations.append(dict(window))
        return events, observations

    if isinstance(window, Sequence) and not isinstance(window, (str, bytes)):
        for item in window:
            if not isinstance(item, Mapping):
                events.append(_event_from_legacy(item))
                continue
            item_events, item_observations = _window_parts(item)
            events.extend(item_events)
            observations.extend(item_observations)
    return events, observations


class RuleBasedSkillSegmenter:
    """Apply high-precision, deterministic rules in a fixed priority order."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.70,
        allow_low_confidence: bool = False,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.confidence_threshold = float(confidence_threshold)
        self.allow_low_confidence = bool(allow_low_confidence)

    def segment(
        self,
        window: Any,
        *,
        manual_override: str | Mapping[str, Any] | Sequence[Any] | None = None,
    ) -> list[Segment]:
        events, observations = _window_parts(window)
        override = manual_override
        if override is None and isinstance(window, Mapping):
            override = window.get("manual_override")
            if override is None and isinstance(window.get("segmentation_meta"), Mapping):
                override = window["segmentation_meta"].get("manual_override")
        if override is not None:
            return self._manual_segments(override, max(0, len(events) - 1))

        # Lifecycle and recovery states supersede ordinary combat/navigation.
        rule_groups = (
            lifecycle_rule(events, observations),
            low_hp_recovery_rule(events, observations),
            combat_sequence_rule(events),
            stagnation_rule(events, observations),
        )
        matches = next((group for group in rule_groups if group), [])
        return [self._from_match(match) for match in matches]

    def _from_match(self, match: RuleMatch) -> Segment:
        eligible = (
            match.confidence >= self.confidence_threshold or self.allow_low_confidence
        )
        return Segment(
            skill_label=match.skill_label,
            start_index=match.start_index,
            end_index=match.end_index,
            confidence=match.confidence,
            rule_ids=[match.rule_id],
            training_eligible=eligible,
        )

    def _manual_segments(self, override: Any, default_end: int) -> list[Segment]:
        values = (
            list(override)
            if isinstance(override, Sequence) and not isinstance(override, (str, bytes))
            else [override]
        )
        segments: list[Segment] = []
        for value in values:
            if isinstance(value, Mapping):
                label = value.get("skill") or value.get("skill_label") or value.get("label")
                if not label:
                    raise ValueError("manual override mapping requires a skill label")
                start = int(value.get("start_index", 0))
                end = int(value.get("end_index", default_end))
                metadata = {
                    key: item
                    for key, item in value.items()
                    if key
                    not in {"skill", "skill_label", "label", "start_index", "end_index"}
                }
            else:
                label = str(value)
                start, end, metadata = 0, default_end, {}
            segments.append(
                Segment(
                    skill_label=str(label),
                    start_index=start,
                    end_index=end,
                    confidence=1.0,
                    rule_ids=["manual.override"],
                    training_eligible=True,
                    manual_override=True,
                    metadata=metadata,
                )
            )
        return segments


__all__ = ["RuleBasedSkillSegmenter", "Segment", "SkillSegmenter"]
