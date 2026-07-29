"""Deterministic rule primitives for human demonstration segmentation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuleMatch:
    skill_label: str
    start_index: int
    end_index: int
    confidence: float
    rule_id: str


def event_token(event: Mapping[str, Any]) -> str:
    """Extract a normalized action token from varied recorder event shapes."""
    for name in ("action", "key", "button", "name", "value"):
        value = event.get(name)
        if value not in (None, ""):
            return str(value).strip().lower().replace("key.", "")
    event_type = str(event.get("type") or "").lower()
    if event_type in {"stop", "food", "attack", "forward"}:
        return event_type
    return ""


def observation_hp(observation: Mapping[str, Any]) -> float | None:
    for name in ("player_hp", "vision_player_hp", "hp"):
        value = observation.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    player = observation.get("player")
    if isinstance(player, Mapping) and player.get("hp") is not None:
        try:
            return float(player["hp"])
        except (TypeError, ValueError):
            return None
    return None


def combat_sequence_rule(events: Sequence[Mapping[str, Any]]) -> list[RuleMatch]:
    """tab → forward → attack maps to acquire → approach → engage."""
    tokens = [event_token(event) for event in events]
    tab = next((i for i, token in enumerate(tokens) if token in {"tab", "target_nearest"}), None)
    if tab is None:
        return []
    forward = next(
        (
            i
            for i in range(tab + 1, len(tokens))
            if tokens[i] in {"w", "forward", "move_forward"}
        ),
        None,
    )
    if forward is None:
        return []
    attack = next(
        (
            i
            for i in range(forward + 1, len(tokens))
            if tokens[i] in {"attack", "left", "1"}
            or str(events[i].get("type") or "") == "attack"
        ),
        None,
    )
    if attack is None:
        return []
    return [
        RuleMatch("acquire_target", tab, tab, 0.96, "combat.tab_acquire"),
        RuleMatch("approach_target", forward, max(forward, attack - 1), 0.91, "combat.forward_approach"),
        RuleMatch("engage_target", attack, attack, 0.94, "combat.attack_engage"),
    ]


def low_hp_recovery_rule(
    events: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> list[RuleMatch]:
    """Low HP plus stop then food maps to disengage then recovery."""
    if not any(
        hp is not None and hp <= 0.35
        for hp in (observation_hp(observation) for observation in observations)
    ):
        return []
    tokens = [event_token(event) for event in events]
    stop = next(
        (
            i
            for i, (event, token) in enumerate(zip(events, tokens))
            if token in {"stop", "s", "escape"}
            or (
                token == "w"
                and str(event.get("type") or "").lower() == "key_up"
            )
        ),
        None,
    )
    if stop is None:
        return []
    food = next(
        (
            i
            for i in range(stop, len(tokens))
            if tokens[i] in {"food", "eat", "drink", "consume_food", "0", "-"}
        ),
        None,
    )
    if food is None:
        return []
    return [
        RuleMatch("disengage", stop, max(stop, food - 1), 0.88, "recovery.low_hp_stop"),
        RuleMatch("recover_health", food, food, 0.95, "recovery.consume_food"),
    ]


def stagnation_rule(
    events: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> list[RuleMatch]:
    """Repeated near-zero motion or an explicit stagnation count implies unstuck."""
    explicit = any(
        int(observation.get("stagnation_count") or observation.get("stagnant") or 0) >= 3
        for observation in observations
    )
    motion_values: list[float] = []
    for observation in observations:
        value = observation.get("motion")
        if value is not None:
            try:
                motion_values.append(float(value))
            except (TypeError, ValueError):
                pass
    attempted_motion = any(
        event_token(event) in {"w", "a", "s", "d", "forward", "move_forward"}
        for event in events
    )
    repeated_zero = (
        attempted_motion
        and len(motion_values) >= 3
        and all(abs(value) <= 0.05 for value in motion_values[-3:])
    )
    if not explicit and not repeated_zero:
        return []
    end = max(0, len(events) - 1)
    confidence = 0.93 if explicit and repeated_zero else 0.82
    return [RuleMatch("unstuck", 0, end, confidence, "navigation.motion_stagnation")]


def lifecycle_rule(
    events: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> list[RuleMatch]:
    """Recognize strong death, ghost, and blocking-modal states."""
    end = max(0, len(events) - 1)
    if any(
        observation.get("is_dead")
        or str(observation.get("life_phase") or "") in {"dead_dialog", "confirm", "rez_picker"}
        for observation in observations
    ):
        return [RuleMatch("death_recovery", 0, end, 0.98, "lifecycle.dead")]
    if any(
        observation.get("is_ghost")
        or str(observation.get("life_phase") or "") == "ghost"
        for observation in observations
    ):
        return [RuleMatch("ghost_runback", 0, end, 0.97, "lifecycle.ghost")]
    if any(
        observation.get("blocking_modal") or observation.get("modal_menu")
        for observation in observations
    ):
        return [RuleMatch("clear_modal", 0, end, 0.90, "ui.blocking_modal")]
    return []


__all__ = [
    "RuleMatch",
    "combat_sequence_rule",
    "event_token",
    "lifecycle_rule",
    "low_hp_recovery_rule",
    "observation_hp",
    "stagnation_rule",
]
