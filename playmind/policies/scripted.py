"""Deterministic scripted high-level policy."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from playmind.policies.base import PolicyDecision

# Canonical skill names (match playmind.skills.*.name / SkillRegistry keys).
DEATH_RECOVERY = "death_recovery"
GHOST_RUNBACK = "ghost_runback"
CLEAR_MODAL = "clear_modal"
UNSTUCK = "unstuck"
RECOVER_HEALTH = "recover_health"
DISENGAGE = "disengage"
ACQUIRE_TARGET = "acquire_target"
VALIDATE_TARGET = "validate_target"
APPROACH_TARGET = "approach_target"
ENGAGE_TARGET = "engage_target"
BASIC_COMBAT = "basic_combat_rotation"
LOOT_TARGET = "loot_target"
EXPLORE = "explore"
INTERACT = "interact"
WAIT = "wait"

DEFAULT_SKILL_ORDER = (
    DEATH_RECOVERY,
    GHOST_RUNBACK,
    CLEAR_MODAL,
    UNSTUCK,
    RECOVER_HEALTH,
    DISENGAGE,
    ACQUIRE_TARGET,
    VALIDATE_TARGET,
    APPROACH_TARGET,
    ENGAGE_TARGET,
    BASIC_COMBAT,
    LOOT_TARGET,
    EXPLORE,
    INTERACT,
    WAIT,
)

_DEAD_PHASES = frozenset({"dead_dialog", "confirm", "rez_picker"})
_LOW_HP = 0.35
_CRITICAL_HP = 0.20


def _obs(context: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept either a bare obs mapping or a wrapped {\"obs\": ...} context."""
    if "obs" in context and isinstance(context["obs"], Mapping):
        return context["obs"]  # type: ignore[return-value]
    return context


def _hp(obs: Mapping[str, Any]) -> float:
    raw = obs.get("vision_player_hp")
    if raw is None:
        raw = obs.get("player_hp")
    if raw is None and isinstance(obs.get("player"), Mapping):
        raw = obs["player"].get("hp")  # type: ignore[index]
    if raw is None:
        return 0.5
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.5


def _phase(obs: Mapping[str, Any]) -> str:
    return str(obs.get("life_phase") or "alive")


def _pick(preferred: Sequence[str], allowed: set[str], fallback: str = WAIT) -> str | None:
    for name in preferred:
        if name in allowed:
            return name
    if fallback in allowed:
        return fallback
    return next(iter(allowed), None)


class ScriptedPolicy:
    """Deterministic skill selection from observation / life-phase cues."""

    model_version: str = "scripted-v1"

    def choose_skill(
        self,
        context: Mapping[str, Any],
        allowed_skills: Sequence[str],
    ) -> PolicyDecision:
        allowed = list(dict.fromkeys(str(s) for s in allowed_skills))
        allowed_set = set(allowed)
        obs = _obs(context)
        phase = _phase(obs)
        hp = _hp(obs)
        is_dead = bool(obs.get("is_dead")) or phase in _DEAD_PHASES
        is_ghost = bool(obs.get("is_ghost")) or phase == "ghost"
        modal = bool(obs.get("modal_menu") or obs.get("blocking_modal"))
        stuck = bool(obs.get("stuck") or context.get("stuck"))
        has_target = bool(obs.get("has_target"))
        in_combat = bool(obs.get("in_combat"))
        hostiles = bool(obs.get("hostiles_near"))
        target_validated = bool(obs.get("target_validated") or obs.get("valid_target"))
        lootable = bool(obs.get("lootable") or obs.get("can_loot"))
        scores: dict[str, float] = {}

        def decide(skill: str | None, confidence: float, reason: str) -> PolicyDecision:
            chosen = skill if skill and skill in allowed_set else _pick(DEFAULT_SKILL_ORDER, allowed_set)
            if chosen is None:
                chosen = WAIT
                used_fallback = True
                reason = f"{reason}; no allowed skills — default wait"
            else:
                used_fallback = skill is None or skill not in allowed_set
                if used_fallback and skill and skill not in allowed_set:
                    reason = f"{reason}; '{skill}' masked → '{chosen}'"
            return PolicyDecision(
                skill=chosen,
                confidence=confidence,
                reason=reason,
                model_version=self.model_version,
                allowed_skills=allowed,
                used_fallback=used_fallback,
                temporal_summary=str(context.get("temporal_summary") or ""),
                debug_scores=dict(scores),
            )

        # --- Emergency / ownership priorities ---
        if is_dead:
            scores[DEATH_RECOVERY] = 1.0
            return decide(
                _pick([DEATH_RECOVERY], allowed_set),
                1.0,
                f"dead phase={phase}",
            )
        if is_ghost:
            scores[GHOST_RUNBACK] = 1.0
            return decide(
                _pick([GHOST_RUNBACK], allowed_set),
                1.0,
                "ghost runback",
            )
        if modal:
            scores[CLEAR_MODAL] = 0.95
            return decide(
                _pick([CLEAR_MODAL], allowed_set),
                0.95,
                "blocking modal",
            )
        if stuck:
            scores[UNSTUCK] = 0.9
            return decide(
                _pick([UNSTUCK], allowed_set),
                0.9,
                "stuck recovery",
            )

        # --- Survival ---
        if hp < _CRITICAL_HP and in_combat:
            scores[DISENGAGE] = 0.85
            scores[RECOVER_HEALTH] = 0.7
            return decide(
                _pick([DISENGAGE, RECOVER_HEALTH], allowed_set),
                0.85,
                f"critical hp={hp:.2f} in combat",
            )
        if hp < _LOW_HP:
            scores[RECOVER_HEALTH] = 0.8
            scores[DISENGAGE] = 0.65 if in_combat else 0.3
            preferred = [RECOVER_HEALTH, DISENGAGE] if in_combat else [RECOVER_HEALTH, WAIT]
            return decide(
                _pick(preferred, allowed_set),
                0.8,
                f"low hp={hp:.2f}",
            )

        # --- Loot after combat ---
        if lootable and not has_target:
            scores[LOOT_TARGET] = 0.75
            return decide(
                _pick([LOOT_TARGET], allowed_set),
                0.75,
                "loot available",
            )

        # --- Targeting / combat ---
        if not has_target:
            if hostiles:
                scores[ACQUIRE_TARGET] = 0.8
                return decide(
                    _pick([ACQUIRE_TARGET, EXPLORE], allowed_set),
                    0.8,
                    "hostiles nearby, no target",
                )
            scores[EXPLORE] = 0.7
            return decide(
                _pick([EXPLORE, ACQUIRE_TARGET, INTERACT], allowed_set),
                0.7,
                "no target — explore",
            )

        if has_target and not target_validated and VALIDATE_TARGET in allowed_set:
            scores[VALIDATE_TARGET] = 0.72
            return decide(
                VALIDATE_TARGET,
                0.72,
                "target needs validation",
            )

        if has_target and in_combat:
            scores[BASIC_COMBAT] = 0.78
            scores[ENGAGE_TARGET] = 0.7
            return decide(
                _pick([BASIC_COMBAT, ENGAGE_TARGET], allowed_set),
                0.78,
                "in combat with target",
            )

        if has_target:
            scores[ENGAGE_TARGET] = 0.74
            scores[APPROACH_TARGET] = 0.6
            scores[BASIC_COMBAT] = 0.55
            return decide(
                _pick([ENGAGE_TARGET, APPROACH_TARGET, BASIC_COMBAT], allowed_set),
                0.74,
                "has target — engage",
            )

        scores[WAIT] = 0.4
        return decide(_pick([WAIT, EXPLORE], allowed_set), 0.4, "idle fallback")
