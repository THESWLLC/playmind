"""Hybrid high-level policy and BC stub."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from playmind.policies.base import PolicyDecision
from playmind.policies.scripted import (
    CLEAR_MODAL,
    DEATH_RECOVERY,
    DISENGAGE,
    GHOST_RUNBACK,
    RECOVER_HEALTH,
    ScriptedPolicy,
    UNSTUCK,
    _CRITICAL_HP,
    _DEAD_PHASES,
    _hp,
    _obs,
    _phase,
)


class BehaviorCloningPolicy:
    """Wraps a loaded ``SkillPolicyV2`` checkpoint, or stubs until one exists.

    When ``policy`` is set (trained SkillPolicyV2), decisions come from the
    model. Otherwise ``strict=True`` raises; ``strict=False`` returns
    low-confidence so HybridPolicy falls back to scripted.
    """

    model_version: str = "bc-stub"

    def __init__(self, *, strict: bool = False, policy: Any | None = None) -> None:
        self.strict = strict
        self._policy = policy
        if policy is not None:
            self.model_version = str(
                getattr(policy, "model_version", None) or "bc-loaded"
            )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Any,
        *,
        strict: bool = False,
    ) -> "BehaviorCloningPolicy":
        """Load ``SkillPolicyV2`` from ``path`` when the file exists."""
        from pathlib import Path

        from playmind.models.policy_v2 import SkillPolicyV2

        p = Path(path)
        meta = p if p.suffix == ".json" else p.with_suffix(".json")
        if not meta.exists() and not p.exists():
            bc = cls(strict=strict)
            bc.model_version = f"bc:missing:{p.name}"
            return bc
        loaded = SkillPolicyV2.load(p if p.exists() else meta)
        return cls(strict=strict, policy=loaded)

    def choose_skill(
        self,
        context: Mapping[str, Any],
        allowed_skills: Sequence[str],
    ) -> PolicyDecision:
        if self._policy is not None:
            decision = self._policy.choose_skill(context, allowed_skills)
            if isinstance(decision, PolicyDecision):
                return decision
            # Defensive: unexpected return shape
            allowed = list(dict.fromkeys(str(s) for s in allowed_skills))
            return PolicyDecision(
                skill=str(getattr(decision, "skill", "wait")),
                confidence=float(getattr(decision, "confidence", 0.0)),
                reason=str(getattr(decision, "reason", "BC policy")),
                model_version=self.model_version,
                allowed_skills=allowed,
                used_fallback=bool(getattr(decision, "used_fallback", False)),
                temporal_summary=str(context.get("temporal_summary") or ""),
                debug_scores={"bc_available": 1.0},
            )
        if self.strict:
            raise NotImplementedError(
                "BehaviorCloningPolicy requires a trained checkpoint"
            )
        allowed = list(dict.fromkeys(str(s) for s in allowed_skills))
        skill = allowed[0] if allowed else "wait"
        return PolicyDecision(
            skill=skill,
            confidence=0.0,
            reason="BC stub — no checkpoint loaded",
            model_version=self.model_version,
            allowed_skills=allowed,
            used_fallback=True,
            temporal_summary=str(context.get("temporal_summary") or ""),
            debug_scores={"bc_available": 0.0},
        )


def _is_emergency(context: Mapping[str, Any]) -> tuple[bool, str]:
    obs = _obs(context)
    phase = _phase(obs)
    if bool(obs.get("is_dead")) or phase in _DEAD_PHASES:
        return True, "emergency:death"
    if bool(obs.get("is_ghost")) or phase == "ghost":
        return True, "emergency:ghost"
    if bool(obs.get("modal_menu") or obs.get("blocking_modal")):
        return True, "emergency:modal"
    if bool(obs.get("stuck") or context.get("stuck")):
        return True, "emergency:stuck"
    if _hp(obs) < _CRITICAL_HP:
        return True, "emergency:critical_hp"
    return False, ""


class HybridPolicy:
    """Emergency scripted first; else primary (BC or scripted); confidence gate; mask-safe.

    When BC is missing or returns low confidence, falls back to ScriptedPolicy.
    Legacy Q is optional and experimental (off by default).
    """

    model_version: str = "hybrid-v1"

    def __init__(
        self,
        *,
        primary: Any | None = None,
        scripted: ScriptedPolicy | None = None,
        legacy_q: Any | None = None,
        confidence_threshold: float = 0.45,
        use_legacy_q_fallback: bool = False,
    ) -> None:
        self.scripted = scripted or ScriptedPolicy()
        # Primary is BC when available; otherwise scripted.
        self.primary = primary if primary is not None else self.scripted
        self.legacy_q = legacy_q
        self.confidence_threshold = float(confidence_threshold)
        self.use_legacy_q_fallback = bool(use_legacy_q_fallback)

    def _clamp_to_mask(
        self,
        decision: PolicyDecision,
        allowed: Sequence[str],
        *,
        reason_suffix: str = "",
    ) -> PolicyDecision:
        allowed_list = list(dict.fromkeys(str(s) for s in allowed))
        allowed_set = set(allowed_list)
        decision.allowed_skills = allowed_list
        if decision.skill in allowed_set:
            return decision
        # Never accept outside mask — prefer wait, else first allowed.
        replacement = "wait" if "wait" in allowed_set else (allowed_list[0] if allowed_list else "wait")
        return PolicyDecision(
            skill=replacement,
            confidence=min(decision.confidence, 0.2),
            reason=(
                f"{decision.reason}; skill '{decision.skill}' outside mask"
                f"{reason_suffix} → '{replacement}'"
            ),
            model_version=decision.model_version or self.model_version,
            allowed_skills=allowed_list,
            used_fallback=True,
            temporal_summary=decision.temporal_summary,
            debug_scores={**decision.debug_scores, "mask_reject": 1.0},
        )

    def choose_skill(
        self,
        context: Mapping[str, Any],
        allowed_skills: Sequence[str],
    ) -> PolicyDecision:
        allowed = list(dict.fromkeys(str(s) for s in allowed_skills))
        emergency, tag = _is_emergency(context)
        if emergency:
            decision = self.scripted.choose_skill(context, allowed)
            decision.reason = f"{tag}; {decision.reason}"
            decision.model_version = self.model_version
            decision.debug_scores = {**decision.debug_scores, "emergency": 1.0}
            return self._clamp_to_mask(decision, allowed)

        # Primary path (BC or scripted).
        try:
            decision = self.primary.choose_skill(context, allowed)
        except NotImplementedError:
            decision = self.scripted.choose_skill(context, allowed)
            decision.used_fallback = True
            decision.reason = f"primary unavailable; {decision.reason}"
            decision.model_version = self.model_version
            return self._clamp_to_mask(decision, allowed)

        if decision.confidence >= self.confidence_threshold and decision.skill in set(allowed):
            decision.model_version = decision.model_version or self.model_version
            return self._clamp_to_mask(decision, allowed)

        # Low confidence → scripted fallback.
        scripted = self.scripted.choose_skill(context, allowed)
        scripted.used_fallback = True
        scripted.reason = (
            f"primary conf={decision.confidence:.2f}<{self.confidence_threshold:.2f}; "
            f"{scripted.reason}"
        )
        scripted.debug_scores = {
            **decision.debug_scores,
            **scripted.debug_scores,
            "primary_confidence": float(decision.confidence),
        }
        scripted.model_version = self.model_version

        if (
            self.use_legacy_q_fallback
            and self.legacy_q is not None
            and scripted.confidence < self.confidence_threshold
        ):
            legacy = self.legacy_q.choose_skill(context, allowed)
            legacy.used_fallback = True
            legacy.reason = f"legacy-q experimental; {legacy.reason}"
            legacy.model_version = self.model_version
            return self._clamp_to_mask(legacy, allowed, reason_suffix=" via legacy")

        return self._clamp_to_mask(scripted, allowed)


# Re-export emergency skill names for tests / callers.
__all__ = [
    "BehaviorCloningPolicy",
    "HybridPolicy",
    "CLEAR_MODAL",
    "DEATH_RECOVERY",
    "DISENGAGE",
    "GHOST_RUNBACK",
    "RECOVER_HEALTH",
    "UNSTUCK",
]
