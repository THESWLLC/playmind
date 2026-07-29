"""Experimental legacy tabular-Q bridge policy."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from playmind.learning import OnlinePolicy
from playmind.policies.base import PolicyDecision

# Heuristic map from raw action strings → skill names.
_ACTION_TO_SKILL: dict[str, str] = {
    "release_spirit": "death_recovery",
    "target_nearest": "acquire_target",
    "attack": "basic_combat_rotation",
    "loot": "loot_target",
    "interact": "interact",
    "wait": "wait",
    "move_north": "explore",
    "move_south": "explore",
    "move_east": "explore",
    "move_west": "explore",
}

RAW_ACTION_BRIDGE = "RawActionBridge"


def raw_action_to_skill(action: str) -> str:
    """Map a legacy raw action string to a skill name (heuristic)."""
    a = (action or "").strip().lower()
    if a in _ACTION_TO_SKILL:
        return _ACTION_TO_SKILL[a]
    if a.startswith("hold:") or a.startswith("move_"):
        return "explore"
    if a.startswith("key:1") or a.startswith("key:2") or a.startswith("key:3"):
        return "basic_combat_rotation"
    if a.startswith("key:tab") or a == "key:tab":
        return "acquire_target"
    if a.startswith("key:esc") or "escape" in a:
        return "clear_modal"
    if "graveyard" in a or "release" in a or "resurrect" in a:
        return "death_recovery"
    if a.startswith("click_label:close") or a.startswith("click_label:cancel"):
        return "clear_modal"
    if a.startswith("ability:") or a.startswith("bind:"):
        return "basic_combat_rotation"
    return RAW_ACTION_BRIDGE


class LegacyQPolicy:
    """Wraps OnlinePolicy; maps chosen raw actions to skills.

    Experimental fallback only — not the primary Learning Architecture V2 path.
    """

    experimental: bool = True
    model_version: str = "legacy-q-v1"

    def __init__(
        self,
        online_policy: OnlinePolicy | None = None,
        *,
        raw_actions: Sequence[str] | None = None,
    ) -> None:
        self.online_policy = online_policy or OnlinePolicy(epsilon=0.1)
        self.raw_actions = list(raw_actions) if raw_actions is not None else None

    def choose_skill(
        self,
        context: Mapping[str, Any],
        allowed_skills: Sequence[str],
    ) -> PolicyDecision:
        allowed = list(dict.fromkeys(str(s) for s in allowed_skills))
        allowed_set = set(allowed)
        obs = context.get("obs") if isinstance(context.get("obs"), Mapping) else context
        if not isinstance(obs, Mapping):
            obs = {}

        actions = list(self.raw_actions) if self.raw_actions else list(
            context.get("raw_actions") or []
        )
        if not actions:
            from playmind.learning import OWNED_ACTIONS

            actions = list(OWNED_ACTIONS)

        raw = self.online_policy.choose(dict(obs), actions)
        mapped = raw_action_to_skill(raw)
        q_val = float(self.online_policy.value(dict(obs), raw))
        # Softmax-ish confidence from Q magnitude (bounded).
        confidence = max(0.05, min(0.85, 0.35 + 0.1 * q_val))

        skill = mapped if mapped in allowed_set else None
        used_fallback = False
        reason = f"legacy Q chose raw '{raw}' → skill '{mapped}'"
        if skill is None:
            # Prefer bridge skill when present; else first allowed.
            if RAW_ACTION_BRIDGE in allowed_set:
                skill = RAW_ACTION_BRIDGE
                used_fallback = True
                reason += f"; '{mapped}' not in mask → {RAW_ACTION_BRIDGE}"
            elif allowed:
                skill = allowed[0]
                used_fallback = True
                reason += f"; '{mapped}' not in mask → '{skill}'"
            else:
                skill = RAW_ACTION_BRIDGE
                used_fallback = True
                reason += "; empty mask"

        return PolicyDecision(
            skill=skill,
            confidence=confidence,
            reason=reason + " [experimental]",
            model_version=self.model_version,
            allowed_skills=allowed,
            used_fallback=used_fallback,
            temporal_summary=str(context.get("temporal_summary") or ""),
            debug_scores={"q_value": q_val, "raw_action_hash": float(hash(raw) % 10_000)},
        )
