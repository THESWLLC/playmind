"""Fixed synthetic scenario runners using ReplayEnv (no live game / actuators)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from playmind.evaluation.metrics import summarize_replay_results
from playmind.policies.base import PolicyDecision
from playmind.policies.hybrid import HybridPolicy
from playmind.policies.legacy_q import LegacyQPolicy
from playmind.policies.scripted import DEFAULT_SKILL_ORDER, ScriptedPolicy
from playmind.replay_env import ReplayEnv

# ---------------------------------------------------------------------------
# Scenario specs: ordered observation + expected skill label pairs
# ---------------------------------------------------------------------------

SCENARIO_SPECS: dict[str, list[dict[str, Any]]] = {
    "death_recovery": [
        {
            "observation": {
                "vision_player_hp": 0.0,
                "is_dead": True,
                "life_phase": "dead_dialog",
                "has_target": False,
            },
            "skill": "death_recovery",
        },
        {
            "observation": {
                "vision_player_hp": 0.0,
                "is_dead": False,
                "is_ghost": True,
                "life_phase": "ghost",
                "has_target": False,
            },
            "skill": "ghost_runback",
        },
    ],
    "combat_basic": [
        {
            "observation": {
                "vision_player_hp": 0.9,
                "has_target": False,
                "hostiles_near": True,
                "life_phase": "alive",
                "in_combat": False,
            },
            "skill": "acquire_target",
        },
        {
            "observation": {
                "vision_player_hp": 0.85,
                "has_target": True,
                "target_hp": 0.7,
                "in_combat": True,
                "life_phase": "alive",
            },
            "skill": "basic_combat_rotation",
        },
        {
            "observation": {
                "vision_player_hp": 0.8,
                "has_target": True,
                "target_hp": 0.0,
                "in_combat": False,
                "life_phase": "alive",
            },
            "skill": "loot_target",
            "key_events": ["confirmed_kill"],
        },
    ],
    "modal_and_stuck": [
        {
            "observation": {
                "vision_player_hp": 0.9,
                "blocking_modal": True,
                "life_phase": "alive",
                "has_target": False,
            },
            "skill": "clear_modal",
        },
        {
            "observation": {
                "vision_player_hp": 0.9,
                "blocking_modal": False,
                "stagnation_count": 12,
                "motion": 0.0,
                "life_phase": "alive",
                "has_target": False,
                "stuck": True,
            },
            "skill": "unstuck",
        },
    ],
    "low_hp_disengage": [
        {
            "observation": {
                "vision_player_hp": 0.15,
                "has_target": True,
                "in_combat": True,
                "life_phase": "alive",
            },
            "skill": "recover_health",
        },
        {
            "observation": {
                "vision_player_hp": 0.25,
                "has_target": True,
                "in_combat": True,
                "hostiles_near": True,
                "life_phase": "alive",
            },
            "skill": "disengage",
        },
    ],
}


def build_synthetic_session(
    scenario_name: str,
    out_dir: str | Path,
    *,
    steps: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Write a minimal demo session directory consumable by ReplayEnv.from_session."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(steps) if steps is not None else list(SCENARIO_SPECS[scenario_name])
    session_id = f"synth-{scenario_name}"
    meta_path = out / "meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for i, step in enumerate(rows):
            sample = {
                "schema_version": 1,
                "sample_id": f"{session_id}-{i:04d}",
                "session_id": session_id,
                "episode_id": f"ep-{scenario_name}",
                "timestamp": float(i),
                "frame_path": None,
                "observation": dict(step.get("observation") or {}),
                "key_events": list(step.get("key_events") or []),
                "goal": scenario_name,
                "skill": step.get("skill"),
                "label": "success",
                "index": i,
            }
            f.write(json.dumps(sample, sort_keys=True) + "\n")
    session_json = {
        "schema_version": 1,
        "session_id": session_id,
        "scenario": scenario_name,
        "synthetic": True,
        "n_samples": len(rows),
        "created_at": time.time(),
        "outcome": "success",
    }
    (out / "session.json").write_text(
        json.dumps(session_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def run_scenario(
    scenario_name: str,
    policy: Any,
    *,
    work_dir: str | Path | None = None,
    steps: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build synthetic session (if needed) and replay with the given policy."""
    if scenario_name not in SCENARIO_SPECS and steps is None:
        raise KeyError(f"Unknown scenario {scenario_name!r}; known={sorted(SCENARIO_SPECS)}")

    if work_dir is None:
        # In-memory ReplayEnv without touching disk.
        rows = list(steps) if steps is not None else list(SCENARIO_SPECS[scenario_name])
        samples = []
        for i, step in enumerate(rows):
            samples.append(
                {
                    "sample_id": f"{scenario_name}-{i}",
                    "episode_id": f"ep-{scenario_name}",
                    "observation": dict(step.get("observation") or {}),
                    "skill": step.get("skill"),
                    "index": i,
                    "key_events": list(step.get("key_events") or []),
                }
            )
        env = ReplayEnv(samples=samples, policy=policy)
    else:
        session_dir = build_synthetic_session(
            scenario_name, Path(work_dir) / scenario_name, steps=steps
        )
        env = ReplayEnv.from_session(session_dir, policy=policy)

    results = env.run()
    summary = summarize_replay_results(results, policy_name=type(policy).__name__)
    summary["scenario"] = scenario_name
    summary["agreement_rate"] = env.agreement_rate(results)
    summary["steps"] = [r.to_dict() for r in results]
    return summary


def run_all_scenarios(
    policy: Any,
    *,
    work_dir: str | Path | None = None,
    scenario_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    names = list(scenario_names) if scenario_names is not None else sorted(SCENARIO_SPECS.keys())
    per: dict[str, Any] = {}
    agreements: list[float] = []
    for name in names:
        per[name] = run_scenario(name, policy, work_dir=work_dir)
        agreements.append(float(per[name].get("agreement_rate") or 0.0))
    return {
        "scenarios": per,
        "mean_agreement": (sum(agreements) / float(len(agreements))) if agreements else 0.0,
        "n_scenarios": len(names),
        "policy": type(policy).__name__,
    }


class _LegacyStubPolicy:
    """LegacyQ bridge that never crashes on synthetic / incomplete observations.

    Uses ``owned_state_key`` (vision-safe) and a minimal action list so offline
    evaluation comparisons stay CPU-only and KeyError-free.
    """

    model_version: str = "legacy-stub-v1"

    def __init__(self) -> None:
        from playmind.learning import OnlinePolicy, owned_state_key

        self._inner = LegacyQPolicy(
            online_policy=OnlinePolicy(epsilon=0.0, key_fn=owned_state_key),
            raw_actions=["wait", "attack", "target_nearest", "move_north", "interact"],
        )

    def choose_skill(self, context: Mapping[str, Any], allowed_skills: Sequence[str]):
        obs = context.get("obs") if isinstance(context.get("obs"), Mapping) else context
        if not isinstance(obs, Mapping):
            obs = {}
        enriched = dict(obs)
        if "player" not in enriched or not isinstance(enriched.get("player"), Mapping):
            hp = enriched.get("vision_player_hp", enriched.get("player_hp", 0.5))
            try:
                hp_f = float(hp) if hp is not None else 0.5
            except (TypeError, ValueError):
                hp_f = 0.5
            enriched["player"] = {"hp": hp_f, "x": 0, "y": 0}
        else:
            player = dict(enriched["player"])  # type: ignore[index]
            player.setdefault("x", 0)
            player.setdefault("y", 0)
            player.setdefault("hp", 0.5)
            enriched["player"] = player
        ctx = dict(context)
        ctx["obs"] = enriched
        ctx.setdefault(
            "raw_actions",
            ["wait", "attack", "target_nearest", "move_north", "interact"],
        )
        decision = self._inner.choose_skill(ctx, allowed_skills)
        decision.model_version = self.model_version
        decision.reason = f"legacy stub; {decision.reason}"
        return decision


class RandomValidSkillPolicy:
    """Deterministic cycling reference with random-policy class frequencies."""

    model_version = "random-valid-skill-v1"

    def __init__(self) -> None:
        self._index = 0

    def reset_state(self) -> None:
        self._index = 0

    def choose_skill(
        self, context: Mapping[str, Any], allowed_skills: Sequence[str]
    ) -> PolicyDecision:
        allowed = list(dict.fromkeys(str(skill) for skill in allowed_skills))
        skill = allowed[self._index % len(allowed)] if allowed else "wait"
        self._index += 1
        return PolicyDecision(
            skill=skill,
            confidence=1.0 / float(max(1, len(allowed))),
            reason="deterministic random-valid-skill reference",
            model_version=self.model_version,
            allowed_skills=allowed,
        )


class HumanDemonstrationPolicy:
    """Reference policy that emits the recorded human skill label."""

    model_version = "human-demonstration-reference-v1"

    def choose_skill(
        self, context: Mapping[str, Any], allowed_skills: Sequence[str]
    ) -> PolicyDecision:
        allowed = list(dict.fromkeys(str(skill) for skill in allowed_skills))
        label = str(context.get("demo_skill") or "wait")
        skill = label if label in set(allowed) else ("wait" if "wait" in allowed else label)
        return PolicyDecision(
            skill=skill,
            confidence=1.0,
            reason="recorded human demonstration reference",
            model_version=self.model_version,
            allowed_skills=allowed,
            used_fallback=skill != label,
            debug_scores={label: 1.0},
        )


def make_baseline_policies() -> dict[str, Any]:
    """CPU-only baselines that also work when no demonstrations exist."""
    return {
        "scripted": ScriptedPolicy(),
        "legacy_stub": _LegacyStubPolicy(),
        "random_valid_skill": RandomValidSkillPolicy(),
        "hybrid": HybridPolicy(),
        "human_demo": HumanDemonstrationPolicy(),
    }
