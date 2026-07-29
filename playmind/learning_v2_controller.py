"""Learning Architecture V2 controller — skill policy over raw-action Q.

Used by OwnedGameLoop when ``learning_v2.enabled`` is true. Legacy Q remains
available via ``policy_mode: legacy_q`` or hybrid experimental fallback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from playmind.action_masking import mask_skills, validate_action
from playmind.episodes import EpisodeManager
from playmind.events import detect_events
from playmind.history import TemporalHistory
from playmind.observations import Observation
from playmind.policies.hybrid import BehaviorCloningPolicy, HybridPolicy
from playmind.policies.legacy_q import LegacyQPolicy
from playmind.policies.scripted import DEFAULT_SKILL_ORDER, ScriptedPolicy
from playmind.rewards_v2 import reward_from_events
from playmind.skills import list_skills
from playmind.skills.base import SkillContext
from playmind.skills.runtime import SkillRuntime


@dataclass
class LearningV2Config:
    enabled: bool = False
    policy_mode: str = "hybrid"  # scripted | hybrid | legacy_q | behavior_clone
    legacy_q_fallback: bool = False
    history_length: int = 16
    confidence_threshold: float = 0.45
    bc_checkpoint: str | None = None
    use_rewards_v2: bool = True
    track_episodes: bool = True
    settings: Any | None = None  # LearningV2Settings when loaded from owned config

    @classmethod
    def from_owned_dict(cls, owned: dict[str, Any]) -> LearningV2Config:
        from playmind.config_v2 import LearningV2Settings

        settings = LearningV2Settings.load_from_owned_config(owned)
        if settings.enabled:
            settings.validate()
        # GUI alias already normalized in LearningV2Settings; keep model_path fallback
        raw = owned.get("learning_v2") or {}
        if not isinstance(raw, dict):
            raw = {}
        ckpt = settings.bc_checkpoint or raw.get("model_path")
        if ckpt is not None:
            ckpt = str(ckpt).strip() or None
        return cls(
            enabled=bool(settings.enabled),
            policy_mode=str(settings.policy_mode),
            legacy_q_fallback=bool(settings.legacy_q_fallback),
            history_length=int(settings.history_length),
            confidence_threshold=float(settings.confidence_threshold),
            bc_checkpoint=ckpt,
            use_rewards_v2=bool(settings.use_rewards_v2),
            track_episodes=bool(settings.track_episodes),
            settings=settings,
        )


@dataclass
class LearningV2Controller:
    cfg: LearningV2Config
    history: TemporalHistory = field(init=False)
    runtime: SkillRuntime = field(default_factory=SkillRuntime)
    episode_mgr: EpisodeManager | None = None
    hybrid: HybridPolicy | None = None
    scripted: ScriptedPolicy = field(default_factory=ScriptedPolicy)
    last_decision_reason: str = ""
    last_skill: str | None = None
    last_policy_mode: str = ""
    last_confidence: float = 0.0
    last_model_version: str | None = None
    last_allowed_skills: list[str] = field(default_factory=list)
    last_masked_skills: list[str] = field(default_factory=list)
    last_reward_breakdown: dict[str, Any] = field(default_factory=dict)
    _legacy_policy: Any = None

    def __post_init__(self) -> None:
        self.history = TemporalHistory(maxlen=max(4, int(self.cfg.history_length)))

    def attach_legacy_q(self, online_policy: Any) -> None:
        self._legacy_policy = online_policy
        legacy = None
        if self.cfg.legacy_q_fallback or self.cfg.policy_mode == "legacy_q":
            legacy = LegacyQPolicy(online_policy)
        if self.cfg.bc_checkpoint:
            bc = BehaviorCloningPolicy.from_checkpoint(
                self.cfg.bc_checkpoint, strict=False
            )
        else:
            bc = BehaviorCloningPolicy(strict=False)

        if self.cfg.policy_mode == "scripted":
            self.hybrid = None
        elif self.cfg.policy_mode == "legacy_q":
            self.hybrid = HybridPolicy(
                primary=legacy or self.scripted,
                scripted=self.scripted,
                legacy_q=legacy,
                confidence_threshold=self.cfg.confidence_threshold,
                use_legacy_q_fallback=True,
            )
        elif self.cfg.policy_mode == "behavior_clone":
            # Prefer BC; still allow scripted emergency fallback via HybridPolicy.
            self.hybrid = HybridPolicy(
                primary=bc,
                scripted=self.scripted,
                legacy_q=legacy,
                confidence_threshold=self.cfg.confidence_threshold,
                use_legacy_q_fallback=self.cfg.legacy_q_fallback,
            )
        else:
            # hybrid: BC → scripted; optional legacy last
            self.hybrid = HybridPolicy(
                primary=bc,
                scripted=self.scripted,
                legacy_q=legacy,
                confidence_threshold=self.cfg.confidence_threshold,
                use_legacy_q_fallback=self.cfg.legacy_q_fallback,
            )

    def _apply_skill_limits(self, skill: Any) -> None:
        """Apply config skill_timeouts / retry_limits onto a fresh skill instance."""
        settings = self.cfg.settings
        if settings is None or skill is None:
            return
        name = getattr(skill, "name", None)
        if not name:
            return
        timeouts = getattr(settings, "skill_timeouts", None) or {}
        retries = getattr(settings, "skill_retry_limits", None) or {}
        if name in timeouts:
            try:
                skill.timeout_s = float(timeouts[name])
            except (TypeError, ValueError):
                pass
        if name in retries:
            try:
                skill.retry_limit = int(retries[name])
            except (TypeError, ValueError):
                pass

    def ensure_episode(self, data_dir: Any, model_version: str = "learning-v2") -> None:
        if not self.cfg.track_episodes:
            return
        if self.episode_mgr is None:
            from pathlib import Path

            self.episode_mgr = EpisodeManager(
                persist_dir=Path(data_dir) / "episodes",
                model_version=model_version,
                configuration_version="learning_v2",
            )
            self.episode_mgr.start(reason="new_run")

    def choose_action(
        self,
        obs: dict[str, Any],
        *,
        tick: int,
        goal_summary: str = "",
    ) -> str:
        typed = Observation.from_legacy_dict(obs)
        summary = self.history.summarize()
        ctx_map: dict[str, Any] = {
            "obs": obs,
            "observation": typed,
            "temporal_summary": summary,
            "stuck": bool(obs.get("stuck_hint") and obs.get("stuck_hint") != "none"),
            "goal": goal_summary,
            "tick": tick,
        }

        all_skills = list_skills() or list(DEFAULT_SKILL_ORDER)
        allowed = mask_skills(obs, all_skills)
        if not allowed:
            allowed = ["wait"]
        allowed_set = set(allowed)
        masked = [s for s in all_skills if s not in allowed_set]

        if self.cfg.policy_mode == "scripted" or self.hybrid is None:
            decision = self.scripted.choose_skill(ctx_map, allowed)
            mode = "scripted"
        else:
            decision = self.hybrid.choose_skill(ctx_map, allowed)
            mode = self.cfg.policy_mode

        skill_name = decision.skill
        self.last_skill = skill_name
        self.last_policy_mode = mode
        self.last_confidence = float(decision.confidence)
        self.last_model_version = decision.model_version
        self.last_allowed_skills = list(allowed)
        self.last_masked_skills = list(masked)
        self.last_decision_reason = (
            f"v2:{mode} skill={skill_name} conf={decision.confidence:.2f} "
            f"fallback={decision.used_fallback} | {decision.reason}"
        )

        skill_ctx = SkillContext(
            obs=obs,
            history_summary=str(summary),
            tick=tick,
            goal=goal_summary,
            now=time.monotonic(),
            meta={"policy": mode, "decision": decision.reason},
        )

        if self.runtime.is_idle() or self.runtime.active_name != skill_name:
            try:
                started = self.runtime.start(skill_name, skill_ctx)
                self._apply_skill_limits(started)
                if self.episode_mgr is not None:
                    self.episode_mgr.note_skill_attempt()
            except KeyError:
                started = self.runtime.start("wait", skill_ctx)
                self._apply_skill_limits(started)
                skill_name = "wait"

        result = self.runtime.step(skill_ctx)
        action = result.requested_action or "wait"
        ok, why = validate_action(obs, action)
        if not ok:
            # Re-mask: prefer wait / explore / tab
            for candidate in ("wait", "key:tab", "hold:w:0.6", "key:esc"):
                ok2, _ = validate_action(obs, candidate)
                if ok2:
                    action = candidate
                    self.last_decision_reason += f" | mask_reject({why})→{action}"
                    break
            else:
                action = "wait"
                self.last_decision_reason += f" | mask_reject({why})→wait"

        if self.episode_mgr is not None:
            if result.status == "success":
                self.episode_mgr.note_skill_attempt(success=True)
            elif result.status in {"failed", "timeout"}:
                self.episode_mgr.note_skill_attempt(success=False)

        return action

    def note_transition(
        self,
        prev: dict[str, Any],
        requested: str,
        executed: str,
        nxt: dict[str, Any],
        *,
        dt: float,
        legacy_reward: float | None = None,
    ) -> float:
        """Update history, events, rewards, episodes. Returns reward to log."""
        events = detect_events(prev, executed, nxt)
        if self.cfg.use_rewards_v2:
            reward_values = None
            if self.cfg.settings is not None:
                reward_values = getattr(self.cfg.settings, "rewards", None)
            breakdown = reward_from_events(events, dt=dt, values=reward_values)
            reward = float(breakdown.total)
            self.last_reward_breakdown = breakdown.to_dict()
        else:
            reward = float(legacy_reward or 0.0)
            self.last_reward_breakdown = {"total": reward, "components": {"legacy": reward}}

        outcome = "ok"
        if any(e.type.value == "DeathConfirmed" for e in events):
            outcome = "death"
        elif any(e.type.value == "KillConfirmed" for e in events):
            outcome = "kill"
        elif reward < -0.5:
            outcome = "bad"

        obs_t = Observation.from_legacy_dict(nxt)
        self.history.push(
            observation=obs_t,
            requested_action=requested,
            executed_action=executed,
            reward=reward,
            outcome=outcome,
            dt_seconds=dt,
        )

        if self.episode_mgr is not None:
            if self.episode_mgr.current is None:
                self.episode_mgr.start(reason="controllable")
            self.episode_mgr.note_reward(reward)
            # Terminal on death transition
            was_alive = (str(prev.get("life_phase") or "alive") == "alive") and not prev.get(
                "is_dead"
            )
            now_dead = bool(nxt.get("is_dead")) or str(nxt.get("life_phase") or "") in {
                "dead_dialog",
                "confirm",
                "rez_picker",
                "ghost",
            }
            if was_alive and now_dead:
                self.episode_mgr.note_death()
                self.episode_mgr.end(reason="death")
                self.episode_mgr.start(reason="resurrected")
            elif nxt.get("quest_complete") or nxt.get("goal_complete"):
                self.episode_mgr.end(reason="goal_complete")
                self.episode_mgr.start(reason="new_objective")

        return reward

    def _sensor_confidence_summary(self) -> dict[str, Any]:
        """Compact confidence map from the latest history observation."""
        if not self.history.observations:
            return {}
        obs = self.history.observations[-1]
        out: dict[str, Any] = {}
        for name in (
            "player_hp",
            "target_hp",
            "has_target",
            "in_combat",
            "motion",
            "hostile_count",
        ):
            try:
                sv = obs.sensor(name)
            except KeyError:
                continue
            out[name] = sv.confidence
        if obs.sensor_warnings:
            out["warnings"] = list(obs.sensor_warnings)[:8]
        return out

    def status_patch(self) -> dict[str, Any]:
        summary = self.history.summarize()
        snap = self.runtime.snapshot()
        ep = self.episode_mgr.current if self.episode_mgr else None
        return {
            "learning_v2": True,
            "policy_mode": self.last_policy_mode or self.cfg.policy_mode,
            "model_path": self.cfg.bc_checkpoint,
            "model_version": self.last_model_version,
            "confidence": self.last_confidence,
            "active_skill": self.runtime.active_name,
            "skill_status": snap.get("last_status"),
            "skill_snapshot": snap,
            "skill_elapsed": getattr(summary, "current_skill_duration", None),
            "allowed_skills": list(self.last_allowed_skills),
            "masked_skills": list(self.last_masked_skills),
            "decision": self.last_decision_reason,
            "reward_v2": self.last_reward_breakdown,
            "episode_id": ep.episode_id if ep else None,
            "episode_reward": float(ep.total_reward) if ep else None,
            "sensor_confidence": self._sensor_confidence_summary(),
            "temporal": summary.__dict__ if hasattr(summary, "__dict__") else str(summary),
        }
