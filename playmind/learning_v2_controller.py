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
from playmind.life_episode import EpisodeLifecycleController, LifecycleState
from playmind.models.feature_schema import structured_feature_vector_v2
from playmind.observations import Observation
from playmind.policies.base import PolicyDecision
from playmind.policies.hybrid import BehaviorCloningPolicy, HybridPolicy
from playmind.policies.legacy_q import LegacyQPolicy
from playmind.policies.scripted import DEFAULT_SKILL_ORDER, ScriptedPolicy
from playmind.rewards_v2 import reward_from_events
from playmind.skill_commitment import (
    CRITICAL_INTERRUPT_REASONS,
    SkillCommitmentTracker,
)
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
    commitment_confidence_margin: float = 0.15
    minimum_commitment_seconds: float = 0.4
    maximum_commitment_seconds: float = 25.0
    controllable_frames: int = 3
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
            commitment_confidence_margin=float(
                getattr(settings, "commitment_confidence_margin", 0.15)
            ),
            minimum_commitment_seconds=float(
                getattr(settings, "minimum_commitment_seconds", 0.4)
            ),
            maximum_commitment_seconds=float(
                getattr(settings, "maximum_commitment_seconds", 25.0)
            ),
            controllable_frames=int(getattr(settings, "controllable_frames", 3)),
            settings=settings,
        )


@dataclass
class LearningV2Controller:
    cfg: LearningV2Config
    history: TemporalHistory = field(init=False)
    runtime: SkillRuntime = field(default_factory=SkillRuntime)
    episode_mgr: EpisodeManager | None = None
    lifecycle: EpisodeLifecycleController | None = None
    hybrid: HybridPolicy | None = None
    scripted: ScriptedPolicy = field(default_factory=ScriptedPolicy)
    commitment: SkillCommitmentTracker = field(init=False)
    last_decision_reason: str = ""
    last_skill: str | None = None
    last_policy_mode: str = ""
    last_confidence: float = 0.0
    last_model_version: str | None = None
    last_allowed_skills: list[str] = field(default_factory=list)
    last_masked_skills: list[str] = field(default_factory=list)
    last_reward_breakdown: dict[str, Any] = field(default_factory=dict)
    last_lifecycle_status: dict[str, Any] = field(default_factory=dict)
    _legacy_policy: Any = None
    _last_runtime_result: Any = None
    _skill_outcome_recorded: bool = True
    _next_policy_query_at: float = 0.0
    _shutdown: bool = False
    _planner_skill: str | None = None

    def queue_planner_skill(self, skill_name: str) -> None:
        """Queue one validated Planner V2 skill for the next policy boundary."""
        self._planner_skill = str(skill_name)
        self._next_policy_query_at = 0.0

    def __post_init__(self) -> None:
        self.history = TemporalHistory(maxlen=max(4, int(self.cfg.history_length)))
        self.commitment = SkillCommitmentTracker(
            confidence_margin=self.cfg.commitment_confidence_margin,
            minimum_commitment_seconds=self.cfg.minimum_commitment_seconds,
            maximum_commitment_seconds=self.cfg.maximum_commitment_seconds,
        )
        # The controller owns every interrupt so commitment accounting cannot be
        # bypassed by SkillRuntime's legacy death auto-switch.
        self.runtime.interrupt_on_death = False

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
        if self.lifecycle is None:
            max_duration = None
            if self.cfg.settings is not None:
                limits = getattr(self.cfg.settings, "episode_limits", None)
                max_duration = getattr(limits, "max_seconds", None)
            self.lifecycle = EpisodeLifecycleController(
                self.episode_mgr,
                frames_alive_controllable=max(1, int(self.cfg.controllable_frames)),
                max_duration_s=float(max_duration) if max_duration is not None else None,
            )

    @staticmethod
    def _emergency_state(obs: dict[str, Any]) -> dict[str, bool]:
        phase = str(obs.get("life_phase") or "").strip().lower()
        try:
            hp = float(
                obs.get("vision_player_hp")
                if obs.get("vision_player_hp") is not None
                else (obs.get("player") or {}).get("hp", 1.0)
            )
        except (TypeError, ValueError, AttributeError):
            hp = 1.0
        stuck_hint = str(obs.get("stuck_hint") or "").strip().lower()
        emergency = {
            "death_confirmed": bool(obs.get("is_dead"))
            or phase in {"dead", "dead_dialog", "confirm", "rez_picker"},
            "ghost": bool(obs.get("is_ghost")) or phase in {"ghost", "runback"},
            "blocking_modal": bool(obs.get("blocking_modal") or obs.get("modal_menu")),
            "critical_health": hp <= 0.15
            and phase not in {"dead", "dead_dialog", "confirm", "rez_picker", "ghost"},
            "severe_stuck": bool(obs.get("severe_stuck"))
            or (
                stuck_hint not in {"", "none"}
                and int(obs.get("stagnant") or obs.get("stagnation_count") or 0) >= 8
            ),
            "loading": bool(obs.get("loading") or obs.get("is_loading"))
            or phase == "loading",
            "lost_focus": any(
                key in obs and obs.get(key) is False
                for key in ("focused", "has_focus", "window_focused")
            ),
            "fatal_sensor_disagreement": bool(
                obs.get("fatal_sensor_disagreement") or obs.get("fatal_sensor")
            ),
        }
        return {reason: active for reason, active in emergency.items() if active}

    def _feature_sequence(
        self,
        current: Observation,
        summary: Any,
    ) -> list[list[float]]:
        observations = [*self.history.observations, current]
        return [
            structured_feature_vector_v2(observation, summary)
            for observation in observations[-max(1, int(self.cfg.history_length)) :]
        ]

    def _choose_policy_skill(
        self,
        context: dict[str, Any],
        allowed: list[str],
        *,
        emergency: bool = False,
    ) -> tuple[Any, str]:
        if emergency or self.cfg.policy_mode == "scripted" or self.hybrid is None:
            mode = "scripted_emergency" if emergency else "scripted"
            return self.scripted.choose_skill(context, allowed), mode
        return self.hybrid.choose_skill(context, allowed), self.cfg.policy_mode

    def _note_skill_outcome(self, status: str) -> None:
        if self._skill_outcome_recorded or status not in {
            "success",
            "failed",
            "timeout",
            "cancelled",
        }:
            return
        if self.episode_mgr is not None:
            self.episode_mgr.note_skill_outcome(status == "success")
        self._skill_outcome_recorded = True

    def _start_skill(
        self,
        skill_name: str,
        skill_ctx: SkillContext,
        *,
        tick: int,
        confidence: float,
        reason: str,
    ) -> str:
        try:
            started = self.runtime.start(skill_name, skill_ctx)
        except KeyError:
            skill_name = "wait"
            started = self.runtime.start(skill_name, skill_ctx)
        self._apply_skill_limits(started)
        if self.episode_mgr is not None:
            self.episode_mgr.note_skill_start()
        self._skill_outcome_recorded = False
        self.commitment.begin_commitment(
            skill_name,
            now=skill_ctx.now,
            tick=tick,
            policy_confidence=confidence,
            decision_reason=reason,
            interruptible=True,
            interrupt_reasons=set(CRITICAL_INTERRUPT_REASONS),
        )
        # A rejected proposal is not polled again on every render tick.
        self._next_policy_query_at = skill_ctx.now + max(
            0.25, float(self.cfg.minimum_commitment_seconds)
        )
        return skill_name

    def choose_action(
        self,
        obs: dict[str, Any],
        *,
        tick: int,
        goal_summary: str = "",
    ) -> str:
        self._shutdown = False
        typed = Observation.from_legacy_dict(obs)
        summary = self.history.summarize()
        now = time.monotonic()
        ctx_map: dict[str, Any] = {
            "obs": obs,
            "observation": typed,
            "temporal_summary": summary,
            "history": self.history,
            "feature_sequence": self._feature_sequence(typed, summary),
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
        self.last_allowed_skills = list(allowed)
        self.last_masked_skills = list(masked)
        skill_ctx = SkillContext(
            obs=obs,
            history_summary=str(summary),
            tick=tick,
            goal=goal_summary,
            now=now,
            meta={"allowed_skills": allowed},
        )

        active = self.commitment.active
        runtime_result = self.runtime.last_result or {"status": self.runtime.status}
        emergency_state = self._emergency_state(obs)
        reconsider = self.commitment.should_reconsider(
            runtime_result,
            {
                "obs": obs,
                "now": now,
                "tick": tick,
                "allowed_skills": allowed,
            },
            emergency_state,
        )
        query_policy = active is None or reconsider.reconsider
        if (
            not query_policy
            and active is not None
            and now >= self._next_policy_query_at
            and now - active.started_at >= active.minimum_commitment_seconds
        ):
            query_policy = True

        if query_policy:
            planned = self._planner_skill
            self._planner_skill = None
            if planned in allowed and not reconsider.force_interrupt:
                decision = PolicyDecision(
                    skill=str(planned),
                    confidence=1.0,
                    reason="validated Planner V2 queue",
                    model_version="planner_v2",
                    allowed_skills=list(allowed),
                )
                mode = "planner_v2"
            else:
                decision, mode = self._choose_policy_skill(
                    ctx_map,
                    allowed,
                    emergency=reconsider.force_interrupt,
                )
            final_reconsider = reconsider
            if active is not None and not reconsider.force_interrupt:
                final_reconsider = self.commitment.should_reconsider(
                    runtime_result,
                    {
                        "obs": obs,
                        "now": now,
                        "tick": tick,
                        "allowed_skills": allowed,
                    },
                    None,
                    proposed_skill=decision.skill,
                    policy_confidence=float(decision.confidence),
                )
            self.last_policy_mode = mode
            self.last_confidence = float(decision.confidence)
            self.last_model_version = decision.model_version
            base_reason = (
                f"v2:{mode} skill={decision.skill} conf={decision.confidence:.2f} "
                f"fallback={decision.used_fallback} | {decision.reason}"
            )

            can_change = active is None or final_reconsider.reconsider
            active_running = (
                self.runtime.active is not None and not self.runtime.is_idle()
            )
            if can_change and (
                not active_running or self.runtime.active_name != decision.skill
            ):
                if active_running:
                    self.runtime.cancel(skill_ctx)
                    self._note_skill_outcome("cancelled")
                skill_name = self._start_skill(
                    decision.skill,
                    skill_ctx,
                    tick=tick,
                    confidence=float(decision.confidence),
                    reason=decision.reason,
                )
                self.last_decision_reason = (
                    f"{base_reason} | commitment:{final_reconsider.reason}"
                )
            elif can_change and active is not None:
                # Policy reaffirmed the same running skill at a reconsideration
                # boundary. Renew the gate without restarting the runtime.
                self.commitment.begin_commitment(
                    active.skill_name,
                    now=now,
                    tick=tick,
                    policy_confidence=float(decision.confidence),
                    decision_reason=decision.reason,
                    interruptible=active.interruptible,
                    interrupt_reasons=set(active.interrupt_reasons),
                )
                self._next_policy_query_at = now + max(
                    0.25, float(self.cfg.minimum_commitment_seconds)
                )
                skill_name = active.skill_name
                self.last_decision_reason = (
                    f"{base_reason} | commitment_reaffirmed:{final_reconsider.reason}"
                )
            else:
                skill_name = active.skill_name if active is not None else decision.skill
                self._next_policy_query_at = now + max(
                    0.25, float(self.cfg.minimum_commitment_seconds)
                )
                self.last_decision_reason = (
                    f"{base_reason} | commitment_hold:{final_reconsider.reason}"
                )
        else:
            mode = self.last_policy_mode or self.cfg.policy_mode
            skill_name = active.skill_name if active is not None else "wait"
            self.last_decision_reason = (
                f"v2:{mode} skill={skill_name} | commitment_hold:{reconsider.reason}"
            )

        self.last_skill = skill_name

        result = self.runtime.step(skill_ctx)
        self._last_runtime_result = result
        self._note_skill_outcome(result.status)
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
        event_nxt = dict(nxt)
        result = self._last_runtime_result
        if result is not None:
            status = str(getattr(result, "status", ""))
            event_nxt.update(
                {
                    "skill_name": self.last_skill,
                    "skill_succeeded": status == "success",
                    "skill_failed": status == "failed",
                    "skill_timeout": status == "timeout",
                    "skill_fail_reason": getattr(result, "reason", status),
                }
            )
        events = detect_events(prev, executed, event_nxt)
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

        obs_t = Observation.from_legacy_dict(event_nxt)
        self.history.push(
            observation=obs_t,
            requested_action=requested,
            executed_action=executed,
            reward=reward,
            outcome=outcome,
            dt_seconds=dt,
        )

        if self.episode_mgr is not None:
            # Attribute the transition reward to the segment that produced it,
            # then let lifecycle close/open boundaries.
            self.episode_mgr.note_reward(reward)
            if self.lifecycle is not None:
                self.last_lifecycle_status = self.lifecycle.update(
                    event_nxt, events, time.monotonic()
                )

        return reward

    def shutdown(self) -> None:
        """Close an open gameplay or recovery episode exactly once."""
        if self._shutdown:
            return
        self._shutdown = True
        if self.lifecycle is not None:
            self.last_lifecycle_status = self.lifecycle.update(
                {"session_end": True, "life_phase": self.lifecycle.state.value},
                [],
                time.monotonic(),
            )
        elif (
            self.episode_mgr is not None
            and self.episode_mgr.current is not None
            and not self.episode_mgr.current.done
        ):
            self.episode_mgr.end("session_end")

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
        if ep is not None and ep.done:
            ep = None
        lifecycle_state = (
            self.lifecycle.state.value
            if self.lifecycle is not None
            else LifecycleState.UNKNOWN.value
        )
        commitment_stats = self.commitment.stats()
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
            "episode_kind": ep.episode_kind if ep else None,
            "lifecycle_state": lifecycle_state,
            "lifecycle": dict(self.last_lifecycle_status),
            "commitment": commitment_stats,
            "commitment_stats": commitment_stats,
            "sensor_confidence": self._sensor_confidence_summary(),
            "temporal": summary.__dict__ if hasattr(summary, "__dict__") else str(summary),
        }
