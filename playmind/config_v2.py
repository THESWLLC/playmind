"""Versioned Learning Architecture V2 settings (Phase 16).

Safe defaults keep scripted/hybrid runnable without training data.
Validate at startup via ``LearningV2Settings.validate()``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from playmind.rewards_v2 import DEFAULT_REWARDS

POLICY_MODES = frozenset({"scripted", "hybrid", "legacy_q", "behavior_clone"})
_POLICY_MODE_ALIASES = {
    "bc": "behavior_clone",
    "behavior-clone": "behavior_clone",
    "behaviour_clone": "behavior_clone",
}
DEVICES = frozenset({"cpu", "cuda", "mps", "auto"})

# Defaults mirror skill class timeouts in playmind/skills/*.
DEFAULT_SKILL_TIMEOUTS: dict[str, float] = {
    "acquire_target": 4.0,
    "validate_target": 2.0,
    "approach_target": 5.0,
    "engage_target": 3.0,
    "basic_combat_rotation": 20.0,
    "loot_target": 4.0,
    "disengage": 4.0,
    "recover_health": 8.0,
    "explore": 6.0,
    "unstuck": 5.0,
    "clear_modal": 4.0,
    "interact": 3.0,
    "wait": 2.0,
    "death_recovery": 25.0,
    "ghost_runback": 40.0,
}

DEFAULT_SKILL_RETRY_LIMITS: dict[str, int] = {
    "acquire_target": 4,
    "validate_target": 2,
    "approach_target": 3,
    "engage_target": 3,
    "basic_combat_rotation": 3,
    "loot_target": 2,
    "disengage": 2,
    "recover_health": 3,
    "explore": 3,
    "unstuck": 4,
    "clear_modal": 4,
    "interact": 3,
    "wait": 1,
    "death_recovery": 6,
    "ghost_runback": 6,
}

DEFAULT_SENSOR_THRESHOLDS: dict[str, float] = {
    "player_hp": 0.40,
    "target_hp": 0.40,
    "has_target": 0.50,
    "in_combat": 0.50,
    "motion": 0.30,
    "hostile_count": 0.40,
    "is_dead": 0.55,
    "is_ghost": 0.55,
    "blocking_modal": 0.50,
}


@dataclass
class DemonstrationSettings:
    """Demonstration recording knobs."""

    enabled: bool = False
    root: str = "data/playmind/demonstrations"
    save_frames: bool = True
    max_session_samples: int = 10_000
    auto_label_skills: bool = False


@dataclass
class EvaluationSettings:
    """Offline evaluation / replay knobs."""

    enabled: bool = False
    report_dir: str = "data/playmind/eval"
    max_replay_samples: int = 5_000
    scenarios: list[str] = field(default_factory=list)


@dataclass
class EpisodeLimitSettings:
    """Soft episode bounds for Learning V2 episode manager."""

    max_steps: int = 10_000
    max_seconds: float = 3_600.0
    max_deaths: int = 50


@dataclass
class LearningV2Settings:
    """Full ``learning_v2`` config section with safe defaults."""

    enabled: bool = False
    policy_mode: str = "hybrid"
    legacy_q_fallback: bool = False
    history_length: int = 16
    bc_checkpoint: str | None = None
    confidence_threshold: float = 0.45
    use_rewards_v2: bool = True
    track_episodes: bool = True

    skill_timeouts: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SKILL_TIMEOUTS)
    )
    skill_retry_limits: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_SKILL_RETRY_LIMITS)
    )
    sensor_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SENSOR_THRESHOLDS)
    )
    rewards: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_REWARDS))
    episode_limits: EpisodeLimitSettings = field(default_factory=EpisodeLimitSettings)
    demonstration: DemonstrationSettings = field(default_factory=DemonstrationSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)

    device: str = "cpu"
    seed: int | None = 0

    schema_version: int = 1

    def validate(self) -> None:
        """Raise ``ValueError`` when settings are inconsistent or out of range."""
        errors: list[str] = []

        mode = str(self.policy_mode or "").strip().lower()
        if mode not in POLICY_MODES:
            errors.append(
                f"policy_mode must be one of {sorted(POLICY_MODES)}, got {self.policy_mode!r}"
            )

        if int(self.history_length) < 1 or int(self.history_length) > 512:
            errors.append(
                f"history_length must be in [1, 512], got {self.history_length!r}"
            )

        conf = float(self.confidence_threshold)
        if conf < 0.0 or conf > 1.0:
            errors.append(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold!r}"
            )

        device = str(self.device or "").strip().lower()
        if device not in DEVICES:
            errors.append(f"device must be one of {sorted(DEVICES)}, got {self.device!r}")

        if self.seed is not None:
            try:
                seed_i = int(self.seed)
            except (TypeError, ValueError):
                errors.append(f"seed must be an int or None, got {self.seed!r}")
            else:
                if seed_i < 0:
                    errors.append(f"seed must be >= 0, got {seed_i}")

        if self.bc_checkpoint is not None:
            ck = str(self.bc_checkpoint).strip()
            if not ck:
                errors.append("bc_checkpoint must be a non-empty path or null")
            elif mode == "legacy_q" and self.enabled:
                errors.append(
                    "bc_checkpoint is incompatible with policy_mode='legacy_q' "
                    "(legacy mode ignores behavior-cloning checkpoints)"
                )

        if mode == "scripted" and self.legacy_q_fallback and self.enabled:
            errors.append(
                "legacy_q_fallback=true is invalid with policy_mode='scripted' "
                "(use hybrid or legacy_q)"
            )

        for name, timeout in self.skill_timeouts.items():
            try:
                t = float(timeout)
            except (TypeError, ValueError):
                errors.append(f"skill_timeouts[{name!r}] must be a number")
                continue
            if t <= 0:
                errors.append(f"skill_timeouts[{name!r}] must be > 0, got {t}")

        for name, retries in self.skill_retry_limits.items():
            try:
                r = int(retries)
            except (TypeError, ValueError):
                errors.append(f"skill_retry_limits[{name!r}] must be an int")
                continue
            if r < 0:
                errors.append(f"skill_retry_limits[{name!r}] must be >= 0, got {r}")

        for name, thr in self.sensor_thresholds.items():
            try:
                t = float(thr)
            except (TypeError, ValueError):
                errors.append(f"sensor_thresholds[{name!r}] must be a number")
                continue
            if t < 0.0 or t > 1.0:
                errors.append(f"sensor_thresholds[{name!r}] must be in [0, 1], got {t}")

        for name, val in self.rewards.items():
            try:
                float(val)
            except (TypeError, ValueError):
                errors.append(f"rewards[{name!r}] must be a number")

        el = self.episode_limits
        if int(el.max_steps) < 1:
            errors.append(f"episode_limits.max_steps must be >= 1, got {el.max_steps}")
        if float(el.max_seconds) <= 0:
            errors.append(
                f"episode_limits.max_seconds must be > 0, got {el.max_seconds}"
            )
        if int(el.max_deaths) < 0:
            errors.append(f"episode_limits.max_deaths must be >= 0, got {el.max_deaths}")

        demo = self.demonstration
        if int(demo.max_session_samples) < 1:
            errors.append(
                "demonstration.max_session_samples must be >= 1, "
                f"got {demo.max_session_samples}"
            )
        if not str(demo.root or "").strip():
            errors.append("demonstration.root must be a non-empty path")

        ev = self.evaluation
        if int(ev.max_replay_samples) < 1:
            errors.append(
                "evaluation.max_replay_samples must be >= 1, "
                f"got {ev.max_replay_samples}"
            )
        if not str(ev.report_dir or "").strip():
            errors.append("evaluation.report_dir must be a non-empty path")

        if errors:
            raise ValueError(
                "Invalid LearningV2Settings:\n- " + "\n- ".join(errors)
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize settings to a JSON-friendly dict."""
        return asdict(self)

    @classmethod
    def load_from_owned_config(cls, owned: Mapping[str, Any] | None) -> "LearningV2Settings":
        """Load from an owned_game config dict (``learning_v2`` section).

        Missing keys keep safe defaults. Nested dicts merge over defaults.
        Does not call ``validate()`` — callers should validate at startup.
        """
        raw: dict[str, Any] = {}
        if isinstance(owned, Mapping):
            section = owned.get("learning_v2")
            if isinstance(section, Mapping):
                raw = dict(section)
            elif owned.get("enabled") is not None or owned.get("policy_mode") is not None:
                # Allow passing the learning_v2 section itself.
                raw = dict(owned)

        def _merge_float_map(defaults: dict[str, float], override: Any) -> dict[str, float]:
            out = dict(defaults)
            if isinstance(override, Mapping):
                for k, v in override.items():
                    try:
                        out[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
            return out

        def _merge_int_map(defaults: dict[str, int], override: Any) -> dict[str, int]:
            out = dict(defaults)
            if isinstance(override, Mapping):
                for k, v in override.items():
                    try:
                        out[str(k)] = int(v)
                    except (TypeError, ValueError):
                        continue
            return out

        ep_raw = raw.get("episode_limits") if isinstance(raw.get("episode_limits"), Mapping) else {}
        demo_raw = raw.get("demonstration") if isinstance(raw.get("demonstration"), Mapping) else {}
        # Alias: "demo" used in some sketches
        if not demo_raw and isinstance(raw.get("demo"), Mapping):
            demo_raw = raw.get("demo") or {}
        eval_raw = raw.get("evaluation") if isinstance(raw.get("evaluation"), Mapping) else {}
        if not eval_raw and isinstance(raw.get("eval"), Mapping):
            eval_raw = raw.get("eval") or {}

        sensor_raw = raw.get("sensor_thresholds")
        if sensor_raw is None:
            sensor_raw = raw.get("sensor_confidence_thresholds")

        rewards_raw = raw.get("rewards")
        if rewards_raw is None:
            rewards_raw = raw.get("reward_values")

        ckpt = raw.get("bc_checkpoint", raw.get("model_checkpoint"))
        if ckpt is not None:
            ckpt = str(ckpt).strip() or None

        seed_raw = raw.get("seed", 0)
        seed: int | None
        if seed_raw is None:
            seed = None
        else:
            try:
                seed = int(seed_raw)
            except (TypeError, ValueError):
                seed = 0

        mode = str(raw.get("policy_mode") or "hybrid").strip().lower()
        mode = _POLICY_MODE_ALIASES.get(mode, mode)

        settings = cls(
            enabled=bool(raw.get("enabled", False)),
            policy_mode=mode,
            legacy_q_fallback=bool(raw.get("legacy_q_fallback", False)),
            history_length=int(raw.get("history_length") or 16),
            bc_checkpoint=ckpt,
            confidence_threshold=float(
                raw.get("confidence_threshold")
                if raw.get("confidence_threshold") is not None
                else 0.45
            ),
            use_rewards_v2=bool(raw.get("use_rewards_v2", True)),
            track_episodes=bool(raw.get("track_episodes", True)),
            skill_timeouts=_merge_float_map(
                DEFAULT_SKILL_TIMEOUTS, raw.get("skill_timeouts")
            ),
            skill_retry_limits=_merge_int_map(
                DEFAULT_SKILL_RETRY_LIMITS, raw.get("skill_retry_limits")
            ),
            sensor_thresholds=_merge_float_map(
                DEFAULT_SENSOR_THRESHOLDS, sensor_raw
            ),
            rewards=_merge_float_map(dict(DEFAULT_REWARDS), rewards_raw),
            episode_limits=EpisodeLimitSettings(
                max_steps=int(ep_raw.get("max_steps") or 10_000),
                max_seconds=float(ep_raw.get("max_seconds") or 3_600.0),
                max_deaths=int(
                    ep_raw.get("max_deaths")
                    if ep_raw.get("max_deaths") is not None
                    else 50
                ),
            ),
            demonstration=DemonstrationSettings(
                enabled=bool(demo_raw.get("enabled", False)),
                root=str(demo_raw.get("root") or "data/playmind/demonstrations"),
                save_frames=bool(demo_raw.get("save_frames", True)),
                max_session_samples=int(demo_raw.get("max_session_samples") or 10_000),
                auto_label_skills=bool(demo_raw.get("auto_label_skills", False)),
            ),
            evaluation=EvaluationSettings(
                enabled=bool(eval_raw.get("enabled", False)),
                report_dir=str(eval_raw.get("report_dir") or "data/playmind/eval"),
                max_replay_samples=int(eval_raw.get("max_replay_samples") or 5_000),
                scenarios=[str(s) for s in (eval_raw.get("scenarios") or [])],
            ),
            device=str(raw.get("device") or "cpu").strip().lower(),
            seed=seed,
            schema_version=int(raw.get("schema_version") or 1),
        )
        return settings

    def resolved_checkpoint_path(self) -> Path | None:
        if not self.bc_checkpoint:
            return None
        return Path(self.bc_checkpoint)


def safe_defaults() -> LearningV2Settings:
    """Return a validated copy of safe Learning V2 defaults."""
    settings = LearningV2Settings()
    settings.validate()
    return deepcopy(settings)
