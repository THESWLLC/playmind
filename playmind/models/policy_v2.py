"""Skill-classifier policy for Learning Architecture V2.

Importing this module does **not** require PyTorch. When ``torch`` is
available, an MLP ``SkillPolicyNet`` is defined for behavior-cloning on
structured Observation / TemporalSummary features (no image dependency).

Until a checkpoint is trained, ``predict`` returns a low-confidence skill
with zero aux heads so HybridPolicy falls back to scripted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch missing
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


DEFAULT_SKILLS = (
    "death_recovery",
    "ghost_runback",
    "clear_modal",
    "unstuck",
    "recover_health",
    "disengage",
    "acquire_target",
    "validate_target",
    "approach_target",
    "engage_target",
    "basic_combat_rotation",
    "loot_target",
    "explore",
    "interact",
    "wait",
)

AUX_KEYS = ("target_valid", "combat", "death")

# Low confidence so HybridPolicy (< threshold ~0.45) falls back to scripted.
UNTRAINED_CONFIDENCE = 0.05
MODEL_VERSION = "skill-policy-v2-mlp-1"

# life_phase one-hot order for stable feature layout
_LIFE_PHASES = (
    "alive",
    "dead_dialog",
    "confirm",
    "rez_picker",
    "ghost",
    "loading",
    "unknown",
)

# Fixed layout sizes (must stay in sync with structured_feature_vector).
_OBS_CORE_DIM = 20  # scalars before life-phase one-hot
_SUMMARY_DIM = 11
_STRUCTURED_FEATURE_DIM = _OBS_CORE_DIM + len(_LIFE_PHASES) + _SUMMARY_DIM


def _opt_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _opt_bool01(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return 1.0 if bool(value) else 0.0


def _as_observation(obj: Any) -> Any:
    """Return Observation if possible; else None."""
    if obj is None:
        return None
    try:
        from playmind.observations import Observation
    except ImportError:  # pragma: no cover
        return None
    if isinstance(obj, Observation):
        return obj
    if isinstance(obj, Mapping):
        return Observation.from_legacy_dict(dict(obj))
    return None


def _as_temporal_summary(obj: Any) -> Any:
    if obj is None:
        return None
    try:
        from playmind.history import TemporalSummary
    except ImportError:  # pragma: no cover
        return None
    if isinstance(obj, TemporalSummary):
        return obj
    if isinstance(obj, Mapping):
        known = {f.name for f in TemporalSummary.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: obj[k] for k in known if k in obj}
        return TemporalSummary(**kwargs)
    return None


def structured_feature_vector(
    observation: Any = None,
    temporal_summary: Any = None,
    *,
    feature_dim: int | None = None,
) -> list[float]:
    """Build a fixed-layout numeric feature vector from Observation + TemporalSummary.

    Accepts Observation / TemporalSummary instances, legacy dicts, or a context
    mapping with ``obs`` / ``observation`` / ``temporal_summary`` / ``summary`` keys.
    """
    obs_in = observation
    summary_in = temporal_summary

    if isinstance(observation, Mapping) and temporal_summary is None:
        # Context-style dict: pull nested fields if present.
        if "obs" in observation or "observation" in observation or "temporal_summary" in observation:
            ctx = observation
            obs_in = ctx.get("observation", ctx.get("obs", ctx))
            summary_in = ctx.get("temporal_summary", ctx.get("summary"))

    obs = _as_observation(obs_in)
    summary = _as_temporal_summary(summary_in)

    # Observation block
    if obs is not None:
        phase = str(getattr(obs, "life_phase", None) or "unknown")
        phase_oh = [1.0 if phase == p else 0.0 for p in _LIFE_PHASES]
        obs_feats = [
            _opt_float(obs.player_hp, 0.5),
            _opt_float(obs.player_hp_confidence, 0.0),
            _opt_float(obs.target_hp, 0.0),
            _opt_float(obs.target_hp_confidence, 0.0),
            _opt_bool01(obs.has_target),
            _opt_float(obs.has_target_confidence, 0.0),
            _opt_bool01(obs.in_combat),
            _opt_float(obs.in_combat_confidence, 0.0),
            _opt_bool01(obs.is_dead),
            _opt_bool01(obs.is_ghost),
            _opt_float(obs.motion, 0.0),
            _opt_float(obs.motion_confidence, 0.0),
            _opt_bool01(obs.hostiles_near),
            _opt_float(obs.hostile_count, 0.0),
            _opt_float(obs.hostile_count_confidence, 0.0),
            _opt_bool01(obs.blocking_modal),
            _opt_float(obs.objective_progress, 0.0),
            float(obs.stagnation_count or 0),
            float(obs.failed_action_streak or 0),
            float(len(obs.sensor_warnings or [])),
            *phase_oh,
        ]
    elif isinstance(obs_in, Mapping):
        # Fallback compact dict path (matches dataset._feature_vector spirit).
        d = dict(obs_in)
        player = d.get("player") if isinstance(d.get("player"), dict) else {}
        hp = d.get("vision_player_hp", d.get("player_hp", player.get("hp") if player else None))
        phase = str(d.get("life_phase") or "unknown")
        phase_oh = [1.0 if phase == p else 0.0 for p in _LIFE_PHASES]
        obs_feats = [
            _opt_float(hp, 0.5),
            _opt_float(d.get("player_hp_confidence"), 0.0),
            _opt_float(d.get("target_hp", d.get("target_hp_est")), 0.0),
            _opt_float(d.get("target_hp_confidence"), 0.0),
            _opt_bool01(d.get("has_target")),
            _opt_float(d.get("has_target_confidence"), 0.0),
            _opt_bool01(d.get("in_combat")),
            _opt_float(d.get("in_combat_confidence"), 0.0),
            _opt_bool01(d.get("is_dead")),
            _opt_bool01(d.get("is_ghost")),
            _opt_float(d.get("motion"), 0.0),
            _opt_float(d.get("motion_confidence"), 0.0),
            _opt_bool01(d.get("hostiles_near")),
            _opt_float(d.get("hostile_count"), 0.0),
            _opt_float(d.get("hostile_count_confidence"), 0.0),
            _opt_bool01(d.get("blocking_modal", d.get("modal_menu"))),
            _opt_float(d.get("objective_progress"), 0.0),
            _opt_float(d.get("stagnation_count"), 0.0),
            _opt_float(d.get("failed_action_streak"), 0.0),
            float(len(d.get("sensor_warnings") or [])),
            *phase_oh,
        ]
    else:
        # Unknown observation → neutral defaults matching Observation layout.
        obs_feats = [0.5] + [0.0] * (_OBS_CORE_DIM - 1) + [0.0] * len(_LIFE_PHASES)
        # Mark "unknown" life phase.
        obs_feats[_OBS_CORE_DIM + _LIFE_PHASES.index("unknown")] = 1.0

    # Temporal summary block
    if summary is not None:
        summary_feats = [
            float(summary.health_trend),
            float(summary.target_health_trend),
            float(summary.motion_trend),
            float(summary.repeated_action_count),
            float(summary.no_progress_duration),
            float(summary.target_flicker_count),
            float(summary.combat_flicker_count),
            float(summary.recent_damage_dealt_est),
            float(summary.recent_damage_received_est),
            float(summary.current_skill_duration),
            float(summary.recent_sensor_disagreement),
        ]
    else:
        summary_feats = [0.0] * _SUMMARY_DIM

    assert len(obs_feats) == _OBS_CORE_DIM + len(_LIFE_PHASES), len(obs_feats)
    assert len(summary_feats) == _SUMMARY_DIM, len(summary_feats)
    vec = obs_feats + summary_feats
    if feature_dim is not None:
        dim = int(feature_dim)
        if len(vec) < dim:
            vec = vec + [0.0] * (dim - len(vec))
        else:
            vec = vec[:dim]
    return vec


DEFAULT_FEATURE_DIM = _STRUCTURED_FEATURE_DIM


def _pad_or_truncate(vec: Sequence[float], dim: int) -> list[float]:
    out = [float(x) for x in vec]
    if len(out) < dim:
        out = out + [0.0] * (dim - len(out))
    else:
        out = out[:dim]
    return out


def _as_feature_list(features: Any, *, feature_dim: int) -> list[float]:
    """Coerce various feature inputs into a padded float vector."""
    if features is None:
        return [0.0] * feature_dim
    # Observation / TemporalSummary / context → structured vector
    try:
        from playmind.observations import Observation
        from playmind.history import TemporalSummary
    except ImportError:  # pragma: no cover
        Observation = None  # type: ignore[assignment,misc]
        TemporalSummary = None  # type: ignore[assignment,misc]

    if Observation is not None and isinstance(features, Observation):
        return structured_feature_vector(features, feature_dim=feature_dim)
    if TemporalSummary is not None and isinstance(features, TemporalSummary):
        return structured_feature_vector(None, features, feature_dim=feature_dim)
    if isinstance(features, Mapping):
        if any(
            k in features
            for k in ("obs", "observation", "temporal_summary", "player_hp", "has_target", "vision_player_hp")
        ):
            return structured_feature_vector(features, feature_dim=feature_dim)
        # Sorted numeric dict fallback
        vals: list[float] = []
        for k in sorted(features.keys()):
            v = features[k]
            if isinstance(v, bool):
                vals.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                vals.append(float(v))
            elif v is None:
                vals.append(0.0)
        return _pad_or_truncate(vals, feature_dim)
    if isinstance(features, (list, tuple)):
        out: list[float] = []
        for v in features:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(0.0)
        return _pad_or_truncate(out, feature_dim)
    if TORCH_AVAILABLE and torch is not None and isinstance(features, torch.Tensor):
        return _pad_or_truncate(
            [float(x) for x in features.detach().cpu().flatten().tolist()],
            feature_dim,
        )
    try:
        return _pad_or_truncate([float(features)], feature_dim)
    except (TypeError, ValueError):
        return [0.0] * feature_dim


if TORCH_AVAILABLE:

    class SkillPolicyNet(nn.Module):  # type: ignore[misc]
        """MLP skill classifier on structured features (no image / CNN dependency).

        Outputs skill logits plus three aux logits: target_valid, combat, death.
        """

        def __init__(
            self,
            n_skills: int,
            feature_dim: int = DEFAULT_FEATURE_DIM,
            hidden: int = 64,
            n_aux: int = len(AUX_KEYS),
        ) -> None:
            super().__init__()
            self.feature_dim = int(feature_dim)
            self.n_skills = int(n_skills)
            self.n_aux = int(n_aux)
            self.backbone = nn.Sequential(
                nn.Linear(self.feature_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.skill_head = nn.Linear(hidden, self.n_skills)
            self.aux_head = nn.Linear(hidden, self.n_aux)

        def forward(self, features: Any) -> tuple[Any, Any]:
            # features: (B, F)
            if features.dim() == 3:
                # (B, T, F) → use last timestep
                features = features[:, -1, :]
            h = self.backbone(features)
            return self.skill_head(h), self.aux_head(h)

    # Back-compat alias for older scaffold name.
    SkillClassifierNet = SkillPolicyNet

else:
    SkillPolicyNet = None  # type: ignore[misc, assignment]
    SkillClassifierNet = None  # type: ignore[misc, assignment]


def _zero_aux() -> dict[str, float]:
    return {k: 0.0 for k in AUX_KEYS}


class SkillPolicyV2:
    """Skill classifier with structured features, JSON+weights checkpointing.

    When untrained, ``predict`` returns ``wait`` (or first skill) with low
    confidence and zero aux outputs.
    """

    model_version: str = MODEL_VERSION

    def __init__(
        self,
        skill_names: Sequence[str] | None = None,
        *,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        hidden: int = 64,
        trained: bool = False,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        names = list(skill_names) if skill_names is not None else list(DEFAULT_SKILLS)
        if not names:
            names = ["wait"]
        self.skill_names = names
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.trained = bool(trained)
        self.config: dict[str, Any] = {
            "model_version": self.model_version,
            "skill_names": list(self.skill_names),
            "feature_dim": self.feature_dim,
            "hidden": self.hidden,
            "aux_keys": list(AUX_KEYS),
            "arch": "mlp",
            "torch_available": TORCH_AVAILABLE,
            **(dict(config) if config else {}),
        }
        self.metadata: dict[str, Any] = {
            "model_version": self.model_version,
            "skill_names": list(self.skill_names),
            "feature_dim": self.feature_dim,
            "trained": self.trained,
            "torch_available": TORCH_AVAILABLE,
            "config": dict(self.config),
            "note": "MLP on structured Observation/TemporalSummary features; torch optional.",
        }
        self._net: Any = None
        if TORCH_AVAILABLE and SkillPolicyNet is not None:
            self._net = SkillPolicyNet(
                len(self.skill_names),
                feature_dim=self.feature_dim,
                hidden=self.hidden,
            )

    def features_from(
        self,
        observation: Any = None,
        temporal_summary: Any = None,
    ) -> list[float]:
        return structured_feature_vector(
            observation, temporal_summary, feature_dim=self.feature_dim
        )

    def predict(
        self,
        features: Any = None,
        *,
        observation: Any = None,
        temporal_summary: Any = None,
    ) -> tuple[str, float, dict[str, float]]:
        """Return ``(skill, confidence, aux)``.

        ``aux`` always contains ``target_valid``, ``combat``, ``death`` (zeros if
        untrained). Features may be a vector, Observation, context mapping, or
        passed via ``observation`` / ``temporal_summary``.
        """
        n = len(self.skill_names)
        untrained_skill = "wait" if "wait" in self.skill_names else self.skill_names[0]
        untrained_conf = UNTRAINED_CONFIDENCE if n else 0.0
        if n > 1:
            untrained_conf = min(UNTRAINED_CONFIDENCE, 1.0 / float(n))

        if features is None and (observation is not None or temporal_summary is not None):
            vec = self.features_from(observation, temporal_summary)
        else:
            vec = _as_feature_list(features, feature_dim=self.feature_dim)

        if not self.trained or self._net is None or not TORCH_AVAILABLE:
            return untrained_skill, float(untrained_conf), _zero_aux()

        assert torch is not None
        self._net.eval()
        with torch.no_grad():
            t = torch.tensor([vec], dtype=torch.float32)
            skill_logits, aux_logits = self._net(t)
            probs = torch.softmax(skill_logits[0], dim=-1)
            idx = int(torch.argmax(probs).item())
            conf = float(probs[idx].item())
            aux_probs = torch.sigmoid(aux_logits[0])
            aux = {
                AUX_KEYS[i]: float(aux_probs[i].item()) if i < aux_probs.numel() else 0.0
                for i in range(len(AUX_KEYS))
            }
        return self.skill_names[idx], conf, aux

    def predict_skill(self, features: Any) -> tuple[str, float]:
        """Back-compat: return ``(skill, confidence)`` only."""
        skill, conf, _aux = self.predict(features)
        return skill, conf

    def choose_skill(
        self,
        context: Mapping[str, Any],
        allowed_skills: Sequence[str],
    ) -> Any:
        """HighLevelPolicy-compatible wrapper around ``predict``."""
        from playmind.policies.base import PolicyDecision

        feats = context.get("features")
        obs = context.get("observation", context.get("obs"))
        summary = context.get("temporal_summary", context.get("summary"))
        if feats is None:
            skill, conf, aux = self.predict(observation=obs or context, temporal_summary=summary)
        else:
            skill, conf, aux = self.predict(feats, observation=obs, temporal_summary=summary)

        allowed = list(dict.fromkeys(str(s) for s in allowed_skills))
        if skill not in set(allowed):
            skill = "wait" if "wait" in allowed else (allowed[0] if allowed else "wait")
            conf = min(conf, UNTRAINED_CONFIDENCE)
        debug = {f"aux_{k}": float(v) for k, v in aux.items()}
        debug["trained"] = 1.0 if self.trained else 0.0
        return PolicyDecision(
            skill=skill,
            confidence=conf,
            reason="SkillPolicyV2 stub" if not self.trained else "SkillPolicyV2",
            model_version=self.model_version,
            allowed_skills=allowed,
            used_fallback=not self.trained,
            temporal_summary=str(summary) if summary is not None else None,
            debug_scores=debug,
        )

    def save(self, path: str | Path, *, config_snapshot: Mapping[str, Any] | None = None) -> Path:
        """Save checkpoint metadata JSON (+ torch weights when available).

        Metadata always includes ``model_version`` and a ``config`` snapshot.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = dict(self.config)
        if config_snapshot:
            cfg.update(dict(config_snapshot))
        cfg["model_version"] = self.model_version
        cfg["skill_names"] = list(self.skill_names)
        cfg["feature_dim"] = self.feature_dim
        cfg["hidden"] = self.hidden
        cfg["trained"] = self.trained
        cfg["torch_available"] = TORCH_AVAILABLE
        cfg["arch"] = "mlp"
        cfg["aux_keys"] = list(AUX_KEYS)

        meta: dict[str, Any] = {
            "model_version": self.model_version,
            "skill_names": list(self.skill_names),
            "feature_dim": self.feature_dim,
            "hidden": self.hidden,
            "trained": self.trained,
            "torch_available": TORCH_AVAILABLE,
            "config": cfg,
            "note": "MLP on structured Observation/TemporalSummary features; torch optional.",
        }
        meta_path = path if path.suffix == ".json" else path.with_suffix(".json")
        weights_name = meta_path.with_suffix(".pt").name
        meta["weights_file"] = weights_name if (TORCH_AVAILABLE and self._net is not None) else None

        tmp = meta_path.with_suffix(meta_path.suffix + f".{os.getpid()}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, meta_path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

        if TORCH_AVAILABLE and self._net is not None and torch is not None:
            weights_path = meta_path.with_suffix(".pt")
            torch.save(
                {
                    "state_dict": self._net.state_dict(),
                    "metadata": meta,
                    "model_version": self.model_version,
                    "config": cfg,
                },
                weights_path,
            )

        self.metadata = meta
        self.config = cfg
        return meta_path

    @classmethod
    def load(cls, path: str | Path) -> "SkillPolicyV2":
        """Load from checkpoint metadata JSON (optional sibling ``.pt``)."""
        path = Path(path)
        meta_path = path if path.suffix == ".json" else path.with_suffix(".json")
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = dict(meta.get("config") or {})
        obj = cls(
            skill_names=meta.get("skill_names") or cfg.get("skill_names") or list(DEFAULT_SKILLS),
            feature_dim=int(meta.get("feature_dim") or cfg.get("feature_dim") or DEFAULT_FEATURE_DIM),
            hidden=int(meta.get("hidden") or cfg.get("hidden") or 64),
            trained=bool(meta.get("trained", False)),
            config=cfg,
        )
        obj.metadata = dict(meta)
        if meta.get("model_version"):
            obj.model_version = str(meta["model_version"])
        weights_file = meta.get("weights_file")
        weights_path = meta_path.with_suffix(".pt")
        if weights_file:
            candidate = meta_path.parent / weights_file
            if candidate.exists():
                weights_path = candidate
        if TORCH_AVAILABLE and obj._net is not None and torch is not None and weights_path.exists():
            blob = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = blob.get("state_dict") if isinstance(blob, dict) else None
            if state is not None:
                obj._net.load_state_dict(state)
        return obj


def torch_install_instructions() -> str:
    return (
        "PyTorch is not installed. Behavior-cloning training needs torch for the MLP.\n"
        "Install CPU torch, then re-run:\n"
        "  pip install torch\n"
        "Docs: https://pytorch.org/get-started/locally/\n"
        "Without torch you can still dry-validate demonstration datasets."
    )
