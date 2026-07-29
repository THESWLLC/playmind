"""Feature schema V2: value / known / confidence structured features.

Unknown must not collapse to the same floats as known false / zero.
Legacy schema v1 (38-D MLP layout) remains loadable via adapters only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

FEATURE_SCHEMA_VERSION = 2
LEGACY_FEATURE_SCHEMA_VERSION = 1

# Life-phase one-hot (stable order).
LIFE_PHASES: tuple[str, ...] = (
    "alive",
    "dead_dialog",
    "confirm",
    "rez_picker",
    "ghost",
    "loading",
    "unknown",
)

# Sensors that use the triple pattern: value, known, confidence.
TRIPLE_SENSORS: tuple[str, ...] = (
    "player_hp",
    "target_hp",
    "has_target",
    "in_combat",
    "is_dead",
    "is_ghost",
    "motion",
    "hostiles_near",
    "hostile_count",
    "blocking_modal",
    "objective_progress",
)

# Scalars that are always "known" counters (clamped).
ALWAYS_KNOWN_SCALARS: tuple[str, ...] = (
    "stagnation_count",
    "failed_action_streak",
    "sensor_warning_count",
)

# Temporal summary fields (from TemporalHistory.summarize).
TEMPORAL_FIELDS: tuple[str, ...] = (
    "health_trend",
    "target_health_trend",
    "motion_trend",
    "repeated_action_count",
    "no_progress_duration",
    "target_flicker_count",
    "combat_flicker_count",
    "recent_damage_dealt_est",
    "recent_damage_received_est",
    "current_skill_duration",
    "recent_sensor_disagreement",
)

# Features that should NOT be z-normalized (masks / one-hots / known flags).
NON_NORMALIZED_SUFFIXES: tuple[str, ...] = ("_known",)
NON_NORMALIZED_PREFIXES: tuple[str, ...] = ("life_phase_",)


def _ordered_feature_names() -> list[str]:
    names: list[str] = []
    for sensor in TRIPLE_SENSORS:
        names.extend([f"{sensor}_value", f"{sensor}_known", f"{sensor}_confidence"])
    names.extend(ALWAYS_KNOWN_SCALARS)
    for phase in LIFE_PHASES:
        names.append(f"life_phase_{phase}")
    names.extend(TEMPORAL_FIELDS)
    return names


FEATURE_NAMES: tuple[str, ...] = tuple(_ordered_feature_names())
FEATURE_DIM: int = len(FEATURE_NAMES)

# Boolean-like triples use 0/1 value when known.
_BOOL_SENSORS = frozenset(
    {
        "has_target",
        "in_combat",
        "is_dead",
        "is_ghost",
        "hostiles_near",
        "blocking_modal",
    }
)

# Clamp ranges for extreme scalars (after optional log1p for counts).
_CLAMP: dict[str, tuple[float, float]] = {
    "stagnation_count": (0.0, 100.0),
    "failed_action_streak": (0.0, 100.0),
    "sensor_warning_count": (0.0, 50.0),
    "repeated_action_count": (0.0, 64.0),
    "no_progress_duration": (0.0, 120.0),
    "target_flicker_count": (0.0, 64.0),
    "combat_flicker_count": (0.0, 64.0),
    "recent_damage_dealt_est": (-2.0, 2.0),
    "recent_damage_received_est": (-2.0, 2.0),
    "current_skill_duration": (0.0, 120.0),
    "recent_sensor_disagreement": (0.0, 50.0),
    "health_trend": (-1.0, 1.0),
    "target_health_trend": (-1.0, 1.0),
    "motion_trend": (-1.0, 1.0),
}


def is_normalized_feature(name: str) -> bool:
    if any(name.endswith(s) for s in NON_NORMALIZED_SUFFIXES):
        return False
    if any(name.startswith(p) for p in NON_NORMALIZED_PREFIXES):
        return False
    return True


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _clamp(name: str, value: float) -> float:
    bounds = _CLAMP.get(name)
    if bounds is None:
        if name.endswith("_value") and name.startswith(("player_hp", "target_hp", "objective", "motion")):
            return max(0.0, min(1.0, value)) if "motion" not in name else max(0.0, min(100.0, value))
        if name.endswith("_confidence"):
            return max(0.0, min(1.0, value))
        return value
    lo, hi = bounds
    return max(lo, min(hi, value))


def _triple(value: Any, confidence: Any, *, boolean: bool = False) -> tuple[float, float, float]:
    """Return (value, known, confidence). Unknown → value=0, known=0, conf=0."""
    conf = _opt_float(confidence)
    if boolean:
        if value is None:
            return 0.0, 0.0, 0.0
        return (1.0 if bool(value) else 0.0), 1.0, (
            max(0.0, min(1.0, conf)) if conf is not None else 1.0
        )
    fv = _opt_float(value)
    if fv is None:
        return 0.0, 0.0, 0.0
    return fv, 1.0, (max(0.0, min(1.0, conf)) if conf is not None else 1.0)


def _as_observation(obj: Any) -> Any:
    if obj is None:
        return None
    try:
        from playmind.observations import Observation
    except ImportError:  # pragma: no cover
        return None
    if isinstance(obj, Observation):
        return obj
    if isinstance(obj, Mapping):
        raw = dict(obj)
        # Schema-v2 callers commonly use the canonical ``player_hp`` spelling,
        # while the migration parser historically consumed
        # ``vision_player_hp``. Preserve the explicit known/unknown value.
        if (
            "player_hp" in raw
            and "vision_player_hp" not in raw
            and not isinstance(raw.get("player"), Mapping)
        ):
            raw["vision_player_hp"] = raw["player_hp"]
        return Observation.from_legacy_dict(raw)
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


def extract_sensor_triples(observation: Any = None) -> dict[str, tuple[float, float, float]]:
    """Map sensor name → (value, known, confidence) for schema sensors."""
    obs = _as_observation(observation)
    out: dict[str, tuple[float, float, float]] = {}
    if obs is None:
        for name in TRIPLE_SENSORS:
            out[name] = (0.0, 0.0, 0.0)
        return out

    out["player_hp"] = _triple(obs.player_hp, obs.player_hp_confidence)
    out["target_hp"] = _triple(obs.target_hp, obs.target_hp_confidence)
    out["has_target"] = _triple(obs.has_target, obs.has_target_confidence, boolean=True)
    out["in_combat"] = _triple(obs.in_combat, obs.in_combat_confidence, boolean=True)
    out["is_dead"] = _triple(obs.is_dead, None, boolean=True)
    out["is_ghost"] = _triple(obs.is_ghost, None, boolean=True)
    out["motion"] = _triple(obs.motion, obs.motion_confidence)
    out["hostiles_near"] = _triple(obs.hostiles_near, None, boolean=True)
    out["hostile_count"] = _triple(obs.hostile_count, obs.hostile_count_confidence)
    out["blocking_modal"] = _triple(obs.blocking_modal, None, boolean=True)
    out["objective_progress"] = _triple(obs.objective_progress, None)
    return out


def structured_feature_vector_v2(
    observation: Any = None,
    temporal_summary: Any = None,
    *,
    feature_dim: int | None = None,
) -> list[float]:
    """Build schema-v2 feature vector with explicit known masks."""
    obs_in = observation
    summary_in = temporal_summary
    if isinstance(observation, Mapping) and temporal_summary is None:
        if any(k in observation for k in ("obs", "observation", "temporal_summary", "summary")):
            ctx = observation
            obs_in = ctx.get("observation", ctx.get("obs", ctx))
            summary_in = ctx.get("temporal_summary", ctx.get("summary"))

    triples = extract_sensor_triples(obs_in)
    obs = _as_observation(obs_in)
    summary = _as_temporal_summary(summary_in)

    by_name: dict[str, float] = {}
    for sensor in TRIPLE_SENSORS:
        value, known, conf = triples[sensor]
        if sensor in _BOOL_SENSORS:
            value = 1.0 if value >= 0.5 else 0.0
        elif sensor == "hostile_count":
            value = _clamp("hostile_count_value", value) if known else 0.0
            value = max(0.0, min(20.0, value))
        elif sensor in {"player_hp", "target_hp", "objective_progress"}:
            value = max(0.0, min(1.0, value)) if known else 0.0
        elif sensor == "motion":
            value = max(0.0, min(100.0, value)) if known else 0.0
        by_name[f"{sensor}_value"] = float(value)
        by_name[f"{sensor}_known"] = float(known)
        by_name[f"{sensor}_confidence"] = float(conf)

    if obs is not None:
        by_name["stagnation_count"] = _clamp(
            "stagnation_count", float(obs.stagnation_count or 0)
        )
        by_name["failed_action_streak"] = _clamp(
            "failed_action_streak", float(obs.failed_action_streak or 0)
        )
        by_name["sensor_warning_count"] = _clamp(
            "sensor_warning_count", float(len(obs.sensor_warnings or []))
        )
        phase = str(getattr(obs, "life_phase", None) or "unknown")
    else:
        by_name["stagnation_count"] = 0.0
        by_name["failed_action_streak"] = 0.0
        by_name["sensor_warning_count"] = 0.0
        phase = "unknown"

    for p in LIFE_PHASES:
        by_name[f"life_phase_{p}"] = 1.0 if phase == p else 0.0

    if summary is not None:
        for field_name in TEMPORAL_FIELDS:
            raw = float(getattr(summary, field_name, 0.0) or 0.0)
            by_name[field_name] = _clamp(field_name, raw)
    else:
        for field_name in TEMPORAL_FIELDS:
            by_name[field_name] = 0.0

    vec = [float(by_name[n]) for n in FEATURE_NAMES]
    if feature_dim is not None:
        dim = int(feature_dim)
        if len(vec) < dim:
            vec = vec + [0.0] * (dim - len(vec))
        else:
            vec = vec[:dim]
    return vec


@dataclass
class FeatureNormalizer:
    """Per-feature mean/std from the training split only."""

    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    schema_version: int = FEATURE_SCHEMA_VERSION
    eps: float = 1e-6

    @classmethod
    def fit(cls, vectors: Sequence[Sequence[float]], *, feature_names: Sequence[str] | None = None) -> "FeatureNormalizer":
        names = list(feature_names) if feature_names is not None else list(FEATURE_NAMES)
        dim = len(names)
        if not vectors:
            return cls(mean=[0.0] * dim, std=[1.0] * dim, feature_names=names)
        n = float(len(vectors))
        means = [0.0] * dim
        for vec in vectors:
            for i in range(dim):
                means[i] += float(vec[i]) if i < len(vec) else 0.0
        means = [m / n for m in means]
        vars_ = [0.0] * dim
        for vec in vectors:
            for i in range(dim):
                x = float(vec[i]) if i < len(vec) else 0.0
                d = x - means[i]
                vars_[i] += d * d
        stds: list[float] = []
        for i, name in enumerate(names):
            if not is_normalized_feature(name):
                means[i] = 0.0
                stds.append(1.0)
                continue
            s = math.sqrt(vars_[i] / max(1.0, n))
            stds.append(s if s > 1e-6 else 1.0)
        return cls(mean=means, std=stds, feature_names=names)

    def transform(self, vector: Sequence[float]) -> list[float]:
        out: list[float] = []
        for i, name in enumerate(self.feature_names):
            x = float(vector[i]) if i < len(vector) else 0.0
            if not is_normalized_feature(name):
                out.append(x)
                continue
            mean = self.mean[i] if i < len(self.mean) else 0.0
            std = self.std[i] if i < len(self.std) else 1.0
            out.append((x - mean) / max(std, self.eps))
        return out

    def transform_sequence(self, seq: Sequence[Sequence[float]]) -> list[list[float]]:
        return [self.transform(v) for v in seq]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": list(self.mean),
            "std": list(self.std),
            "feature_names": list(self.feature_names),
            "schema_version": int(self.schema_version),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "FeatureNormalizer":
        if not raw:
            names = list(FEATURE_NAMES)
            return cls(mean=[0.0] * len(names), std=[1.0] * len(names), feature_names=names)
        names = [str(x) for x in (raw.get("feature_names") or FEATURE_NAMES)]
        mean = [float(x) for x in (raw.get("mean") or [0.0] * len(names))]
        std = [float(x) for x in (raw.get("std") or [1.0] * len(names))]
        return cls(
            mean=mean,
            std=std,
            feature_names=names,
            schema_version=int(raw.get("schema_version") or FEATURE_SCHEMA_VERSION),
        )


def schema_metadata() -> dict[str, Any]:
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_dim": FEATURE_DIM,
        "triple_sensors": list(TRIPLE_SENSORS),
        "life_phases": list(LIFE_PHASES),
        "temporal_fields": list(TEMPORAL_FIELDS),
    }


def assert_compatible_feature_schema(
    checkpoint_meta: Mapping[str, Any],
    *,
    expected_version: int = FEATURE_SCHEMA_VERSION,
    expected_names: Sequence[str] | None = None,
) -> None:
    """Raise ValueError on incompatible feature ordering / version."""
    ver = int(checkpoint_meta.get("feature_schema_version") or 0)
    names = checkpoint_meta.get("feature_names") or checkpoint_meta.get("config", {}).get("feature_names")
    expected = list(expected_names) if expected_names is not None else list(FEATURE_NAMES)
    if ver and ver != expected_version:
        raise ValueError(
            f"Incompatible feature_schema_version={ver}; expected {expected_version}. "
            "Load via legacy adapter or retrain."
        )
    if names is not None:
        got = [str(x) for x in names]
        if got != expected:
            raise ValueError(
                "Incompatible feature name ordering in checkpoint. "
                "Refusing silent remapping."
            )
