"""Skill-classifier policy stub for Learning Architecture V2.

Importing this module does **not** require PyTorch. When ``torch`` is
available, a tiny CNN+GRU-shaped ``nn.Module`` is defined for future training;
otherwise ``SkillPolicyV2`` is a pure-Python (numpy/sklearn-free) stub.

Real CNN+GRU behavior-cloning training needs torch — install with::

    pip install torch

Until a checkpoint is trained, ``predict_skill`` returns a uniform /
low-confidence skill so HybridPolicy falls back to scripted.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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

# Low confidence so HybridPolicy (< threshold ~0.45) falls back to scripted.
UNTRAINED_CONFIDENCE = 0.05
MODEL_VERSION = "skill-policy-v2-stub"


if TORCH_AVAILABLE:

    class SkillClassifierNet(nn.Module):  # type: ignore[misc]
        """Minimal CNN+GRU-shaped stub — not a production vision model.

        Accepts a flat feature vector (demo path) or a tiny fake image batch for
        scaffolding. Real training should replace this with frame encoders.
        """

        def __init__(self, n_skills: int, feature_dim: int = 32, hidden: int = 64) -> None:
            super().__init__()
            self.feature_dim = feature_dim
            self.fc = nn.Sequential(
                nn.Linear(feature_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            # GRU over a length-1 "sequence" of projected features (scaffold).
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.head = nn.Linear(hidden, n_skills)

        def forward(self, features: Any) -> Any:
            # features: (B, F) or (B, T, F)
            if features.dim() == 2:
                x = self.fc(features)
                x = x.unsqueeze(1)
            else:
                b, t, f = features.shape
                x = self.fc(features.reshape(b * t, f)).reshape(b, t, -1)
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])

else:
    SkillClassifierNet = None  # type: ignore[misc, assignment]


def _as_feature_list(features: Any) -> list[float]:
    if features is None:
        return []
    if isinstance(features, Mapping):
        # Stable-ish order by sorted keys for dict observations / summaries.
        vals: list[float] = []
        for k in sorted(features.keys()):
            v = features[k]
            if isinstance(v, bool):
                vals.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                vals.append(float(v))
            elif v is None:
                vals.append(0.0)
        return vals
    if isinstance(features, (list, tuple)):
        out: list[float] = []
        for v in features:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(0.0)
        return out
    if TORCH_AVAILABLE and torch is not None and isinstance(features, torch.Tensor):
        return [float(x) for x in features.detach().cpu().flatten().tolist()]
    try:
        return [float(features)]
    except (TypeError, ValueError):
        return []


class SkillPolicyV2:
    """Skill classifier with JSON checkpoint metadata; optional torch weights.

    When untrained (default), ``predict_skill`` returns the first skill (or
    ``wait``) with uniform/low confidence. Documents that real CNN+GRU training
    requires torch.
    """

    model_version: str = MODEL_VERSION

    def __init__(
        self,
        skill_names: Sequence[str] | None = None,
        *,
        feature_dim: int = 32,
        trained: bool = False,
    ) -> None:
        names = list(skill_names) if skill_names is not None else list(DEFAULT_SKILLS)
        if not names:
            names = ["wait"]
        self.skill_names = names
        self.feature_dim = int(feature_dim)
        self.trained = bool(trained)
        self.metadata: dict[str, Any] = {
            "model_version": self.model_version,
            "skill_names": list(self.skill_names),
            "feature_dim": self.feature_dim,
            "trained": self.trained,
            "torch_available": TORCH_AVAILABLE,
            "note": "Real CNN+GRU training needs torch; this is a stub/scaffold.",
        }
        self._net: Any = None
        if TORCH_AVAILABLE and SkillClassifierNet is not None:
            self._net = SkillClassifierNet(len(self.skill_names), feature_dim=self.feature_dim)

    def predict_skill(self, features: Any) -> tuple[str, float]:
        """Return ``(skill, confidence)``.

        Untrained: uniform prior over skills with low confidence.
        With torch + trained weights: softmax argmax (scaffold).
        """
        n = len(self.skill_names)
        if not self.trained or self._net is None or not TORCH_AVAILABLE:
            # Uniform / low confidence when untrained.
            skill = "wait" if "wait" in self.skill_names else self.skill_names[0]
            conf = UNTRAINED_CONFIDENCE if n else 0.0
            if n > 1:
                # Document uniform prior in confidence scale.
                conf = min(UNTRAINED_CONFIDENCE, 1.0 / float(n))
            return skill, float(conf)

        vec = _as_feature_list(features)
        # Pad / truncate to feature_dim.
        if len(vec) < self.feature_dim:
            vec = vec + [0.0] * (self.feature_dim - len(vec))
        else:
            vec = vec[: self.feature_dim]
        assert torch is not None
        self._net.eval()
        with torch.no_grad():
            t = torch.tensor([vec], dtype=torch.float32)
            logits = self._net(t)[0]
            probs = torch.softmax(logits, dim=-1)
            idx = int(torch.argmax(probs).item())
            conf = float(probs[idx].item())
        return self.skill_names[idx], conf

    def choose_skill(
        self,
        context: Mapping[str, Any],
        allowed_skills: Sequence[str],
    ) -> Any:
        """HighLevelPolicy-compatible wrapper around ``predict_skill``."""
        from playmind.policies.base import PolicyDecision

        feats = context.get("features")
        if feats is None:
            feats = context.get("obs") or context
        skill, conf = self.predict_skill(feats)
        allowed = list(dict.fromkeys(str(s) for s in allowed_skills))
        if skill not in set(allowed):
            skill = "wait" if "wait" in allowed else (allowed[0] if allowed else "wait")
            conf = min(conf, UNTRAINED_CONFIDENCE)
        return PolicyDecision(
            skill=skill,
            confidence=conf,
            reason="SkillPolicyV2 stub" if not self.trained else "SkillPolicyV2",
            model_version=self.model_version,
            allowed_skills=allowed,
            used_fallback=not self.trained,
            debug_scores={"trained": 1.0 if self.trained else 0.0},
        )

    def save(self, path: str | Path) -> Path:
        """Save checkpoint metadata JSON (and torch weights when available)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.metadata)
        meta.update(
            {
                "model_version": self.model_version,
                "skill_names": list(self.skill_names),
                "feature_dim": self.feature_dim,
                "trained": self.trained,
                "torch_available": TORCH_AVAILABLE,
                "note": "Real CNN+GRU training needs torch; this is a stub/scaffold.",
            }
        )
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
            torch.save({"state_dict": self._net.state_dict(), "metadata": meta}, weights_path)

        self.metadata = meta
        return meta_path

    @classmethod
    def load(cls, path: str | Path) -> "SkillPolicyV2":
        """Load from checkpoint metadata JSON (optional sibling ``.pt``)."""
        path = Path(path)
        meta_path = path if path.suffix == ".json" else path.with_suffix(".json")
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(
            skill_names=meta.get("skill_names") or list(DEFAULT_SKILLS),
            feature_dim=int(meta.get("feature_dim") or 32),
            trained=bool(meta.get("trained", False)),
        )
        obj.metadata = dict(meta)
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
        "PyTorch is not installed. Real CNN+GRU behavior-cloning training needs torch.\n"
        "Install CPU torch, then re-run:\n"
        "  pip install torch\n"
        "Docs: https://pytorch.org/get-started/locally/"
    )
