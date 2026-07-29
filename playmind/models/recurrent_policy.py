"""Recurrent structured-observation skill policy.

The module remains importable without PyTorch.  Dataset inspection and dry
validation therefore do not acquire a heavyweight runtime dependency.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from playmind.models.feature_schema import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FeatureNormalizer,
    assert_compatible_feature_schema,
    structured_feature_vector_v2,
)
from playmind.models.policy_v2 import (
    DEFAULT_SKILLS,
    LegacyCheckpointError,
    TORCH_AVAILABLE,
    UNTRAINED_CONFIDENCE,
)

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence
except ImportError:  # pragma: no cover - exercised on torch-free installs
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    pack_padded_sequence = None  # type: ignore[assignment]


DEFAULT_AUX_KEYS: tuple[str, ...] = (
    "target_valid",
    "combat",
    "death",
    "player_hp_delta",
    "target_hp_delta",
    "progress_delta",
    "skill_success",
)
AUX_KEYS = DEFAULT_AUX_KEYS
AUX_TYPES: dict[str, str] = {
    "target_valid": "binary",
    "combat": "binary",
    "death": "binary",
    "player_hp_delta": "regression",
    "target_hp_delta": "regression",
    "progress_delta": "regression",
    "skill_success": "binary",
}
MODEL_VERSION = "recurrent-skill-policy-v2-1"
CHECKPOINT_SCHEMA_VERSION = 2


def seed_everything(seed: int = 0, *, deterministic: bool = True) -> int:
    """Seed Python and torch, including CUDA when available."""
    value = int(seed)
    random.seed(value)
    if torch is not None:
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (AttributeError, TypeError):  # pragma: no cover - old torch
                pass
    return value


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


if TORCH_AVAILABLE and torch is not None and nn is not None:
    from playmind.models.encoders import StructuredObservationEncoder

    class RecurrentSkillPolicyNet(nn.Module):  # type: ignore[misc]
        """Encode every valid timestep, then classify from the final GRU state.

        ``padding_mask`` uses the project convention ``True == valid``.  Masks
        may describe left or right padding; valid rows are compacted in their
        original order before packing.
        """

        def __init__(
            self,
            feature_dim: int,
            n_skills: int,
            encoder_dim: int = 96,
            hidden_dim: int = 128,
            num_layers: int = 1,
            dropout: float = 0.0,
            bidirectional: bool = False,
            n_aux: int = len(DEFAULT_AUX_KEYS),
            *,
            aux_names: Sequence[str] | None = None,
            seed: int = 0,
            stateful: bool = False,
        ) -> None:
            super().__init__()
            self.feature_dim = int(feature_dim)
            self.n_skills = int(n_skills)
            self.encoder_dim = int(encoder_dim)
            self.hidden_dim = int(hidden_dim)
            self.num_layers = int(num_layers)
            self.dropout = float(dropout)
            self.bidirectional = bool(bidirectional)
            self.stateful = bool(stateful)
            names = list(aux_names or DEFAULT_AUX_KEYS[: max(0, int(n_aux))])
            if aux_names is not None and n_aux != len(DEFAULT_AUX_KEYS):
                names = names[: max(0, int(n_aux))]
            self.aux_names = tuple(str(x) for x in names)
            self.n_aux = len(self.aux_names)
            self._hidden_state: Any = None

            # Keep initialization reproducible without permanently consuming the
            # caller's CPU RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                self.encoder = StructuredObservationEncoder(
                    self.feature_dim, self.encoder_dim, dropout=self.dropout
                )
                self.gru = nn.GRU(
                    input_size=self.encoder_dim,
                    hidden_size=self.hidden_dim,
                    num_layers=self.num_layers,
                    batch_first=True,
                    dropout=self.dropout if self.num_layers > 1 else 0.0,
                    bidirectional=self.bidirectional,
                )
                head_dim = self.hidden_dim * (2 if self.bidirectional else 1)
                self.skill_head = nn.Linear(head_dim, self.n_skills)
                self.aux_heads = nn.ModuleDict(
                    {name: nn.Linear(head_dim, 1) for name in self.aux_names}
                )

        def reset_state(self, batch_indices: Sequence[int] | None = None) -> None:
            """Reset all state, or zero selected batch rows."""
            if batch_indices is None or self._hidden_state is None:
                self._hidden_state = None
                return
            state = self._hidden_state.clone()
            for index in batch_indices:
                if 0 <= int(index) < state.shape[1]:
                    state[:, int(index), :] = 0
            self._hidden_state = state

        def detach_state(self) -> None:
            if self._hidden_state is not None:
                self._hidden_state = self._hidden_state.detach()

        @staticmethod
        def _resolve_mask(
            features: Any,
            lengths: Any = None,
            padding_mask: Any = None,
        ) -> tuple[Any, Any]:
            batch, time, _ = features.shape
            device = features.device
            if padding_mask is not None:
                mask = torch.as_tensor(padding_mask, dtype=torch.bool, device=device)
                if tuple(mask.shape) != (batch, time):
                    raise ValueError(
                        f"padding_mask must have shape {(batch, time)}, got {tuple(mask.shape)}"
                    )
                mask_lengths = mask.sum(dim=1).to(dtype=torch.long)
                if lengths is not None:
                    supplied = torch.as_tensor(lengths, dtype=torch.long, device=device)
                    if supplied.numel() != batch or not torch.equal(
                        supplied.reshape(-1), mask_lengths
                    ):
                        raise ValueError("lengths do not match padding_mask valid counts")
                lengths_t = mask_lengths
            elif lengths is not None:
                lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=device).reshape(-1)
                if lengths_t.numel() != batch:
                    raise ValueError(f"lengths must contain {batch} values")
                steps = torch.arange(time, device=device).unsqueeze(0)
                mask = steps < lengths_t.unsqueeze(1)
            else:
                lengths_t = torch.full((batch,), time, dtype=torch.long, device=device)
                mask = torch.ones((batch, time), dtype=torch.bool, device=device)
            if bool((lengths_t <= 0).any()) or bool((lengths_t > time).any()):
                raise ValueError(f"all sequence lengths must be in [1, {time}]")
            return lengths_t, mask

        @staticmethod
        def _compact_valid(encoded: Any, mask: Any, lengths: Any) -> Any:
            """Move valid rows to the left while retaining temporal order."""
            batch, time, width = encoded.shape
            compact = encoded.new_zeros((batch, time, width))
            for i in range(batch):
                valid = encoded[i][mask[i]]
                compact[i, : int(lengths[i].item())] = valid
            return compact

        def forward(
            self,
            features: Any,
            lengths: Any = None,
            padding_mask: Any = None,
            hidden_state: Any = None,
        ) -> tuple[Any, dict[str, Any]]:
            if features.dim() == 2:
                features = features.unsqueeze(1)
            if features.dim() != 3:
                raise ValueError("features must have shape (batch, time, feature_dim)")
            if features.shape[-1] != self.feature_dim:
                raise ValueError(
                    f"expected feature_dim={self.feature_dim}, got {features.shape[-1]}"
                )
            lengths_t, mask = self._resolve_mask(features, lengths, padding_mask)
            encoded = self.encoder(features)
            compact = self._compact_valid(encoded, mask, lengths_t)

            initial = hidden_state
            if initial is None and self.stateful and self._hidden_state is not None:
                if self._hidden_state.shape[1] == features.shape[0]:
                    initial = self._hidden_state.to(features.device)
                else:
                    self.reset_state()
            packed = pack_padded_sequence(
                compact,
                lengths_t.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _packed_out, final_hidden = self.gru(packed, initial)
            if self.stateful:
                self._hidden_state = final_hidden.detach()

            if self.bidirectional:
                # Last layer's forward and backward final states.
                final = torch.cat((final_hidden[-2], final_hidden[-1]), dim=-1)
            else:
                final = final_hidden[-1]
            skill_logits = self.skill_head(final)
            aux_outputs = {
                name: head(final).squeeze(-1) for name, head in self.aux_heads.items()
            }
            return skill_logits, aux_outputs

else:  # pragma: no cover - behavior is tested through wrapper dry paths
    RecurrentSkillPolicyNet = None  # type: ignore[misc, assignment]


class RecurrentSkillPolicyV2:
    """Stateless-by-default recurrent policy wrapper and checkpoint owner."""

    model_version = MODEL_VERSION

    def __init__(
        self,
        skill_names: Sequence[str] | None = None,
        *,
        feature_dim: int = FEATURE_DIM,
        encoder_dim: int = 96,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        aux_names: Sequence[str] | None = None,
        history_length: int = 16,
        normalizer: FeatureNormalizer | Mapping[str, Any] | None = None,
        trained: bool = False,
        seed: int = 0,
        stateful: bool = False,
        training_config: Mapping[str, Any] | None = None,
        device: str | None = None,
    ) -> None:
        names = list(skill_names or DEFAULT_SKILLS)
        self.skill_names = names or ["wait"]
        self.feature_dim = int(feature_dim)
        self.encoder_dim = int(encoder_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.aux_names = tuple(str(x) for x in (aux_names or DEFAULT_AUX_KEYS))
        self.history_length = max(1, int(history_length))
        self.trained = bool(trained)
        self.seed = int(seed)
        self.stateful = bool(stateful)
        self.training_config = dict(training_config or {})
        if isinstance(normalizer, FeatureNormalizer):
            self.normalizer = normalizer
        elif normalizer is not None:
            self.normalizer = FeatureNormalizer.from_dict(normalizer)
        else:
            self.normalizer = None
        self.device = self._resolve_device(device)
        self._net: Any = None
        self.training_state: dict[str, Any] = {}
        self.metadata = self._build_metadata()
        if RecurrentSkillPolicyNet is not None and torch is not None:
            self._net = RecurrentSkillPolicyNet(
                self.feature_dim,
                len(self.skill_names),
                encoder_dim=self.encoder_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                n_aux=len(self.aux_names),
                aux_names=self.aux_names,
                seed=self.seed,
                stateful=self.stateful,
            )
            self._net.to(self.device)

    @property
    def net(self) -> Any:
        return self._net

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        requested = str(device or "auto")
        if requested == "auto":
            return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and (
            torch is None or not torch.cuda.is_available()
        ):
            raise ValueError(f"CUDA device requested but unavailable: {requested}")
        return requested

    def to(self, device: str) -> "RecurrentSkillPolicyV2":
        self.device = self._resolve_device(device)
        if self._net is not None:
            self._net.to(self.device)
        return self

    def reset_state(self, batch_indices: Sequence[int] | None = None) -> None:
        if self._net is not None:
            self._net.reset_state(batch_indices)

    def _build_metadata(self) -> dict[str, Any]:
        architecture = {
            "encoder_dim": self.encoder_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "stateful": self.stateful,
        }
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_type": "recurrent_skill_policy",
            "model_version": self.model_version,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "feature_dim": self.feature_dim,
            "skill_names": list(self.skill_names),
            "aux_names": list(self.aux_names),
            "history_length": self.history_length,
            "architecture": architecture,
            "training_config": dict(self.training_config),
            "normalization_stats": (
                self.normalizer.to_dict() if self.normalizer is not None else None
            ),
            "input_modalities": ["structured"],
            "trained": self.trained,
            "seed": self.seed,
            "git_commit": _git_commit(),
        }

    def _coerce_sequence(self, values: Any) -> list[list[float]]:
        if values is None:
            return [structured_feature_vector_v2(feature_dim=self.feature_dim)]
        if torch is not None and isinstance(values, torch.Tensor):
            raw = values.detach().cpu().tolist()
            if values.dim() == 1:
                raw = [raw]
            return [[float(x) for x in row] for row in raw]
        if not isinstance(values, (list, tuple)):
            values = [values]
        if values and isinstance(values[0], (int, float, bool)):
            values = [values]
        sequence: list[list[float]] = []
        for value in values:
            if isinstance(value, Mapping) or not isinstance(value, (list, tuple)):
                sequence.append(
                    structured_feature_vector_v2(value, feature_dim=self.feature_dim)
                )
            else:
                row = [float(x) for x in value]
                row = (row + [0.0] * self.feature_dim)[: self.feature_dim]
                sequence.append(row)
        return sequence[-self.history_length :] or [
            structured_feature_vector_v2(feature_dim=self.feature_dim)
        ]

    def sequence_from_context(self, context: Mapping[str, Any]) -> list[list[float]]:
        if context.get("feature_sequence") is not None:
            return self._coerce_sequence(context["feature_sequence"])
        history = context.get("history")
        candidates = context.get("observations", context.get("observation_history"))
        if candidates is None and history is not None:
            candidates = getattr(history, "observations", history)
        if candidates is not None:
            return self._coerce_sequence(list(candidates))
        observation = context.get("observation", context.get("obs", context))
        summary = context.get("temporal_summary", context.get("summary"))
        return [
            structured_feature_vector_v2(
                observation, summary, feature_dim=self.feature_dim
            )
        ]

    def _normalize_tensor(self, features: Any, padding_mask: Any = None) -> Any:
        if self.normalizer is None or torch is None:
            return features
        mean = torch.tensor(
            self.normalizer.mean, dtype=features.dtype, device=features.device
        )
        std = torch.tensor(
            self.normalizer.std, dtype=features.dtype, device=features.device
        ).clamp_min(self.normalizer.eps)
        normalized = (features - mean) / std
        # FeatureNormalizer intentionally leaves masks and one-hot fields alone.
        from playmind.models.feature_schema import is_normalized_feature

        apply = torch.tensor(
            [is_normalized_feature(n) for n in self.normalizer.feature_names],
            dtype=torch.bool,
            device=features.device,
        )
        normalized = torch.where(apply, normalized, features)
        if padding_mask is not None:
            mask = torch.as_tensor(
                padding_mask, dtype=torch.bool, device=features.device
            ).unsqueeze(-1)
            normalized = torch.where(mask, normalized, torch.zeros_like(normalized))
        return normalized

    def predict_sequence(
        self,
        features_BT: Any,
        lengths: Any = None,
        padding_mask: Any = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Return batched skill logits and raw auxiliary outputs."""
        if self._net is None or torch is None:
            raise RuntimeError("PyTorch is required for recurrent prediction")
        if isinstance(features_BT, torch.Tensor):
            x = features_BT.to(device=self.device, dtype=torch.float32)
        else:
            x = torch.tensor(features_BT, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.reshape(1, 1, -1)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
        x = self._normalize_tensor(x, padding_mask)
        self._net.eval()
        with torch.no_grad():
            return self._net(x, lengths=lengths, padding_mask=padding_mask)

    def predict(
        self,
        features: Any = None,
        *,
        observation: Any = None,
        temporal_summary: Any = None,
    ) -> tuple[str, float, dict[str, float]]:
        if features is None:
            sequence = [
                structured_feature_vector_v2(
                    observation, temporal_summary, feature_dim=self.feature_dim
                )
            ]
        else:
            sequence = self._coerce_sequence(features)
        fallback = "wait" if "wait" in self.skill_names else self.skill_names[0]
        if not self.trained or self._net is None or torch is None:
            return fallback, min(UNTRAINED_CONFIDENCE, 1.0 / len(self.skill_names)), {
                key: 0.0 for key in self.aux_names
            }
        logits, raw_aux = self.predict_sequence([sequence], lengths=[len(sequence)])
        probs = torch.softmax(logits[0], dim=-1)
        index = int(torch.argmax(probs).item())
        aux: dict[str, float] = {}
        for name, output in raw_aux.items():
            value = output[0]
            if AUX_TYPES.get(name) == "binary":
                value = torch.sigmoid(value)
            aux[name] = float(value.item())
        return self.skill_names[index], float(probs[index].item()), aux

    def choose_skill(
        self,
        context: Mapping[str, Any],
        allowed_skills: Sequence[str],
    ) -> Any:
        """Choose after masking disallowed logits, before softmax."""
        from playmind.policies.base import PolicyDecision

        allowed = list(dict.fromkeys(str(x) for x in allowed_skills))
        allowed_indices = [
            i for i, name in enumerate(self.skill_names) if name in set(allowed)
        ]
        fallback = "wait" if "wait" in allowed else (allowed[0] if allowed else "wait")
        sequence = self.sequence_from_context(context)
        aux_values = {key: 0.0 for key in self.aux_names}
        used_fallback = not self.trained or self._net is None or not allowed_indices
        if used_fallback or torch is None:
            skill = fallback
            confidence = min(
                UNTRAINED_CONFIDENCE, 1.0 / max(1, len(allowed_indices))
            )
        else:
            logits, raw_aux = self.predict_sequence(
                [sequence], lengths=[len(sequence)]
            )
            masked = torch.full_like(logits[0], float("-inf"))
            masked[allowed_indices] = logits[0, allowed_indices]
            probabilities = torch.softmax(masked, dim=-1)
            index = int(torch.argmax(probabilities).item())
            skill = self.skill_names[index]
            confidence = float(probabilities[index].item())
            for name, output in raw_aux.items():
                value = output[0]
                if AUX_TYPES.get(name) == "binary":
                    value = torch.sigmoid(value)
                aux_values[name] = float(value.item())
        debug = {f"aux_{k}": float(v) for k, v in aux_values.items()}
        debug["trained"] = 1.0 if self.trained else 0.0
        return PolicyDecision(
            skill=skill,
            confidence=confidence,
            reason="RecurrentSkillPolicyV2",
            model_version=self.model_version,
            allowed_skills=allowed,
            used_fallback=used_fallback,
            temporal_summary=str(context.get("temporal_summary"))
            if context.get("temporal_summary") is not None
            else None,
            debug_scores=debug,
        )

    def save(
        self,
        path: str | Path,
        *,
        training_config: Mapping[str, Any] | None = None,
        training_state: Mapping[str, Any] | None = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if training_config:
            self.training_config.update(dict(training_config))
        self.metadata = self._build_metadata()
        meta_path = path if path.suffix == ".json" else path.with_suffix(".json")
        self.metadata["weights_file"] = (
            meta_path.with_suffix(".pt").name if self._net is not None else None
        )
        tmp = meta_path.with_suffix(meta_path.suffix + f".{os.getpid()}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(self.metadata, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, meta_path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        if self._net is not None and torch is not None:
            state = dict(training_state or {})
            torch.save(
                {
                    "state_dict": self._net.state_dict(),
                    "metadata": self.metadata,
                    "training_state": state,
                },
                meta_path.with_suffix(".pt"),
            )
            self.training_state = state
        return meta_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
    ) -> "RecurrentSkillPolicyV2":
        path = Path(path)
        meta_path = path if path.suffix == ".json" else path.with_suffix(".json")
        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        model_type = str(metadata.get("model_type") or "")
        if model_type != "recurrent_skill_policy":
            raise LegacyCheckpointError(
                f"Checkpoint model_type={model_type or 'missing'} is not recurrent. "
                "Use playmind.models.policy_v2.load_legacy_mlp()."
            )
        if int(metadata.get("checkpoint_schema_version") or 0) != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "Incompatible or missing checkpoint_schema_version; "
                f"expected {CHECKPOINT_SCHEMA_VERSION}."
            )
        if int(metadata.get("feature_schema_version") or 0) != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "Incompatible or missing feature_schema_version; "
                f"expected {FEATURE_SCHEMA_VERSION}."
            )
        if metadata.get("feature_names") is None:
            raise ValueError("Checkpoint is missing required feature_names metadata.")
        assert_compatible_feature_schema(metadata)
        if int(metadata.get("feature_dim") or 0) != FEATURE_DIM:
            raise ValueError(
                f"Incompatible feature_dim={metadata.get('feature_dim')}; "
                f"expected schema-v2 dimension {FEATURE_DIM}."
            )
        architecture = dict(metadata.get("architecture") or {})
        obj = cls(
            skill_names=metadata.get("skill_names"),
            feature_dim=int(metadata["feature_dim"]),
            encoder_dim=int(architecture.get("encoder_dim", 96)),
            hidden_dim=int(architecture.get("hidden_dim", 128)),
            num_layers=int(architecture.get("num_layers", 1)),
            dropout=float(architecture.get("dropout", 0.0)),
            bidirectional=bool(architecture.get("bidirectional", False)),
            aux_names=metadata.get("aux_names") or DEFAULT_AUX_KEYS,
            history_length=int(metadata.get("history_length", 16)),
            normalizer=metadata.get("normalization_stats"),
            trained=bool(metadata.get("trained", False)),
            seed=int(metadata.get("seed", 0)),
            stateful=bool(architecture.get("stateful", False)),
            training_config=metadata.get("training_config") or {},
            device=device,
        )
        obj.metadata = dict(metadata)
        weights_name = metadata.get("weights_file")
        weights_path = (
            meta_path.parent / str(weights_name)
            if weights_name
            else meta_path.with_suffix(".pt")
        )
        if obj._net is not None and torch is not None and weights_path.exists():
            blob = torch.load(weights_path, map_location=obj.device, weights_only=False)
            state_dict = blob.get("state_dict") if isinstance(blob, dict) else blob
            if state_dict is not None:
                obj._net.load_state_dict(state_dict)
            if isinstance(blob, dict):
                obj.training_state = dict(blob.get("training_state") or {})
        return obj


__all__ = [
    "AUX_KEYS",
    "AUX_TYPES",
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_AUX_KEYS",
    "MODEL_VERSION",
    "RecurrentSkillPolicyNet",
    "RecurrentSkillPolicyV2",
    "seed_everything",
]
