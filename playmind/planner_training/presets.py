"""Hardware-aware planner fine-tuning presets.

Presets deliberately do not guess a production model or its license.  GPU
training callers must provide ``base_model`` explicitly; the tiny CPU preset
is reserved for smoke/CI runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainingPreset:
    name: str
    base_model: str | None
    max_seq_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    load_in_4bit: bool
    quant_type: str
    compute_dtype: str
    epochs: float
    learning_rate: float
    device: str
    experimental: bool = False
    description: str = ""

    def validate(self, *, allow_smoke_default: bool = False) -> None:
        if not self.base_model and not allow_smoke_default:
            raise ValueError(
                f"preset {self.name!r} requires base_model; pass --base-model "
                "after reviewing the model's license and terms"
            )
        if self.max_seq_length < 64:
            raise ValueError("max_seq_length must be at least 64")
        if self.micro_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch sizes and accumulation must be positive")
        if self.lora_r < 1 or self.lora_alpha < self.lora_r:
            raise ValueError("LoRA alpha must be at least the positive LoRA rank")
        if self.quant_type not in {"nf4", "fp4"}:
            raise ValueError("quant_type must be 'nf4' or 'fp4'")
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")

    def with_base_model(self, base_model: str | None) -> "TrainingPreset":
        return replace(self, base_model=base_model or self.base_model)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def seq_length(self) -> int:
        return self.max_seq_length

    @property
    def microbatch(self) -> int:
        return self.micro_batch_size

    @property
    def grad_accum(self) -> int:
        return self.gradient_accumulation_steps

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "seq_length": "max_seq_length",
            "microbatch": "micro_batch_size",
            "grad_accum": "gradient_accumulation_steps",
            "grad_checkpoint": "gradient_checkpointing",
            "lora_rank": "lora_r",
        }
        return getattr(self, aliases.get(str(key), str(key)))


PRESETS: dict[str, TrainingPreset] = {
    "rtx_4070_ti_3b_qlora": TrainingPreset(
        name="rtx_4070_ti_3b_qlora",
        base_model=None,
        max_seq_length=1024,
        micro_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        load_in_4bit=True,
        quant_type="nf4",
        compute_dtype="auto_bf16",
        epochs=3.0,
        learning_rate=2e-4,
        device="cuda",
        description="3B QLoRA preset for a 12GB RTX 4070 Ti.",
    ),
    "rtx_4070_ti_7b_qlora_experimental": TrainingPreset(
        name="rtx_4070_ti_7b_qlora_experimental",
        base_model=None,
        max_seq_length=768,
        micro_batch_size=1,
        gradient_accumulation_steps=32,
        gradient_checkpointing=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        load_in_4bit=True,
        quant_type="nf4",
        compute_dtype="auto_bf16",
        epochs=2.0,
        learning_rate=1.5e-4,
        device="cuda",
        experimental=True,
        description=(
            "Experimental 7B QLoRA preset. 12GB cards may OOM; use the 3B "
            "preset or reduce sequence length if that occurs."
        ),
    ),
    "cpu_tiny_smoke": TrainingPreset(
        name="cpu_tiny_smoke",
        base_model="sshleifer/tiny-gpt2",
        max_seq_length=128,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        load_in_4bit=False,
        quant_type="nf4",
        compute_dtype="fp32",
        epochs=1.0,
        learning_rate=5e-4,
        device="cpu",
        description=(
            "Dependency-free CI smoke preset. Smoke mode writes a synthetic "
            "local artifact and never downloads the default model."
        ),
    ),
}


def list_presets() -> list[str]:
    return sorted(PRESETS)


def get_preset(
    name: str,
    *,
    base_model: str | None = None,
    allow_smoke_default: bool = False,
) -> TrainingPreset:
    try:
        preset = PRESETS[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown preset {name!r}; choose one of {', '.join(list_presets())}"
        ) from exc
    configured = preset.with_base_model(base_model)
    configured.validate(
        allow_smoke_default=allow_smoke_default or configured.name == "cpu_tiny_smoke"
    )
    return configured


def validate_preset(
    preset: str | TrainingPreset,
    *,
    base_model: str | None = None,
    allow_smoke_default: bool = False,
) -> TrainingPreset:
    configured = (
        get_preset(
            preset,
            base_model=base_model,
            allow_smoke_default=allow_smoke_default,
        )
        if isinstance(preset, str)
        else preset.with_base_model(base_model)
    )
    configured.validate(allow_smoke_default=allow_smoke_default)
    return configured


def model_license_metadata(base_model: str | Path | None) -> dict[str, Any]:
    """Read license fields from a local config without making assumptions."""
    metadata: dict[str, Any] = {
        "base_model": str(base_model or ""),
        "license": None,
        "source": None,
        "notice": "No license metadata found; verify the model card and terms.",
    }
    if not base_model:
        return metadata
    candidate = Path(str(base_model)).expanduser()
    config_path = candidate / "config.json" if candidate.is_dir() else None
    if config_path is None or not config_path.is_file():
        return metadata
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return metadata
    if not isinstance(raw, Mapping):
        return metadata
    fields = {
        key: raw[key]
        for key in ("license", "license_name", "licenses")
        if key in raw and raw[key] not in (None, "")
    }
    metadata.update(
        {
            "license": fields or None,
            "source": str(config_path),
            "notice": (
                "License metadata copied from local config.json; verify the "
                "authoritative model card and terms."
                if fields
                else metadata["notice"]
            ),
        }
    )
    return metadata


__all__ = [
    "PRESETS",
    "TrainingPreset",
    "get_preset",
    "list_presets",
    "model_license_metadata",
    "validate_preset",
]
