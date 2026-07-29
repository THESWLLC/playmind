"""Import-safe SFT/QLoRA training entry point for the PlayMind planner."""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import random
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from playmind.planner_data.schemas import planner_messages
from playmind.planner_training.presets import (
    TrainingPreset,
    get_preset,
    list_presets,
    model_license_metadata,
)
from playmind.planner_v2.model_registry import DEFAULT_REGISTRY_PATH, ModelRegistry

DEFAULT_RUNS_ROOT = Path("models/playmind/runs")
DEFAULT_SFT_TRAIN = Path("data/playmind/planner/sft/train.jsonl")
DEFAULT_SFT_VAL = Path("data/playmind/planner/sft/val.jsonl")


@dataclass
class SFTTrainingConfig:
    base_model: str | None = None
    preset: str = "rtx_4070_ti_3b_qlora"
    train_file: str | Path = DEFAULT_SFT_TRAIN
    eval_file: str | Path | None = DEFAULT_SFT_VAL
    runs_root: str | Path = DEFAULT_RUNS_ROOT
    registry_path: str | Path = DEFAULT_REGISTRY_PATH
    run_id: str | None = None
    seed: int = 42
    max_steps: int | None = None
    early_stopping_patience: int = 2
    resume_from_checkpoint: str | bool | None = None
    smoke: bool = False
    dry_run: bool = False
    register_candidate: bool = True


class MissingTrainingDependency(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id(prefix: str, seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{stamp}-s{seed}"


def _coerce_config(
    config: SFTTrainingConfig | Mapping[str, Any] | None,
    **overrides: Any,
) -> SFTTrainingConfig:
    if config is None:
        values: dict[str, Any] = {}
    elif isinstance(config, SFTTrainingConfig):
        values = asdict(config)
    elif isinstance(config, Mapping):
        values = dict(config)
    else:
        raise TypeError("config must be SFTTrainingConfig, a mapping, or None")
    values.update({key: value for key, value in overrides.items() if value is not None})
    allowed = {item.name for item in fields(SFTTrainingConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise TypeError(f"unknown SFT config fields: {', '.join(unknown)}")
    return SFTTrainingConfig(**values)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(dict(value))
    return rows


def _synthetic_sft_rows() -> list[dict[str, Any]]:
    states_and_plans = [
        (
            {"goal": "stay safe", "life_phase": "alive", "available_skills": ["wait"]},
            {"skills": ["wait"], "rationale": "sensor state is uncertain"},
        ),
        (
            {
                "goal": "recover",
                "life_phase": "alive",
                "available_skills": ["recover_health", "wait"],
                "sensors": {"player_hp": {"value": 0.1, "known": True, "confidence": 1.0}},
            },
            {"skills": ["recover_health"], "rationale": "health is critical"},
        ),
    ]
    return [
        {
            "example_id": f"smoke-{index}",
            "messages": planner_messages(state, plan),
            "planner_state": state,
            "plan": plan,
        }
        for index, (state, plan) in enumerate(states_and_plans)
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _message_text(row: Mapping[str, Any], tokenizer: Any) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        state = row.get("planner_state") or {}
        plan = row.get("plan") or {}
        messages = planner_messages(
            dict(state) if isinstance(state, Mapping) else {},
            dict(plan) if isinstance(plan, Mapping) else {},
        )
    chunks = []
    for message in messages:
        role = str(message.get("role") or "user").title()
        chunks.append(f"### {role}:\n{message.get('content', '')}")
    return "\n\n".join(chunks) + "\n"


def _load_modules() -> dict[str, Any]:
    missing: list[str] = []
    modules: dict[str, Any] = {}
    for name in ("torch", "transformers", "peft", "trl", "datasets"):
        try:
            modules[name] = importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise MissingTrainingDependency(
            "Real planner SFT training requires optional packages: "
            + ", ".join(missing)
            + ". Install torch transformers datasets peft trl accelerate; "
            "install bitsandbytes for CUDA QLoRA. --smoke needs none of them."
        )
    try:
        modules["bitsandbytes"] = importlib.import_module("bitsandbytes")
    except ImportError:
        modules["bitsandbytes"] = None
    return modules


def _dtype(torch: Any, requested: str) -> Any:
    if requested == "auto_bf16":
        supported = bool(
            torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        return torch.bfloat16 if supported else torch.float16
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(requested, torch.float32)


def _supported_kwargs(callable_object: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return dict(values)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(values)
    return {key: value for key, value in values.items() if key in signature.parameters}


def _completion_collator(
    trl: Any, transformers: Any, tokenizer: Any
) -> tuple[Any, str]:
    collator_cls = getattr(trl, "DataCollatorForCompletionOnlyLM", None)
    if collator_cls is not None:
        marker = "\n\n### Assistant:\n"
        marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        try:
            return (
                collator_cls(response_template=marker_ids, tokenizer=tokenizer),
                "trl",
            )
        except (TypeError, ValueError):
            return collator_cls(marker_ids, tokenizer=tokenizer), "trl"

    class CompletionOnlyCollator:
        """Mask every label through the final assistant response marker."""

        def __init__(self) -> None:
            self.base = transformers.DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            )
            self.markers = [
                tokenizer.encode(value, add_special_tokens=False)
                for value in ("\n\n### Assistant:\n", "### Assistant:\n")
            ]

        @staticmethod
        def _last_index(sequence: Sequence[int], marker: Sequence[int]) -> int:
            for index in range(len(sequence) - len(marker), -1, -1):
                if list(sequence[index : index + len(marker)]) == list(marker):
                    return index
            return -1

        def __call__(self, features: Sequence[Mapping[str, Any]]) -> Any:
            batch = self.base(features)
            for row_index, input_ids in enumerate(batch["input_ids"]):
                sequence = input_ids.tolist()
                match = (-1, 0)
                for marker in self.markers:
                    index = self._last_index(sequence, marker)
                    if index > match[0]:
                        match = (index, len(marker))
                if match[0] < 0:
                    raise ValueError(
                        "assistant response marker was truncated; reduce prompt "
                        "size or increase max_seq_length"
                    )
                batch["labels"][row_index, : match[0] + match[1]] = -100
            return batch

    return CompletionOnlyCollator(), "manual_label_masking"


def _train_real(
    config: SFTTrainingConfig,
    preset: TrainingPreset,
    run_dir: Path,
) -> tuple[dict[str, Any], str, str]:
    modules = _load_modules()
    torch = modules["torch"]
    transformers = modules["transformers"]
    peft = modules["peft"]
    trl = modules["trl"]
    datasets = modules["datasets"]

    cuda = bool(torch.cuda.is_available())
    use_qlora = bool(preset.load_in_4bit and cuda and modules["bitsandbytes"])
    if preset.load_in_4bit and not use_qlora:
        fallback = (
            "bitsandbytes CUDA QLoRA unavailable; using full LoRA with "
            + ("fp16" if cuda else "fp32")
            + " weights. This uses substantially more memory."
        )
        print(fallback, file=sys.stderr)
    else:
        fallback = ""
    dtype = _dtype(torch, preset.compute_dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        preset.base_model, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "torch_dtype": dtype,
    }
    quantization = "none"
    if use_qlora:
        model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=preset.quant_type,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = "auto"
        quantization = f"4bit-{preset.quant_type}"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        preset.base_model, **model_kwargs
    )
    if use_qlora:
        model = peft.prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=preset.gradient_checkpointing
        )
    if preset.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=preset.lora_r,
            lora_alpha=preset.lora_alpha,
            lora_dropout=preset.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )

    train_rows = _read_jsonl(config.train_file)
    if not train_rows:
        raise ValueError(f"SFT training data is empty: {config.train_file}")
    eval_path = Path(config.eval_file) if config.eval_file else None
    eval_rows = _read_jsonl(eval_path) if eval_path and eval_path.is_file() else []
    train_text = [_message_text(row, tokenizer) for row in train_rows]
    eval_text = [_message_text(row, tokenizer) for row in eval_rows]
    train_dataset = datasets.Dataset.from_dict({"text": train_text})
    eval_dataset = (
        datasets.Dataset.from_dict({"text": eval_text}) if eval_text else None
    )
    collator, loss_mode = _completion_collator(trl, transformers, tokenizer)
    output_dir = run_dir / "checkpoints"
    argument_values = {
        "output_dir": str(output_dir),
        "num_train_epochs": preset.epochs,
        "max_steps": config.max_steps if config.max_steps is not None else -1,
        "per_device_train_batch_size": preset.micro_batch_size,
        "per_device_eval_batch_size": preset.micro_batch_size,
        "gradient_accumulation_steps": preset.gradient_accumulation_steps,
        "gradient_checkpointing": preset.gradient_checkpointing,
        "learning_rate": preset.learning_rate,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": 50,
        "eval_strategy": "steps" if eval_dataset is not None else "no",
        "evaluation_strategy": "steps" if eval_dataset is not None else "no",
        "eval_steps": 50,
        "load_best_model_at_end": eval_dataset is not None,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": config.seed,
        "data_seed": config.seed,
        "bf16": dtype is torch.bfloat16,
        "fp16": dtype is torch.float16 and cuda,
        "report_to": [],
        "remove_unused_columns": False,
    }
    config_cls = getattr(trl, "SFTConfig", None)
    if config_cls is not None:
        argument_values.update(
            {
                "max_seq_length": preset.max_seq_length,
                "max_length": preset.max_seq_length,
                "dataset_text_field": "text",
            }
        )
        training_args = config_cls(
            **_supported_kwargs(config_cls.__init__, argument_values)
        )
    else:
        training_args = transformers.TrainingArguments(
            **_supported_kwargs(
                transformers.TrainingArguments.__init__, argument_values
            )
        )
    trainer_values: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
        "dataset_text_field": "text",
        "max_seq_length": preset.max_seq_length,
        "data_collator": collator,
    }
    trainer = trl.SFTTrainer(
        **{
            key: value
            for key, value in _supported_kwargs(
                trl.SFTTrainer.__init__, trainer_values
            ).items()
            if value is not None
        }
    )
    if eval_dataset is not None and config.early_stopping_patience > 0:
        callback = transformers.EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience
        )
        trainer.add_callback(callback)
    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metrics = dict(getattr(result, "metrics", {}) or {})
    metrics["train_examples"] = len(train_rows)
    metrics["eval_examples"] = len(eval_rows)
    metrics["completion_loss_mode"] = loss_mode
    return metrics, quantization, fallback


def _write_metrics(path: Path, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in sorted(metrics.items()):
            writer.writerow(
                [key, json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value]
            )


def _register_candidate(
    registry_path: str | Path,
    *,
    run_id: str,
    preset: TrainingPreset,
    run_dir: Path,
    quantization: str,
    metrics: Mapping[str, Any],
    dataset_version: str,
) -> dict[str, Any]:
    registry = ModelRegistry(registry_path)
    return registry.register(
        run_id,
        display_name=f"PlayMind planner {run_id}",
        base_model=preset.base_model,
        adapter_path=str(run_dir / "adapter"),
        quantization=quantization,
        dataset_version=dataset_version,
        train_metrics=metrics,
        status="candidate",
        reason="training completed; evaluation and explicit promotion required",
    )


def train_sft(
    config: SFTTrainingConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Train an adapter, or execute a dependency-free synthetic smoke run."""
    settings = _coerce_config(config, **overrides)
    smoke = bool(settings.smoke or settings.dry_run)
    if smoke and settings.preset == "rtx_4070_ti_3b_qlora" and not settings.base_model:
        settings.preset = "cpu_tiny_smoke"
    preset = get_preset(
        settings.preset,
        base_model=settings.base_model,
        allow_smoke_default=smoke,
    )
    random.seed(settings.seed)
    identifier = settings.run_id or _run_id("planner-sft", settings.seed)
    run_dir = Path(settings.runs_root) / identifier
    run_dir.mkdir(parents=True, exist_ok=False)
    started = _utc_now()
    license_metadata = model_license_metadata(preset.base_model)
    print(json.dumps({"model_license_metadata": license_metadata}, sort_keys=True))
    dataset_path = Path(settings.train_file)
    quantization = "none"
    fallback_message = ""
    try:
        if smoke:
            rows = _synthetic_sft_rows()
            dataset_path = run_dir / "synthetic_sft.jsonl"
            _write_jsonl(dataset_path, rows)
            adapter_dir = run_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "smoke_artifact.json").write_text(
                json.dumps(
                    {
                        "type": "synthetic-smoke-adapter",
                        "base_model": preset.base_model,
                        "steps": min(2, settings.max_steps or 2),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            metrics: dict[str, Any] = {
                "train_loss": 0.0,
                "train_steps": min(2, settings.max_steps or 2),
                "train_examples": len(rows),
                "smoke": True,
                "completion_loss_mode": "synthetic_label_masking",
            }
            quantization = "none-smoke"
        else:
            if not dataset_path.is_file():
                raise FileNotFoundError(f"SFT training file not found: {dataset_path}")
            metrics, quantization, fallback_message = _train_real(
                settings, preset, run_dir
            )
    except RuntimeError as exc:
        is_oom = "out of memory" in str(exc).lower()
        try:
            is_oom = is_oom or isinstance(
                exc, getattr(importlib.import_module("torch").cuda, "OutOfMemoryError", ())
            )
        except (ImportError, TypeError):
            pass
        if is_oom:
            raise RuntimeError(
                "CUDA out of memory during planner training. Try "
                "rtx_4070_ti_3b_qlora, reduce max_seq_length, or increase "
                "gradient accumulation while keeping microbatch=1."
            ) from exc
        raise

    dataset_version = dataset_path.name
    registered = None
    if settings.register_candidate:
        registered = _register_candidate(
            settings.registry_path,
            run_id=identifier,
            preset=preset,
            run_dir=run_dir,
            quantization=quantization,
            metrics=metrics,
            dataset_version=dataset_version,
        )
        if registered["status"] != "candidate":
            raise RuntimeError("training registration must never set production status")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "training_type": "sft",
        "status": "completed",
        "run_id": identifier,
        "started_at": started,
        "completed_at": _utc_now(),
        "run_dir": str(run_dir),
        "adapter_path": str(run_dir / "adapter"),
        "dataset_path": str(dataset_path),
        "preset": preset.to_dict(),
        "seed": settings.seed,
        "smoke": smoke,
        "quantization": quantization,
        "fallback_message": fallback_message,
        "license_metadata": license_metadata,
        "metrics": metrics,
        "registry": registered,
    }
    manifest_path = run_dir / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _write_metrics(run_dir / "metrics.csv", metrics)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model")
    parser.add_argument("--preset", choices=list_presets(), default="rtx_4070_ti_3b_qlora")
    parser.add_argument("--train-file", default=str(DEFAULT_SFT_TRAIN))
    parser.add_argument("--eval-file", default=str(DEFAULT_SFT_VAL))
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", nargs="?", const=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-register", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = train_sft(
            SFTTrainingConfig(
                base_model=args.base_model,
                preset=args.preset,
                train_file=args.train_file,
                eval_file=args.eval_file,
                runs_root=args.runs_root,
                registry_path=args.registry_path,
                run_id=args.run_id,
                seed=args.seed,
                max_steps=args.max_steps,
                early_stopping_patience=args.early_stopping_patience,
                resume_from_checkpoint=args.resume_from_checkpoint,
                smoke=args.smoke,
                dry_run=args.dry_run,
                register_candidate=not args.no_register,
            )
        )
    except (ValueError, FileNotFoundError, FileExistsError, MissingTrainingDependency) as exc:
        print(f"planner SFT training failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_RUNS_ROOT",
    "DEFAULT_SFT_TRAIN",
    "DEFAULT_SFT_VAL",
    "MissingTrainingDependency",
    "SFTTrainingConfig",
    "build_parser",
    "main",
    "train_sft",
]
