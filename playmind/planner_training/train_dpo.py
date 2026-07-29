"""Import-safe preference/DPO training for PlayMind planner adapters."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from playmind.planner_data.schemas import PLANNER_SYSTEM_PROMPT
from playmind.planner_training.presets import (
    get_preset,
    list_presets,
    model_license_metadata,
)
from playmind.planner_training.train_sft import (
    DEFAULT_RUNS_ROOT,
    MissingTrainingDependency,
    _dtype,
    _load_modules,
    _register_candidate,
    _supported_kwargs,
)
from playmind.planner_v2.model_registry import DEFAULT_REGISTRY_PATH

DEFAULT_PREFERENCE_TRAIN = Path("data/playmind/planner/preferences/train.jsonl")
DEFAULT_PREFERENCE_VAL = Path("data/playmind/planner/preferences/val.jsonl")


@dataclass
class DPOTrainingConfig:
    base_model: str | None = None
    preset: str = "rtx_4070_ti_3b_qlora"
    train_file: str | Path = DEFAULT_PREFERENCE_TRAIN
    eval_file: str | Path | None = DEFAULT_PREFERENCE_VAL
    runs_root: str | Path = DEFAULT_RUNS_ROOT
    registry_path: str | Path = DEFAULT_REGISTRY_PATH
    run_id: str | None = None
    seed: int = 42
    beta: float = 0.1
    max_steps: int | None = None
    resume_from_checkpoint: str | bool | None = None
    smoke: bool = False
    dry_run: bool = False
    register_candidate: bool = True


def _coerce_config(
    config: DPOTrainingConfig | Mapping[str, Any] | None,
    **overrides: Any,
) -> DPOTrainingConfig:
    if config is None:
        values: dict[str, Any] = {}
    elif isinstance(config, DPOTrainingConfig):
        values = asdict(config)
    elif isinstance(config, Mapping):
        values = dict(config)
    else:
        raise TypeError("config must be DPOTrainingConfig, a mapping, or None")
    values.update({key: value for key, value in overrides.items() if value is not None})
    allowed = {item.name for item in fields(DPOTrainingConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise TypeError(f"unknown DPO config fields: {', '.join(unknown)}")
    return DPOTrainingConfig(**values)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(dict(value))
    return rows


def _prompt(row: Mapping[str, Any]) -> str:
    state = row.get("planner_state")
    return (
        f"### System:\n{PLANNER_SYSTEM_PROMPT}\n\n### User:\n"
        + json.dumps(
            {"planner_state": dict(state) if isinstance(state, Mapping) else {}},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n\n### Assistant:\n"
    )


def _plan_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    return str(value or "")


def _preference_columns(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    columns = {"prompt": [], "chosen": [], "rejected": []}
    for row in rows:
        chosen = _plan_text(row.get("chosen"))
        rejected = _plan_text(row.get("rejected"))
        if not chosen or not rejected or chosen == rejected:
            continue
        columns["prompt"].append(_prompt(row))
        columns["chosen"].append(chosen)
        columns["rejected"].append(rejected)
    return columns


def _synthetic_preferences() -> list[dict[str, Any]]:
    return [
        {
            "planner_state": {
                "goal": "survive",
                "available_skills": ["recover_health", "engage_target"],
            },
            "chosen": {"skills": ["recover_health"], "rationale": "low health"},
            "rejected": {"skills": ["engage_target"], "rationale": "unsafe"},
        },
        {
            "planner_state": {
                "goal": "wait safely",
                "available_skills": ["wait", "engage_target"],
            },
            "chosen": {"skills": ["wait"], "rationale": "unknown sensors"},
            "rejected": {"skills": ["engage_target"], "rationale": "invented context"},
        },
    ]


def _write_metrics(path: Path, metrics: Mapping[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows((key, value) for key, value in sorted(metrics.items()))


def _write_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> Path:
    path = run_dir / "training_manifest.json"
    path.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _train_real(
    config: DPOTrainingConfig, preset: Any, run_dir: Path
) -> tuple[dict[str, Any], str, str]:
    modules = _load_modules()
    trl = modules["trl"]
    if not hasattr(trl, "DPOTrainer"):
        raise MissingTrainingDependency(
            "Installed trl does not expose DPOTrainer. Upgrade trl, or use "
            "--smoke to validate the pipeline without ML dependencies."
        )
    torch = modules["torch"]
    transformers = modules["transformers"]
    peft = modules["peft"]
    datasets = modules["datasets"]
    cuda = bool(torch.cuda.is_available())
    use_qlora = bool(preset.load_in_4bit and cuda and modules["bitsandbytes"])
    fallback = ""
    if preset.load_in_4bit and not use_qlora:
        fallback = (
            "bitsandbytes CUDA QLoRA unavailable; using full LoRA with "
            + ("fp16" if cuda else "fp32")
            + " weights. This uses substantially more memory."
        )
        print(fallback, file=sys.stderr)
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
        model_kwargs.update(
            {
                "quantization_config": transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=preset.quant_type,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                ),
                "device_map": "auto",
            }
        )
        quantization = f"4bit-{preset.quant_type}"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        preset.base_model, **model_kwargs
    )
    train_rows = _read_jsonl(config.train_file)
    eval_path = Path(config.eval_file) if config.eval_file else None
    eval_rows = _read_jsonl(eval_path) if eval_path and eval_path.is_file() else []
    train_columns = _preference_columns(train_rows)
    if not train_columns["prompt"]:
        raise ValueError(f"no valid preference pairs in {config.train_file}")
    eval_columns = _preference_columns(eval_rows)
    train_dataset = datasets.Dataset.from_dict(train_columns)
    eval_dataset = (
        datasets.Dataset.from_dict(eval_columns) if eval_columns["prompt"] else None
    )
    args_values = {
        "output_dir": str(run_dir / "checkpoints"),
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
        "seed": config.seed,
        "data_seed": config.seed,
        "bf16": dtype is torch.bfloat16,
        "fp16": dtype is torch.float16 and cuda,
        "report_to": [],
        "remove_unused_columns": False,
        "beta": config.beta,
        "max_length": preset.max_seq_length,
        "max_prompt_length": max(32, preset.max_seq_length // 2),
    }
    args_cls = getattr(trl, "DPOConfig", transformers.TrainingArguments)
    training_args = args_cls(
        **_supported_kwargs(args_cls.__init__, args_values)
    )
    peft_config = peft.LoraConfig(
        r=preset.lora_r,
        lora_alpha=preset.lora_alpha,
        lora_dropout=preset.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    trainer_values = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "beta": config.beta,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
        "peft_config": peft_config,
        "max_length": preset.max_seq_length,
        "max_prompt_length": max(32, preset.max_seq_length // 2),
    }
    trainer = trl.DPOTrainer(
        **{
            key: value
            for key, value in _supported_kwargs(
                trl.DPOTrainer.__init__, trainer_values
            ).items()
            if value is not None
        }
    )
    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metrics = dict(getattr(result, "metrics", {}) or {})
    metrics.update(
        {
            "train_preferences": len(train_columns["prompt"]),
            "eval_preferences": len(eval_columns["prompt"]),
            "beta": config.beta,
        }
    )
    return metrics, quantization, fallback


def train_dpo(
    config: DPOTrainingConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    settings = _coerce_config(config, **overrides)
    smoke = bool(settings.smoke or settings.dry_run)
    if smoke and settings.preset == "rtx_4070_ti_3b_qlora" and not settings.base_model:
        settings.preset = "cpu_tiny_smoke"
    preset = get_preset(
        settings.preset,
        base_model=settings.base_model,
        allow_smoke_default=smoke,
    )
    identifier = settings.run_id or (
        "planner-dpo-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"-s{settings.seed}"
    )
    run_dir = Path(settings.runs_root) / identifier
    run_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    license_metadata = model_license_metadata(preset.base_model)
    print(json.dumps({"model_license_metadata": license_metadata}, sort_keys=True))
    fallback = ""
    quantization = "none"
    try:
        if smoke:
            rows = _synthetic_preferences()
            synthetic_path = run_dir / "synthetic_preferences.jsonl"
            with synthetic_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            adapter_dir = run_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "smoke_artifact.json").write_text(
                json.dumps(
                    {
                        "type": "synthetic-smoke-dpo-adapter",
                        "steps": min(2, settings.max_steps or 2),
                        "beta": settings.beta,
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
                "train_preferences": len(rows),
                "beta": settings.beta,
                "smoke": True,
            }
            dataset_path = synthetic_path
            quantization = "none-smoke"
        else:
            dataset_path = Path(settings.train_file)
            if not dataset_path.is_file():
                raise FileNotFoundError(
                    f"preference training file not found: {dataset_path}"
                )
            metrics, quantization, fallback = _train_real(
                settings, preset, run_dir
            )
    except MissingTrainingDependency as exc:
        manifest = {
            "schema_version": 1,
            "training_type": "dpo",
            "status": "skipped",
            "run_id": identifier,
            "started_at": started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "reason": str(exc),
            "install": (
                "Install torch transformers datasets peft trl accelerate; add "
                "bitsandbytes for CUDA QLoRA, then rerun. --smoke needs none."
            ),
            "registry": None,
        }
        path = _write_manifest(run_dir, manifest)
        manifest["manifest_path"] = str(path)
        return manifest

    registered = None
    if settings.register_candidate:
        registered = _register_candidate(
            settings.registry_path,
            run_id=identifier,
            preset=preset,
            run_dir=run_dir,
            quantization=quantization,
            metrics=metrics,
            dataset_version=Path(dataset_path).name,
            smoke=smoke,
        )
        if registered["status"] != "candidate":
            raise RuntimeError("DPO registration must never set production status")
    manifest = {
        "schema_version": 1,
        "training_type": "dpo",
        "status": "completed",
        "run_id": identifier,
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "adapter_path": str(run_dir / "adapter"),
        "dataset_path": str(dataset_path),
        "preset": preset.to_dict(),
        "seed": settings.seed,
        "beta": settings.beta,
        "smoke": smoke,
        "quantization": quantization,
        "fallback_message": fallback,
        "license_metadata": license_metadata,
        "metrics": metrics,
        "registry": registered,
    }
    path = _write_manifest(run_dir, manifest)
    _write_metrics(run_dir / "metrics.csv", metrics)
    manifest["manifest_path"] = str(path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model")
    parser.add_argument("--preset", choices=list_presets(), default="rtx_4070_ti_3b_qlora")
    parser.add_argument("--train-file", default=str(DEFAULT_PREFERENCE_TRAIN))
    parser.add_argument("--eval-file", default=str(DEFAULT_PREFERENCE_VAL))
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint", nargs="?", const=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-register", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = train_dpo(
            DPOTrainingConfig(
                base_model=args.base_model,
                preset=args.preset,
                train_file=args.train_file,
                eval_file=args.eval_file,
                runs_root=args.runs_root,
                registry_path=args.registry_path,
                run_id=args.run_id,
                seed=args.seed,
                beta=args.beta,
                max_steps=args.max_steps,
                resume_from_checkpoint=args.resume_from_checkpoint,
                smoke=args.smoke,
                dry_run=args.dry_run,
                register_candidate=not args.no_register,
            )
        )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"planner DPO training failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PREFERENCE_TRAIN",
    "DEFAULT_PREFERENCE_VAL",
    "DPOTrainingConfig",
    "build_parser",
    "main",
    "train_dpo",
]
