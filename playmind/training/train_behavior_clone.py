"""Train recurrent (default) or legacy MLP behavior-cloning policies."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from playmind.models.feature_schema import FEATURE_DIM
from playmind.models.policy_v2 import (
    DEFAULT_FEATURE_DIM,
    DEFAULT_SKILLS,
    TORCH_AVAILABLE,
    SkillPolicyV2,
    structured_feature_vector,
    torch_install_instructions,
)
from playmind.models.recurrent_policy import (
    AUX_TYPES,
    DEFAULT_AUX_KEYS,
    RecurrentSkillPolicyV2,
    seed_everything,
)
from playmind.training.dataset import (
    DemonstrationDataset,
    assert_no_split_leakage,
)

DEFAULT_METRICS_CSV = Path("data/playmind/training/metrics.csv")
DEFAULT_CHECKPOINT = Path("models/checkpoints/recurrent_skill_policy.json")


def _skill_vocab(dataset: DemonstrationDataset, fallback: Sequence[str] | None = None) -> list[str]:
    skills = sorted(dataset.skill_counts)
    return skills or list(fallback or DEFAULT_SKILLS)


def _vectorize_item(item: Mapping[str, Any], feature_dim: int) -> list[float]:
    return structured_feature_vector(item.get("observation") or {}, feature_dim=feature_dim)


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> list[list[int]]:
    indices = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, prediction in zip(y_true, y_pred):
        if truth in indices and prediction in indices:
            matrix[indices[truth]][indices[prediction]] += 1
    return matrix


def print_confusion_matrix(
    mat: Sequence[Sequence[int]],
    labels: Sequence[str],
    *,
    title: str = "Confusion matrix (rows=true, cols=pred)",
) -> None:
    print(title)
    if not labels:
        print("  (empty)")
        return
    width = max(8, max(len(str(label)) for label in labels) + 1)
    print(" " * width + "".join(f"{str(label)[:width - 1]:>{width}}" for label in labels))
    for label, row in zip(labels, mat):
        print(f"{str(label)[:width - 1]:>{width}}" + "".join(f"{int(v):>{width}}" for v in row))


def append_metrics_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(dict(row))


def dry_validate(
    data_dir: Path,
    *,
    window_size: int = 16,
    history_length: int | None = None,
    split: str = "train",
    seed: int = 0,
) -> dict[str, Any]:
    length = int(history_length if history_length is not None else window_size)
    dataset = DemonstrationDataset(
        data_dir, history_length=length, split=split, seed=seed  # type: ignore[arg-type]
    )
    summary = dataset.validate()
    print("DemonstrationDataset validation:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if not summary["windows"]:
        print("  note: no windows found — record labeled demos under", data_dir)
    return summary


def _evaluate_top1(policy: Any, dataset: DemonstrationDataset) -> tuple[float, list[str], list[str]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    for index in range(len(dataset)):
        item = dataset[index]
        truth = str(item.get("skill") or "wait")
        if isinstance(policy, RecurrentSkillPolicyV2):
            prediction, _confidence, _aux = policy.predict(item["features"])
        else:
            prediction, _confidence, _aux = policy.predict(
                _vectorize_item(item, policy.feature_dim)
            )
        y_true.append(truth)
        y_pred.append(str(prediction))
    accuracy = (
        sum(a == b for a, b in zip(y_true, y_pred)) / float(len(y_true))
        if y_true
        else 0.0
    )
    return accuracy, y_true, y_pred


def train_behavior_clone(
    data_dir: Path | str,
    *,
    history_length: int = 16,
    window_size: int | None = None,
    stride: int = 1,
    min_sequence_length: int = 1,
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 0,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    metrics_csv: Path | str = DEFAULT_METRICS_CSV,
    model_type: str = "recurrent",
    hidden: int = 128,
    encoder_dim: int = 96,
    num_layers: int = 1,
    dropout: float = 0.0,
    feature_dim: int | None = None,
    device: str = "auto",
    amp: bool = True,
    grad_clip: float = 1.0,
    aux_loss_weight: float = 0.2,
    aux_loss_weights: Mapping[str, float] | None = None,
    class_weights: bool = True,
    resume: bool = False,
    dry_validate_only: bool = False,
) -> dict[str, Any]:
    """Train with validation selection, then evaluate the untouched test split."""
    data_dir = Path(data_dir)
    checkpoint = Path(checkpoint)
    metrics_csv = Path(metrics_csv)
    history_length = int(window_size if window_size is not None else history_length)
    dataset_args = {
        "history_length": history_length,
        "stride": stride,
        "min_sequence_length": min_sequence_length,
        "seed": seed,
    }
    train_ds = DemonstrationDataset(data_dir, split="train", **dataset_args)
    val_ds = DemonstrationDataset(data_dir, split="val", **dataset_args)
    test_ds = DemonstrationDataset(data_dir, split="test", **dataset_args)
    assert_no_split_leakage(train_ds, val_ds, test_ds)
    train_summary, val_summary, test_summary = (
        train_ds.validate(),
        val_ds.validate(),
        test_ds.validate(),
    )
    print("Episode-wise split:")
    print(f"  train windows={len(train_ds)} val windows={len(val_ds)} test windows={len(test_ds)}")

    common_result = {
        "trained": False,
        "torch_available": TORCH_AVAILABLE,
        "train": train_summary,
        "val": val_summary,
        "test": test_summary,
    }
    if dry_validate_only:
        print("Dry validation only; skipping training loop.")
        return {"ok": True, **common_result}
    if not TORCH_AVAILABLE:
        print(torch_install_instructions())
        return {"ok": False, "error": "torch_not_installed", **common_result}
    if not len(train_ds):
        print("No labeled windows in the training split.")
        return {"ok": False, "error": "empty_training_split", **common_result}

    import torch
    import torch.nn.functional as F

    seed_everything(seed)
    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available()
        else "cpu" if device == "auto"
        else device
    )
    if str(resolved_device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {resolved_device}")
    skills = _skill_vocab(train_ds)
    training_config = {
        **dataset_args,
        "data_dir": str(data_dir),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "patience": patience,
        "model_type": model_type,
        "device": resolved_device,
        "amp": bool(amp),
        "grad_clip": grad_clip,
        "class_weights": class_weights,
    }
    if resume and checkpoint.with_suffix(".json").exists():
        policy: Any = (
            RecurrentSkillPolicyV2.load(checkpoint, device=resolved_device)
            if model_type == "recurrent"
            else SkillPolicyV2.load(checkpoint)
        )
    elif model_type == "recurrent":
        recurrent_dim = int(feature_dim or FEATURE_DIM)
        if recurrent_dim != FEATURE_DIM:
            raise ValueError(f"recurrent feature_dim must be schema-v2 FEATURE_DIM={FEATURE_DIM}")
        policy = RecurrentSkillPolicyV2(
            skill_names=skills,
            feature_dim=recurrent_dim,
            encoder_dim=encoder_dim,
            hidden_dim=hidden,
            num_layers=num_layers,
            dropout=dropout,
            history_length=history_length,
            normalizer=train_ds.fit_normalizer(),
            seed=seed,
            training_config=training_config,
            device=resolved_device,
        )
    elif model_type == "mlp":
        policy = SkillPolicyV2(
            skill_names=skills,
            feature_dim=int(feature_dim or DEFAULT_FEATURE_DIM),
            hidden=hidden,
            config=training_config,
        )
        policy._net.to(resolved_device)
    else:
        raise ValueError("model_type must be 'recurrent' or 'mlp'")
    if policy._net is None:
        return {"ok": False, "error": "torch_net_unavailable", **common_result}
    policy._net.to(resolved_device)

    skill_to_idx = {skill: index for index, skill in enumerate(policy.skill_names)}
    counts = Counter(
        str(train_ds[index]["skill"])
        for index in range(len(train_ds))
        if train_ds[index].get("skill") in skill_to_idx
    )
    weight_tensor = None
    if class_weights:
        total = float(sum(counts.values()))
        weight_tensor = torch.tensor(
            [
                total / max(1.0, len(counts) * counts.get(skill, 1))
                for skill in policy.skill_names
            ],
            dtype=torch.float32,
            device=resolved_device,
        )
    optimizer = torch.optim.AdamW(policy._net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 2)
    )
    use_amp = bool(amp and str(resolved_device).startswith("cuda"))
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:  # pragma: no cover - compatibility with older supported torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch = 0
    if resume and getattr(policy, "training_state", None):
        state = policy.training_state
        if state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state.get("epoch", 0))

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(metrics_csv.parent / "tensorboard"))
    except ImportError:
        pass

    configured_aux_weights = {
        name: float((aux_loss_weights or {}).get(name, aux_loss_weight))
        for name in DEFAULT_AUX_KEYS
    }
    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    no_improvement = 0
    history: list[dict[str, Any]] = []

    def run_epoch(dataset: DemonstrationDataset, training: bool, epoch: int) -> tuple[float, float]:
        policy._net.train(training)
        order = list(range(len(dataset)))
        if training:
            random.Random(seed + epoch).shuffle(order)
        total_loss = total_correct = total_items = batches = 0
        for offset in range(0, len(order), max(1, batch_size)):
            items = [dataset[index] for index in order[offset : offset + batch_size]]
            if not items:
                continue
            targets = torch.tensor(
                [skill_to_idx.get(str(item.get("skill")), 0) for item in items],
                dtype=torch.long,
                device=resolved_device,
            )
            sample_weights = torch.tensor(
                [float(item.get("sample_weight", 1.0)) for item in items],
                dtype=torch.float32,
                device=resolved_device,
            )
            if model_type == "recurrent":
                features = torch.tensor(
                    [item["features"] for item in items],
                    dtype=torch.float32,
                    device=resolved_device,
                )
                lengths = torch.tensor(
                    [item["length"] for item in items], dtype=torch.long, device=resolved_device
                )
                masks = torch.tensor(
                    [item["padding_mask"] for item in items],
                    dtype=torch.bool,
                    device=resolved_device,
                )
                features = policy._normalize_tensor(features, masks)
            else:
                features = torch.tensor(
                    [_vectorize_item(item, policy.feature_dim) for item in items],
                    dtype=torch.float32,
                    device=resolved_device,
                )
                lengths = masks = None
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_amp
                ):
                    if model_type == "recurrent":
                        logits, aux_outputs = policy._net(
                            features, lengths=lengths, padding_mask=masks
                        )
                    else:
                        logits, legacy_aux = policy._net(features)
                        aux_outputs = {
                            name: legacy_aux[:, index]
                            for index, name in enumerate(("target_valid", "combat", "death"))
                        }
                    ce = F.cross_entropy(
                        logits, targets, weight=weight_tensor, reduction="none"
                    )
                    loss = (ce * sample_weights).mean()
                    for name, output in aux_outputs.items():
                        values = [item["aux_targets"].get(name) for item in items]
                        valid = torch.tensor(
                            [value is not None for value in values],
                            dtype=torch.bool,
                            device=resolved_device,
                        )
                        if not bool(valid.any()):
                            continue
                        aux_target = torch.tensor(
                            [float(value or 0.0) for value in values],
                            dtype=torch.float32,
                            device=resolved_device,
                        )
                        if AUX_TYPES.get(name) == "binary":
                            aux_loss = F.binary_cross_entropy_with_logits(
                                output[valid], aux_target[valid]
                            )
                        else:
                            aux_loss = F.smooth_l1_loss(output[valid], aux_target[valid])
                        loss = loss + configured_aux_weights.get(name, aux_loss_weight) * aux_loss
                if training:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(policy._net.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
            total_loss += float(loss.detach().item())
            total_correct += int((logits.argmax(dim=-1) == targets).sum().item())
            total_items += len(items)
            batches += 1
        return total_loss / max(1, batches), total_correct / max(1, total_items)

    for epoch in range(start_epoch, max(start_epoch + 1, epochs)):
        train_loss, train_acc = run_epoch(train_ds, True, epoch)
        selection_ds = val_ds if len(val_ds) else train_ds
        val_loss, val_acc = run_epoch(selection_ds, False, epoch)
        scheduler.step(val_loss)
        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr": optimizer.param_groups[0]["lr"],
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "n_skills": len(policy.skill_names),
            "seed": seed,
        }
        append_metrics_csv(metrics_csv, row)
        history.append(row)
        if writer is not None:
            for key in ("train_loss", "train_acc", "val_loss", "val_acc", "lr"):
                writer.add_scalar(key, row[key], epoch + 1)
        print(
            f"epoch={epoch + 1}/{epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in policy._net.state_dict().items()
            }
            no_improvement = 0
            policy.trained = True
            training_state = {
                "epoch": epoch + 1,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_loss": best_val,
            }
            if isinstance(policy, RecurrentSkillPolicyV2):
                policy.save(
                    checkpoint,
                    training_config={"best_val_loss": best_val, "epochs_ran": epoch + 1},
                    training_state=training_state,
                )
            else:
                policy.save(
                    checkpoint,
                    config_snapshot={"best_val_loss": best_val, "epochs_ran": epoch + 1},
                )
        else:
            no_improvement += 1
            if no_improvement >= max(1, patience):
                print(f"Early stopping at epoch {epoch + 1}")
                break
    if writer is not None:
        writer.close()
    if best_state is not None:
        policy._net.load_state_dict(best_state)
    policy.trained = True
    if isinstance(policy, RecurrentSkillPolicyV2):
        out = policy.save(
            checkpoint,
            training_config={
                "best_val_loss": best_val,
                "epochs_ran": len(history),
                "metrics_csv": str(metrics_csv),
            },
            training_state=getattr(policy, "training_state", {}),
        )
    else:
        out = policy.save(
            checkpoint,
            config_snapshot={
                "best_val_loss": best_val,
                "epochs_ran": len(history),
                "metrics_csv": str(metrics_csv),
            },
        )

    # Test is touched only after validation-based model selection.
    if model_type == "mlp":
        # The legacy inference wrapper creates CPU tensors.
        policy._net.to("cpu")
    test_acc, test_true, test_pred = _evaluate_top1(policy, test_ds)
    val_acc, val_true, val_pred = _evaluate_top1(policy, val_ds)
    labels = list(policy.skill_names)
    matrix = confusion_matrix(test_true, test_pred, labels)
    print_confusion_matrix(matrix, labels, title="Test confusion matrix (rows=true, cols=pred)")
    return {
        "ok": True,
        "trained": True,
        "checkpoint": str(out),
        "metrics_csv": str(metrics_csv),
        "best_val_loss": best_val,
        "history": history,
        "confusion": matrix,
        "labels": labels,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "val_confusion": confusion_matrix(val_true, val_pred, labels),
        "model_type": model_type,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PlayMind behavior clone")
    parser.add_argument("--data-dir", default="data/playmind/demonstrations")
    parser.add_argument("--history-length", "--window-size", dest="history_length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-sequence-length", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--encoder-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--model-type", choices=["recurrent", "mlp"], default="recurrent")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--metrics-csv", default=str(DEFAULT_METRICS_CSV))
    parser.add_argument("--dry-validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    summary = dry_validate(
        data_dir,
        history_length=args.history_length,
        split=args.split,
        seed=args.seed,
    )
    result = train_behavior_clone(
        data_dir,
        history_length=args.history_length,
        stride=args.stride,
        min_sequence_length=args.min_sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        checkpoint=args.checkpoint,
        metrics_csv=args.metrics_csv,
        model_type=args.model_type,
        hidden=args.hidden,
        encoder_dim=args.encoder_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        feature_dim=args.feature_dim,
        device=args.device,
        amp=not args.no_amp,
        grad_clip=args.grad_clip,
        resume=args.resume,
        dry_validate_only=args.dry_validate_only,
    )
    print(f"validated_windows={summary['windows']}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
