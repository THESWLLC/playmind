"""Behavior-cloning training for SkillPolicyV2 (torch optional).

Episode-wise train/val splits via DemonstrationDataset. Without torch: validates
the dataset only. With torch: trains MLP with early stopping, CSV metrics, and
confusion-matrix printing.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from playmind.models.policy_v2 import (
    AUX_KEYS,
    DEFAULT_FEATURE_DIM,
    DEFAULT_SKILLS,
    TORCH_AVAILABLE,
    SkillPolicyV2,
    structured_feature_vector,
    torch_install_instructions,
)
from playmind.training.dataset import DemonstrationDataset

DEFAULT_METRICS_CSV = Path("data/playmind/training/metrics.csv")
DEFAULT_CHECKPOINT = Path("models/checkpoints/skill_policy_v2.json")


def _skill_vocab(ds: DemonstrationDataset, fallback: Sequence[str] | None = None) -> list[str]:
    skills = sorted({s for s in (ds[i].get("skill") for i in range(len(ds))) if s})
    if skills:
        return skills
    return list(fallback or DEFAULT_SKILLS)


def _vectorize_item(item: Mapping[str, Any], feature_dim: int) -> list[float]:
    obs = item.get("observation") or {}
    # Prefer structured Observation/TemporalSummary layout.
    return structured_feature_vector(obs, feature_dim=feature_dim)


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> list[list[int]]:
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    mat = [[0 for _ in range(n)] for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t not in idx or p not in idx:
            continue
        mat[idx[t]][idx[p]] += 1
    return mat


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
    width = max(8, max((len(str(l)) for l in labels), default=4) + 1)
    header = " " * width + "".join(f"{str(l)[: width - 1]:>{width}}" for l in labels)
    print(header)
    for i, row in enumerate(mat):
        lab = str(labels[i])[: width - 1]
        cells = "".join(f"{int(v):>{width}}" for v in row)
        print(f"{lab:>{width}}{cells}")


def append_metrics_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row[k] for k in fieldnames})


def dry_validate(
    data_dir: Path,
    *,
    window_size: int = 4,
    split: str = "train",
    seed: int = 0,
) -> dict[str, Any]:
    ds = DemonstrationDataset(
        data_dir, window_size=window_size, split=split, seed=seed  # type: ignore[arg-type]
    )
    summary = ds.validate()
    print("DemonstrationDataset validation:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if summary["windows"] == 0:
        print("  note: no windows found — record demos under", data_dir)
    return summary


def _eval_accuracy(
    policy: SkillPolicyV2,
    ds: DemonstrationDataset,
) -> tuple[float, list[str], list[str]]:
    if len(ds) == 0:
        return 0.0, [], []
    y_true: list[str] = []
    y_pred: list[str] = []
    correct = 0
    for i in range(len(ds)):
        item = ds[i]
        skill_true = item.get("skill") or "wait"
        vec = _vectorize_item(item, policy.feature_dim)
        pred, _conf, _aux = policy.predict(vec)
        y_true.append(str(skill_true))
        y_pred.append(str(pred))
        if pred == skill_true:
            correct += 1
    return correct / float(len(ds)), y_true, y_pred


def train_behavior_clone(
    data_dir: Path | str,
    *,
    window_size: int = 4,
    batch_size: int = 8,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 0,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    metrics_csv: Path | str = DEFAULT_METRICS_CSV,
    hidden: int = 64,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    dry_validate_only: bool = False,
) -> dict[str, Any]:
    """Train or dry-validate. Returns a result summary dict."""
    data_dir = Path(data_dir)
    checkpoint = Path(checkpoint)
    metrics_csv = Path(metrics_csv)

    train_ds = DemonstrationDataset(
        data_dir, window_size=window_size, split="train", seed=seed
    )
    val_ds = DemonstrationDataset(
        data_dir, window_size=window_size, split="val", seed=seed
    )
    train_summary = train_ds.validate()
    val_summary = val_ds.validate()
    print("Episode-wise split:")
    print(f"  train windows={train_summary['windows']} episodes={train_summary['episode_splits']}")
    print(f"  val   windows={val_summary['windows']}")

    if not TORCH_AVAILABLE or dry_validate_only:
        if not TORCH_AVAILABLE:
            print()
            print(torch_install_instructions())
            print()
            print("Exiting after dry validation (torch not required to inspect demos).")
        else:
            print("Dry validation only; skipping training loop.")
        return {
            "ok": True,
            "trained": False,
            "torch_available": TORCH_AVAILABLE,
            "train": train_summary,
            "val": val_summary,
        }

    import torch
    import torch.nn.functional as F

    skills = _skill_vocab(train_ds)
    if not any(train_ds[i].get("skill") for i in range(len(train_ds))):
        print("No skill labels in train split; using default skill list for scaffold.")

    policy = SkillPolicyV2(
        skill_names=skills,
        feature_dim=feature_dim,
        hidden=hidden,
        trained=False,
        config={
            "window_size": window_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "patience": patience,
            "seed": seed,
            "data_dir": str(data_dir),
        },
    )
    if policy._net is None:
        print("Torch net unavailable; saving metadata only.")
        out = policy.save(checkpoint)
        return {"ok": True, "trained": False, "checkpoint": str(out)}

    skill_to_idx = {s: i for i, s in enumerate(policy.skill_names)}
    opt = torch.optim.Adam(policy._net.parameters(), lr=lr)
    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    epochs_without_improve = 0
    history: list[dict[str, Any]] = []

    def _run_epoch(ds: DemonstrationDataset, *, train: bool) -> float:
        if train:
            policy._net.train()
        else:
            policy._net.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in ds.iter_batches(batch_size):
            vecs: list[list[float]] = []
            labels: list[int] = []
            for feat_obs, skill in zip(batch["observation"], batch["skill"]):
                item = {"observation": feat_obs, "skill": skill}
                vecs.append(_vectorize_item(item, policy.feature_dim))
                labels.append(skill_to_idx.get(skill or "wait", 0))
            if not vecs:
                continue
            x = torch.tensor(vecs, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.long)
            if train:
                logits, _aux = policy._net(x)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                with torch.no_grad():
                    logits, _aux = policy._net(x)
                    loss = F.cross_entropy(logits, y)
            total_loss += float(loss.item())
            n_batches += 1
        return total_loss / max(1, n_batches)

    for epoch in range(max(1, epochs)):
        train_loss = _run_epoch(train_ds, train=True)
        # Temporarily mark trained so eval path uses the net.
        policy.trained = True
        val_loss = _run_epoch(val_ds, train=False) if len(val_ds) else train_loss
        val_acc, y_true, y_pred = _eval_accuracy(policy, val_ds) if len(val_ds) else (0.0, [], [])
        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "n_skills": len(policy.skill_names),
            "patience": patience,
            "seed": seed,
        }
        append_metrics_csv(metrics_csv, row)
        history.append(row)
        print(
            f"epoch={epoch + 1}/{epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in policy._net.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= max(1, patience):
                print(f"Early stopping at epoch {epoch + 1} (patience={patience})")
                break

    if best_state is not None:
        policy._net.load_state_dict(best_state)
    policy.trained = True
    policy.metadata["trained"] = True
    out = policy.save(
        checkpoint,
        config_snapshot={
            "best_val_loss": best_val,
            "epochs_ran": len(history),
            "metrics_csv": str(metrics_csv),
        },
    )
    print("Wrote checkpoint", out)
    print("Wrote metrics", metrics_csv)

    # Final confusion on val (or train if val empty).
    eval_ds = val_ds if len(val_ds) else train_ds
    _acc, y_true, y_pred = _eval_accuracy(policy, eval_ds)
    labels = list(policy.skill_names)
    mat = confusion_matrix(y_true, y_pred, labels)
    print_confusion_matrix(mat, labels)

    return {
        "ok": True,
        "trained": True,
        "checkpoint": str(out),
        "metrics_csv": str(metrics_csv),
        "best_val_loss": best_val,
        "history": history,
        "confusion": mat,
        "labels": labels,
        "val_acc": _acc,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train SkillPolicyV2 behavior clone")
    p.add_argument(
        "--data-dir",
        default="data/playmind/demonstrations",
        help="Root directory of demonstration sessions",
    )
    p.add_argument("--window-size", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5, help="Early-stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--feature-dim", type=int, default=DEFAULT_FEATURE_DIM)
    p.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    p.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Output checkpoint metadata JSON path",
    )
    p.add_argument(
        "--metrics-csv",
        default=str(DEFAULT_METRICS_CSV),
        help="Append per-epoch metrics CSV",
    )
    p.add_argument(
        "--dry-validate-only",
        action="store_true",
        help="Only validate dataset even if torch is present",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    data_dir = Path(args.data_dir)

    # Always print a single-split dry validation first for CLI visibility.
    summary = dry_validate(
        data_dir, window_size=args.window_size, split=args.split, seed=args.seed
    )

    result = train_behavior_clone(
        data_dir,
        window_size=args.window_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        checkpoint=args.checkpoint,
        metrics_csv=args.metrics_csv,
        hidden=args.hidden,
        feature_dim=args.feature_dim,
        dry_validate_only=args.dry_validate_only or not TORCH_AVAILABLE,
    )
    print(f"validated_windows={summary['windows']}")
    if not result.get("trained"):
        return 0
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
