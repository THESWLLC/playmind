"""Evaluate a SkillPolicyV2 checkpoint on demonstration splits (torch optional)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from playmind.models.policy_v2 import (
    DEFAULT_FEATURE_DIM,
    SkillPolicyV2,
    structured_feature_vector,
)
from playmind.training.dataset import DemonstrationDataset
from playmind.training.train_behavior_clone import (
    confusion_matrix,
    print_confusion_matrix,
)


def _vectorize_item(item: Mapping[str, Any], feature_dim: int) -> list[float]:
    return structured_feature_vector(item.get("observation") or {}, feature_dim=feature_dim)


def precision_recall_per_skill(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for lab in labels:
        tp = fp = fn = 0
        for t, p in zip(y_true, y_pred):
            if p == lab and t == lab:
                tp += 1
            elif p == lab and t != lab:
                fp += 1
            elif t == lab and p != lab:
                fn += 1
        prec = tp / float(tp + fp) if (tp + fp) else 0.0
        rec = tp / float(tp + fn) if (tp + fn) else 0.0
        stats[lab] = {
            "precision": prec,
            "recall": rec,
            "support": float(sum(1 for t in y_true if t == lab)),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }
    return stats


def evaluate_behavior_clone(
    data_dir: Path | str,
    checkpoint: Path | str | None = None,
    *,
    split: str = "test",
    window_size: int = 4,
    seed: int = 0,
    policy: SkillPolicyV2 | None = None,
) -> dict[str, Any]:
    """Compute top-1 accuracy, per-skill P/R, and confusion matrix."""
    data_dir = Path(data_dir)
    ds = DemonstrationDataset(
        data_dir, window_size=window_size, split=split, seed=seed  # type: ignore[arg-type]
    )
    if policy is None:
        if checkpoint is None:
            policy = SkillPolicyV2()
        else:
            policy = SkillPolicyV2.load(checkpoint)

    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    for i in range(len(ds)):
        item = ds[i]
        skill_true = str(item.get("skill") or "wait")
        vec = _vectorize_item(item, policy.feature_dim)
        pred, conf, _aux = policy.predict(vec)
        y_true.append(skill_true)
        y_pred.append(str(pred))
        confidences.append(float(conf))

    labels = sorted(set(y_true) | set(y_pred) | set(policy.skill_names))
    n = len(y_true)
    top1 = (sum(1 for t, p in zip(y_true, y_pred) if t == p) / float(n)) if n else 0.0
    per_skill = precision_recall_per_skill(y_true, y_pred, labels)
    mat = confusion_matrix(y_true, y_pred, labels)

    report = {
        "split": split,
        "n_samples": n,
        "top1_accuracy": top1,
        "mean_confidence": (sum(confidences) / float(n)) if n else 0.0,
        "per_skill": per_skill,
        "confusion": mat,
        "labels": labels,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "model_version": getattr(policy, "model_version", None),
        "trained": bool(getattr(policy, "trained", False)),
        "dataset": ds.validate(),
    }
    return report


def format_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"split={report.get('split')} n={report.get('n_samples')}",
        f"top-1 accuracy={float(report.get('top1_accuracy') or 0):.4f}",
        f"mean confidence={float(report.get('mean_confidence') or 0):.4f}",
        f"trained={report.get('trained')} model_version={report.get('model_version')}",
        "per-skill precision/recall:",
    ]
    for lab, stats in (report.get("per_skill") or {}).items():
        lines.append(
            f"  {lab}: P={stats['precision']:.3f} R={stats['recall']:.3f} "
            f"support={int(stats['support'])}"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate SkillPolicyV2 behavior clone")
    p.add_argument("--data-dir", default="data/playmind/demonstrations")
    p.add_argument("--checkpoint", default="models/checkpoints/skill_policy_v2.json")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--window-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default=None, help="Optional path to write JSON report")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ckpt = Path(args.checkpoint)
    policy: SkillPolicyV2 | None
    if ckpt.exists():
        policy = SkillPolicyV2.load(ckpt)
    else:
        print(f"checkpoint not found ({ckpt}); evaluating untrained stub")
        policy = SkillPolicyV2()

    report = evaluate_behavior_clone(
        args.data_dir,
        checkpoint=ckpt if ckpt.exists() else None,
        split=args.split,
        window_size=args.window_size,
        seed=args.seed,
        policy=policy,
    )
    print(format_report(report))
    print_confusion_matrix(report["confusion"], report["labels"])
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Drop huge nested dataset detail duplication if needed — keep full for now.
        serializable = dict(report)
        with out.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, sort_keys=True)
            f.write("\n")
        print("Wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
