"""Evaluate recurrent or legacy behavior-cloning checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from playmind.models.policy_v2 import (
    TORCH_AVAILABLE,
    SkillPolicyV2,
    structured_feature_vector,
)
from playmind.models.recurrent_policy import RecurrentSkillPolicyV2
from playmind.training.dataset import (
    DemonstrationDataset,
    assert_no_split_leakage,
)
from playmind.training.train_behavior_clone import confusion_matrix, print_confusion_matrix


def _vectorize_item(item: Mapping[str, Any], feature_dim: int) -> list[float]:
    return structured_feature_vector(item.get("observation") or {}, feature_dim=feature_dim)


def precision_recall_per_skill(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        stats[label] = {
            "precision": precision,
            "recall": recall,
            "f1": (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
            "support": float(sum(t == label for t in y_true)),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }
    return stats


def _load_policy(path: Path, device: str = "auto") -> Any:
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("model_type") == "recurrent_skill_policy":
        return RecurrentSkillPolicyV2.load(path, device=device)
    return SkillPolicyV2.load(path)


def _confidence_histogram(confidences: Sequence[float], bins: int = 10) -> list[int]:
    histogram = [0] * bins
    for confidence in confidences:
        index = min(bins - 1, max(0, int(float(confidence) * bins)))
        histogram[index] += 1
    return histogram


def _expected_calibration_error(
    truths: Sequence[str],
    predictions: Sequence[str],
    confidences: Sequence[float],
    bins: int = 10,
) -> float:
    """Simple fixed-bin ECE; retained as a replaceable calibration stub."""
    total = len(truths)
    if not total:
        return 0.0
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = [
            i
            for i, confidence in enumerate(confidences)
            if low <= confidence < high or (index == bins - 1 and confidence == 1.0)
        ]
        if not selected:
            continue
        accuracy = sum(truths[i] == predictions[i] for i in selected) / len(selected)
        mean_confidence = sum(confidences[i] for i in selected) / len(selected)
        ece += len(selected) / total * abs(accuracy - mean_confidence)
    return ece


def _predict_ranked(policy: Any, item: Mapping[str, Any]) -> tuple[list[str], float]:
    if (
        isinstance(policy, RecurrentSkillPolicyV2)
        and TORCH_AVAILABLE
        and policy.trained
        and policy._net is not None
    ):
        import torch

        logits, _aux = policy.predict_sequence(
            [item["features"]],
            lengths=[item["length"]],
            padding_mask=[item["padding_mask"]],
        )
        probabilities = torch.softmax(logits[0], dim=-1)
        order = torch.argsort(probabilities, descending=True).tolist()
        return [policy.skill_names[int(index)] for index in order], float(
            probabilities[int(order[0])].item()
        )
    if isinstance(policy, RecurrentSkillPolicyV2):
        prediction, confidence, _aux = policy.predict(item["features"])
    else:
        prediction, confidence, _aux = policy.predict(
            _vectorize_item(item, policy.feature_dim)
        )
    remaining = [skill for skill in policy.skill_names if skill != prediction]
    return [str(prediction), *remaining], float(confidence)


def evaluate_behavior_clone(
    data_dir: Path | str,
    checkpoint: Path | str | None = None,
    *,
    split: str = "test",
    history_length: int = 16,
    window_size: int | None = None,
    seed: int = 0,
    device: str = "auto",
    policy: Any = None,
) -> dict[str, Any]:
    """Compute ranking, calibration, per-skill, and sequence diagnostics."""
    data_dir = Path(data_dir)
    history_length = int(window_size if window_size is not None else history_length)
    split_datasets = {
        name: DemonstrationDataset(
            data_dir, history_length=history_length, split=name, seed=seed  # type: ignore[arg-type]
        )
        for name in ("train", "val", "test")
    }
    assert_no_split_leakage(*split_datasets.values())
    dataset = (
        DemonstrationDataset(
            data_dir, history_length=history_length, split="all", seed=seed
        )
        if split == "all"
        else split_datasets[split]
    )
    if policy is None:
        policy = (
            _load_policy(Path(checkpoint), device=device)
            if checkpoint is not None
            else RecurrentSkillPolicyV2(history_length=history_length, device=device)
        )

    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    top2_correct = top3_correct = 0
    for index in range(len(dataset)):
        item = dataset[index]
        truth = str(item.get("skill") or "wait")
        ranked, confidence = _predict_ranked(policy, item)
        prediction = ranked[0]
        y_true.append(truth)
        y_pred.append(prediction)
        confidences.append(confidence)
        top2_correct += truth in ranked[:2]
        top3_correct += truth in ranked[:3]

    labels = sorted(set(y_true) | set(y_pred) | set(policy.skill_names))
    count = len(y_true)
    report = {
        "split": split,
        "n_samples": count,
        "top1_accuracy": (
            sum(t == p for t, p in zip(y_true, y_pred)) / float(count) if count else 0.0
        ),
        "top2_accuracy": top2_correct / float(count) if count else 0.0,
        "top3_accuracy": top3_correct / float(count) if count else 0.0,
        "mean_confidence": sum(confidences) / float(count) if count else 0.0,
        "confidence_histogram": _confidence_histogram(confidences),
        "confidence_bin_edges": [index / 10.0 for index in range(11)],
        "ece": _expected_calibration_error(y_true, y_pred, confidences),
        "per_skill": precision_recall_per_skill(y_true, y_pred, labels),
        "confusion": confusion_matrix(y_true, y_pred, labels),
        "labels": labels,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "model_version": getattr(policy, "model_version", None),
        "model_type": getattr(policy, "metadata", {}).get("model_type"),
        "trained": bool(getattr(policy, "trained", False)),
        "dataset": dataset.validate(),
        "sequence_length_stats": dataset.validate()["sequence_lengths"],
        "leakage_check": {"passed": True, "shared_episode_ids": []},
    }
    return report


def format_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"split={report.get('split')} n={report.get('n_samples')}",
        f"top-1 accuracy={float(report.get('top1_accuracy') or 0):.4f}",
        f"top-2 accuracy={float(report.get('top2_accuracy') or 0):.4f}",
        f"top-3 accuracy={float(report.get('top3_accuracy') or 0):.4f}",
        f"mean confidence={float(report.get('mean_confidence') or 0):.4f}",
        f"ECE={float(report.get('ece') or 0):.4f}",
        f"trained={report.get('trained')} model_version={report.get('model_version')}",
        "per-skill precision/recall/F1:",
    ]
    for label, stats in (report.get("per_skill") or {}).items():
        lines.append(
            f"  {label}: P={stats['precision']:.3f} R={stats['recall']:.3f} "
            f"F1={stats['f1']:.3f} support={int(stats['support'])}"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PlayMind behavior clone")
    parser.add_argument("--data-dir", default="data/playmind/demonstrations")
    parser.add_argument("--checkpoint", default="models/checkpoints/recurrent_skill_policy.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--history-length", "--window-size", dest="history_length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-out", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        policy = _load_policy(checkpoint, device=args.device)
    else:
        print(f"checkpoint not found ({checkpoint}); evaluating untrained recurrent stub")
        policy = RecurrentSkillPolicyV2(
            history_length=args.history_length, device=args.device
        )
    report = evaluate_behavior_clone(
        args.data_dir,
        checkpoint=checkpoint if checkpoint.exists() else None,
        split=args.split,
        history_length=args.history_length,
        seed=args.seed,
        device=args.device,
        policy=policy,
    )
    print(format_report(report))
    print_confusion_matrix(report["confusion"], report["labels"])
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("Wrote", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
