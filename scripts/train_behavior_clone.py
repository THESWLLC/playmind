#!/usr/bin/env python3
"""Behavior-cloning training CLI for Learning Architecture V2.

Works without CUDA / game. If torch is missing: dry-validates the dataset,
prints install instructions, and exits 0. If torch is present: runs a minimal
training-loop skeleton on DemonstrationDataset batches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train SkillPolicyV2 behavior clone (stub)")
    p.add_argument(
        "--data-dir",
        default="data/playmind/demonstrations",
        help="Root directory of demonstration sessions",
    )
    p.add_argument("--window-size", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    p.add_argument(
        "--checkpoint",
        default="models/checkpoints/skill_policy_v2.json",
        help="Output checkpoint metadata JSON path",
    )
    p.add_argument(
        "--dry-validate-only",
        action="store_true",
        help="Only validate dataset even if torch is present",
    )
    return p


def dry_validate(data_dir: Path, window_size: int, split: str) -> dict:
    from playmind.training.dataset import DemonstrationDataset

    ds = DemonstrationDataset(data_dir, window_size=window_size, split=split)  # type: ignore[arg-type]
    summary = ds.validate()
    print("DemonstrationDataset validation:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if summary["windows"] == 0:
        print("  note: no windows found — record demos under", data_dir)
    return summary


def train_skeleton(
    data_dir: Path,
    *,
    window_size: int,
    batch_size: int,
    epochs: int,
    split: str,
    checkpoint: Path,
) -> None:
    import torch
    import torch.nn.functional as F

    from playmind.models.policy_v2 import SkillPolicyV2, TORCH_AVAILABLE
    from playmind.training.dataset import DemonstrationDataset

    assert TORCH_AVAILABLE
    ds = DemonstrationDataset(data_dir, window_size=window_size, split=split)  # type: ignore[arg-type]
    summary = ds.validate()
    print("Training skeleton on", summary["windows"], "windows")

    # Collect skill vocabulary from labeled windows.
    skills = sorted({s for s in (w.get("skill") for w in (ds[i] for i in range(len(ds)))) if s})
    if not skills:
        skills = list(SkillPolicyV2().skill_names)
        print("No skill labels in data; using default skill list for scaffold.")

    policy = SkillPolicyV2(skill_names=skills, feature_dim=32, trained=False)
    if policy._net is None:
        print("Torch net unavailable; saving metadata only.")
        policy.save(checkpoint)
        return

    skill_to_idx = {s: i for i, s in enumerate(policy.skill_names)}
    opt = torch.optim.Adam(policy._net.parameters(), lr=1e-3)
    policy._net.train()

    for epoch in range(max(1, epochs)):
        total_loss = 0.0
        n_batches = 0
        for batch in ds.iter_batches(batch_size):
            # Use last-frame feature vector; pad to feature_dim.
            vecs = []
            labels = []
            for feat, skill in zip(batch["feature"], batch["skill"]):
                v = list(feat)[: policy.feature_dim]
                if len(v) < policy.feature_dim:
                    v = v + [0.0] * (policy.feature_dim - len(v))
                vecs.append(v)
                labels.append(skill_to_idx.get(skill or "wait", 0))
            if not vecs:
                continue
            x = torch.tensor(vecs, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.long)
            logits = policy._net(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        avg = total_loss / max(1, n_batches)
        print(f"epoch={epoch + 1}/{epochs} loss={avg:.4f} batches={n_batches}")

    policy.trained = True
    policy.metadata["trained"] = True
    out = policy.save(checkpoint)
    print("Wrote checkpoint", out)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)

    try:
        from playmind.models.policy_v2 import TORCH_AVAILABLE, torch_install_instructions
    except Exception as exc:  # pragma: no cover
        print("Failed to import policy stub:", exc, file=sys.stderr)
        return 1

    summary = dry_validate(data_dir, args.window_size, args.split)

    if not TORCH_AVAILABLE or args.dry_validate_only:
        if not TORCH_AVAILABLE:
            print()
            print(torch_install_instructions())
            print()
            print("Exiting 0 after dry validation (torch not required to inspect demos).")
        else:
            print("Dry validation only; skipping training loop.")
        print(f"validated_windows={summary['windows']}")
        return 0

    train_skeleton(
        data_dir,
        window_size=args.window_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        split=args.split,
        checkpoint=Path(args.checkpoint),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
