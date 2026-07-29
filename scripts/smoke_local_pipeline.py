#!/usr/bin/env python3
"""End-to-end local smoke: synthetic demos → recurrent train → eval.

This does NOT replace calibrating your game or recording real demos.
It proves the learning pipeline runs on this machine.

Usage:
  python3 scripts/smoke_local_pipeline.py
  python3 scripts/smoke_local_pipeline.py --epochs 3 --device cpu
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Smoke-test PlayMind local learning pipeline")
    p.add_argument("--demo-dir", default="data/playmind/demonstrations")
    p.add_argument(
        "--checkpoint",
        default="models/checkpoints/recurrent_skill_policy_v2.json",
    )
    p.add_argument("--eval-dir", default="data/playmind/eval/smoke")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--history-length", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--sessions", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args(argv)

    py = sys.executable
    demo_dir = Path(args.demo_dir)
    ckpt = Path(args.checkpoint)
    eval_dir = Path(args.eval_dir)

    if not args.skip_generate:
        rc = _run(
            [
                py,
                "scripts/generate_synthetic_demos.py",
                "--out-dir",
                str(demo_dir),
                "--sessions",
                str(args.sessions),
                "--episodes-per-session",
                "2",
                "--steps",
                str(args.steps),
            ]
        )
        if rc != 0:
            return rc

    if not args.skip_train:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        rc = _run(
            [
                py,
                "scripts/train_behavior_clone.py",
                "--data-dir",
                str(demo_dir),
                "--checkpoint",
                str(ckpt),
                "--history-length",
                str(args.history_length),
                "--model-type",
                "recurrent",
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                "8",
                "--patience",
                "3",
            ]
        )
        if rc != 0:
            return rc

    if not args.skip_eval:
        eval_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            py,
            "scripts/run_evaluation.py",
            "--data-dir",
            str(demo_dir),
            "--compare-scripted",
            "--output-dir",
            str(eval_dir),
        ]
        if ckpt.exists():
            cmd.extend(["--checkpoints", str(ckpt)])
        rc = _run(cmd)
        if rc != 0:
            return rc

    snippet = {
        "learning_v2": {
            "enabled": True,
            "policy_mode": "hybrid",
            "bc_checkpoint": str(ckpt).replace("\\", "/"),
            "history_length": args.history_length,
            "confidence_threshold": 0.45,
        }
    }
    print("\n=== SMOKE PIPELINE COMPLETE ===")
    print("Synthetic demos:", demo_dir)
    print("Checkpoint:", ckpt if ckpt.exists() else "(missing)")
    print("Eval reports:", eval_dir)
    print("\nPaste into config/owned_game.json (after real calibration):")
    print(json.dumps(snippet, indent=2))
    print(
        "\nStill required on YOUR PC:\n"
        "  1) Set capture.window_title + rois + keymap for your game\n"
        "  2) Record real demos in owned GUI (replace synthetic)\n"
        "  3) Re-train on real demos\n"
        "  4) Dry-run, then --live only with ownership gates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
