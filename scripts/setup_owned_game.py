#!/usr/bin/env python3
"""Copy owned_game example config and print the local calibration checklist.

Usage:
  python3 scripts/setup_owned_game.py
  python3 scripts/setup_owned_game.py --window-title "My Game"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Set up owned_game.json for local PlayMind")
    p.add_argument(
        "--example",
        default="config/owned_game.example.json",
        help="Source example config",
    )
    p.add_argument(
        "--dest",
        default="config/owned_game.json",
        help="Destination config (gitignored)",
    )
    p.add_argument("--window-title", default=None, help="Game window title substring")
    p.add_argument("--game-name", default=None)
    p.add_argument(
        "--bc-checkpoint",
        default="models/checkpoints/recurrent_skill_policy_v2.json",
        help="Path written into learning_v2.bc_checkpoint",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing dest")
    p.add_argument("--list-windows", action="store_true", help="List windows then exit")
    args = p.parse_args(argv)

    if args.list_windows:
        return subprocess_list_windows()

    example = Path(args.example)
    dest = Path(args.dest)
    if not example.exists():
        print(f"Missing example config: {example}", file=sys.stderr)
        return 1
    if dest.exists() and not args.force:
        print(f"Already exists: {dest} (pass --force to overwrite)")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(example, dest)
        print(f"Copied {example} → {dest}")

    cfg = json.loads(dest.read_text(encoding="utf-8"))
    if args.window_title:
        cfg.setdefault("capture", {})["window_title"] = args.window_title
    if args.game_name:
        cfg["game_name"] = args.game_name
    lv = cfg.setdefault("learning_v2", {})
    lv["enabled"] = True
    lv["policy_mode"] = lv.get("policy_mode") or "hybrid"
    lv["bc_checkpoint"] = args.bc_checkpoint
    dest.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Updated learning_v2.bc_checkpoint → {args.bc_checkpoint}")

    print(
        """
Local checklist (must be done on the machine that runs the game):
  [1] List windows:
        python3 scripts/capture_once.py --list-windows
  [2] Capture a frame:
        python3 scripts/capture_once.py --window "<title>" --out data/playmind/capture_sample.png
  [3] Edit rois in config/owned_game.json (hp_roi, target bar, etc.)
  [4] Edit config/keymap.example.json → your keys (or point keymap_path)
  [5] Dry-run (no keys):
        python3 scripts/run_owned_loop.py --config config/owned_game.json --max-ticks 20
  [6] Record real demos:
        python3 -m playmind.owned_gui
        # Advanced V2 → Start/Stop demo, mark success/failure
  [7] Train recurrent BC on REAL demos:
        python3 scripts/train_behavior_clone.py \\
          --data-dir data/playmind/demonstrations \\
          --checkpoint models/checkpoints/recurrent_skill_policy_v2.json \\
          --history-length 16 --model-type recurrent --device auto
  [8] Eval:
        python3 scripts/run_evaluation.py \\
          --data-dir data/playmind/demonstrations \\
          --checkpoints models/checkpoints/recurrent_skill_policy_v2.json \\
          --output-dir data/playmind/eval/latest
  [9] Live keys ONLY if you own the game:
        set i_own_this_game=true, enable_keyboard=true, then:
        python3 scripts/run_owned_loop.py --config config/owned_game.json --live --max-ticks 50

Optional pipeline smoke (synthetic demos, not real play):
  python3 scripts/smoke_local_pipeline.py --device cpu
"""
    )
    return 0


def subprocess_list_windows() -> int:
    import subprocess

    return subprocess.call(
        [sys.executable, str(ROOT / "scripts" / "capture_once.py"), "--list-windows"],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    raise SystemExit(main())
