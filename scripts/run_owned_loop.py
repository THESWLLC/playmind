#!/usr/bin/env python3
"""Run owned-game capture→vision→act loop (dry-run by default)."""

from __future__ import annotations

import argparse
from pathlib import Path

from playmind.owned_loop import OwnedGameLoop, OwnedLoopConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/owned_game.example.json")
    p.add_argument("--live", action="store_true", help="Allow keyboard if config enables it")
    p.add_argument("--ollama", action="store_true")
    p.add_argument("--max-ticks", type=int, default=5)
    p.add_argument("--directive", default="")
    args = p.parse_args()

    loop = OwnedGameLoop(
        cfg=OwnedLoopConfig(
            config_path=Path(args.config),
            dry_run=not args.live,
            use_ollama=args.ollama,
            max_ticks=args.max_ticks,
        ),
        directive=args.directive or None,
    )
    loop.run()


if __name__ == "__main__":
    main()
