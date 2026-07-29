#!/usr/bin/env python3
"""Owned-game loop: screen → vision LLM → act → learn."""

from __future__ import annotations

import argparse
from pathlib import Path

from playmind.owned_loop import OwnedGameLoop, OwnedLoopConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/owned_game.example.json")
    p.add_argument("--live", action="store_true")
    p.add_argument("--ollama", action="store_true", help="Use LLM brain (vision+text)")
    p.add_argument("--ollama-model", default="llama3.2", help="Text fallback model")
    p.add_argument("--vision-model", default="qwen2.5vl:7b", help="Ollama vision model that sees the frame")
    p.add_argument("--no-screen-llm", action="store_true", help="Text LLM only (no image)")
    p.add_argument("--max-ticks", type=int, default=5)
    p.add_argument("--directive", default="")
    p.add_argument("--learn", action="store_true", default=True)
    p.add_argument("--no-learn", action="store_true")
    p.add_argument("--learned", action="store_true")
    p.add_argument("--epsilon", type=float, default=0.12)
    args = p.parse_args()

    loop = OwnedGameLoop(
        cfg=OwnedLoopConfig(
            config_path=Path(args.config),
            dry_run=not args.live,
            use_ollama=args.ollama,
            ollama_model=args.ollama_model,
            vision_model=args.vision_model,
            use_screen_llm=not args.no_screen_llm,
            max_ticks=args.max_ticks,
            learn=not args.no_learn,
            use_learned_policy=args.learned,
            epsilon=args.epsilon,
        ),
        directive=args.directive or None,
    )
    loop.run()


if __name__ == "__main__":
    main()
