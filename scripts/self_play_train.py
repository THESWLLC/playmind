#!/usr/bin/env python3
"""Run many demo episodes so the online policy + finetune export grow on their own."""

from __future__ import annotations

import argparse
from pathlib import Path

from playmind.agent import AgentConfig, PlayMindAgent
from playmind.demo_world import DemoWorld


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--data-dir", default="data/playmind")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    wins = 0
    # Keep one policy across episodes so it actually accumulates learning.
    cfg = AgentConfig(learn=True, use_learned_policy=False, data_dir=data_dir)
    agent = PlayMindAgent(world=DemoWorld(), config=cfg)
    for ep in range(1, args.episodes + 1):
        agent.world = DemoWorld()
        done = False
        for _ in range(args.max_steps):
            result = agent.tick()
            if result["done"]:
                done = True
                break
        wins += int(done)
        if ep % 10 == 0:
            agent.save()
            print(f"ep={ep} wins={wins}/{ep} buffer={len(agent.buffer.rows)}")
    agent.save()
    print(f"Done. Wins {wins}/{args.episodes}. Data in {data_dir}")


if __name__ == "__main__":
    main()
