"""CLI: python -m playmind [--learn] [--teach] [--ollama] [--episodes N]."""

from __future__ import annotations

import argparse
import time

from playmind.agent import AgentConfig, PlayMindAgent
from playmind.demo_world import ACTIONS, DemoWorld


def run_episode(agent: PlayMindAgent, max_steps: int, interactive: bool) -> bool:
    for _ in range(max_steps):
        obs = agent.observe()
        action = agent.propose_action(obs)
        question = agent.maybe_ask(obs, action)
        if question and interactive:
            print(agent.world.render_ascii())
            print(question)
            ans = input("> ").strip()
            if ans:
                if ans.lower() in ACTIONS:
                    agent.answer_teach(ans.lower(), obs)
                    action = ans.lower()
                elif ans.lower().startswith("dir "):
                    agent.set_directive(ans[4:])
        result = agent.tick(action)
        if interactive:
            print(agent.world.render_ascii())
            print(
                f"action={result['action']} reward={result['reward']:.2f} "
                f"kills={result['obs']['quest_kills']} done={result['done']}"
            )
            time.sleep(0.02)
        if result["done"]:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="PlayMind demo agent (owned games only)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--learn", action="store_true", default=True)
    parser.add_argument("--no-learn", action="store_true")
    parser.add_argument(
        "--learned",
        action="store_true",
        help="Act using the learned policy instead of the heuristic/LLM planner",
    )
    parser.add_argument("--teach", action="store_true")
    parser.add_argument("--ollama", action="store_true")
    parser.add_argument("--ollama-model", default="dolphin-llama3")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--directive", default="")
    args = parser.parse_args()

    learn = not args.no_learn
    wins = 0
    for ep in range(1, args.episodes + 1):
        cfg = AgentConfig(
            use_ollama=args.ollama,
            ollama_model=args.ollama_model,
            learn=learn,
            use_learned_policy=args.learned,
            teach_mode=args.teach,
        )
        agent = PlayMindAgent(world=DemoWorld(), config=cfg)
        if args.directive:
            agent.set_directive(args.directive)
        print(f"\n=== Episode {ep} ===")
        won = run_episode(agent, args.max_steps, interactive=args.interactive or args.teach)
        agent.save()
        wins += int(won)
        print(f"Episode {ep}: {'QUEST COMPLETE' if won else 'failed/timeout'}")
        print(f"Experience steps stored: {len(agent.buffer.rows)}")

    print(f"\nWins {wins}/{args.episodes}")
    print("Policy/experience saved under data/playmind/")


if __name__ == "__main__":
    main()
