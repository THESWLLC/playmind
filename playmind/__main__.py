"""CLI: python -m playmind [options]."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playmind.actuators import DemoActuator, DryRunKeyboardActuator, ParsecKeyboardActuator
from playmind.agent import AgentConfig, PlayMindAgent
from playmind.demo_world import DemoWorld
from playmind.teach import prompt_teacher


def build_actuator(name: str):
    if name == "dry-run":
        return DryRunKeyboardActuator()
    if name == "parsec-stub":
        return ParsecKeyboardActuator(enabled=False)
    return DemoActuator()


def run_episode(agent: PlayMindAgent, max_steps: int, interactive: bool) -> bool:
    for _ in range(max_steps):
        obs = agent.observe()
        action = agent.propose_action(obs)
        force_teach = agent.config.teach_mode and interactive
        question = agent.maybe_ask(obs, action) if force_teach else None

        if force_teach and (question or obs.get("steps", 0) == 0):
            print(agent.world.render_ascii())
            if agent.last_vision and agent.last_vision.quest_text:
                print("vision:", agent.last_vision.quest_text)
            cmd, payload = prompt_teacher(action, obs)
            if cmd == "quit":
                return False
            if cmd == "directive" and payload:
                agent.set_directive(payload)
                continue
            if cmd == "action" and payload:
                agent.answer_teach(payload, obs)
                action = payload
            elif cmd == "help":
                continue
            elif cmd == "skip":
                pass
            # accept -> keep suggested action

        result = agent.tick(action)
        if interactive and not agent.config.teach_mode:
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
    parser.add_argument("--teach", action="store_true", help="Ask the human while playing")
    parser.add_argument("--vision", action="store_true", help="Emit/read demo vision frames")
    parser.add_argument("--ollama", action="store_true")
    parser.add_argument("--ollama-model", default="dolphin-llama3")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--directive", default="")
    parser.add_argument(
        "--actuator",
        choices=["demo", "dry-run", "parsec-stub"],
        default="demo",
        help="demo=in-process, dry-run=log keys, parsec-stub=future Parsec hook",
    )
    parser.add_argument("--data-dir", default="data/playmind")
    args = parser.parse_args()

    learn = not args.no_learn
    interactive = args.interactive or args.teach
    wins = 0

    cfg = AgentConfig(
        use_ollama=args.ollama,
        ollama_model=args.ollama_model,
        learn=learn,
        use_learned_policy=args.learned,
        teach_mode=args.teach,
        use_vision=args.vision,
        data_dir=Path(args.data_dir),
    )
    agent = PlayMindAgent(world=DemoWorld(), config=cfg, actuator=build_actuator(args.actuator))
    if args.directive:
        agent.set_directive(args.directive)

    for ep in range(1, args.episodes + 1):
        agent.world = DemoWorld()
        print(f"\n=== Episode {ep} ===")
        won = run_episode(agent, args.max_steps, interactive=interactive)
        agent.save()
        wins += int(won)
        print(f"Episode {ep}: {'QUEST COMPLETE' if won else 'failed/timeout'}")
        print(f"Experience steps stored: {len(agent.buffer.rows)}")

    print(f"\nWins {wins}/{args.episodes}")
    print(f"Artifacts saved under {cfg.data_dir}/")


if __name__ == "__main__":
    main()
