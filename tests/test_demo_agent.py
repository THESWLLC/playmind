from pathlib import Path

from playmind.agent import AgentConfig, PlayMindAgent
from playmind.demo_world import DemoWorld
from playmind.learning import state_key
from playmind.planner import HeuristicPlanner


def test_heuristic_can_complete_quest() -> None:
    world = DemoWorld()
    planner = HeuristicPlanner()
    for _ in range(100):
        obs = world.observe()
        action = planner.plan(obs)
        _, _, done, _ = world.step(action)
        if done:
            break
    assert world.quest_complete


def test_agent_learns_and_saves(tmp_path: Path) -> None:
    cfg = AgentConfig(learn=True, teach_mode=False, data_dir=tmp_path / "d")
    agent = PlayMindAgent(world=DemoWorld(), config=cfg)
    for _ in range(40):
        result = agent.tick()
        if result["done"]:
            break
    agent.save()
    assert (cfg.data_dir / "policy.json").exists()
    assert (cfg.data_dir / "experience.jsonl").exists()
    assert (cfg.data_dir / "finetune.jsonl").exists()


def test_teach_boosts_policy(tmp_path: Path) -> None:
    cfg = AgentConfig(learn=True, teach_mode=True, data_dir=tmp_path / "teach")
    agent = PlayMindAgent(world=DemoWorld(), config=cfg)
    obs = agent.observe()
    agent.answer_teach("open_quest_log", obs)
    assert agent.policy.q[state_key(obs)]["open_quest_log"] > 0
