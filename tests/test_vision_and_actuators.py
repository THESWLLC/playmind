from pathlib import Path

from playmind.actuators import DryRunKeyboardActuator
from playmind.agent import AgentConfig, PlayMindAgent
from playmind.demo_world import DemoWorld
from playmind.vision import parse_quest_from_text, read_frame, save_demo_ascii_frame


def test_ascii_vision_reads_quest(tmp_path: Path) -> None:
    frame = tmp_path / "frame.txt"
    save_demo_ascii_frame("@..\n", "Kill 3 Wolves (0/3). Talk to Mira.", frame)
    reading = read_frame(frame)
    assert reading.quest_text
    assert "Kill" in reading.quest_text or "Wolf" in reading.quest_text


def test_parse_quest_from_text() -> None:
    assert parse_quest_from_text("Welcome\nKill 8 boars in the woods\n") is not None


def test_dry_run_actuator_logs(tmp_path: Path) -> None:
    log = tmp_path / "keys.jsonl"
    act = DryRunKeyboardActuator(log_path=log)
    act.send("attack")
    assert log.exists()
    assert "attack" in log.read_text(encoding="utf-8")


def test_agent_vision_flag(tmp_path: Path) -> None:
    cfg = AgentConfig(
        learn=True,
        use_vision=True,
        data_dir=tmp_path / "d",
        vision_frame_path=tmp_path / "d" / "frames" / "latest.txt",
    )
    agent = PlayMindAgent(world=DemoWorld(), config=cfg)
    obs = agent.observe()
    assert "vision_quest_text" in obs or obs.get("quest_text")
    agent.tick()
    assert cfg.vision_frame_path.exists()
