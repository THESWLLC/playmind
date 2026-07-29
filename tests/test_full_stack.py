import json
from pathlib import Path

from playmind.session import SessionConfig, SessionScheduler
from playmind.owned_loop import load_owned_config


def test_session_scheduler_break_and_stop(monkeypatch) -> None:
    cfg = SessionConfig(play_minutes=0.0001, break_minutes_min=0.0001, break_minutes_max=0.0001, max_wall_hours=0.0001)
    sched = SessionScheduler(config=cfg)
    # Force times
    sched.started_at = 0
    sched.segment_started_at = 0
    monkeypatch.setattr("playmind.session.time.time", lambda: 10.0)
    assert sched.should_stop()
    assert sched.should_start_break()
    mins = sched.start_break()
    assert mins > 0
    assert sched.on_break
    monkeypatch.setattr("playmind.session.time.time", lambda: 10.0 + 60)
    assert sched.break_done()
    sched.end_break()
    assert not sched.on_break


def test_owned_config_example_loads() -> None:
    cfg = load_owned_config(Path("config/owned_game.example.json"))
    assert cfg["i_own_this_game"] is False
    assert "capture" in cfg


def test_build_ollama_modelfile(tmp_path: Path) -> None:
    data = tmp_path / "finetune.jsonl"
    sample = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "{\"hp\":1}"},
            {"role": "assistant", "content": "attack"},
        ]
    }
    data.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    out = tmp_path / "Modelfile"
    # Inline minimal builder logic call via subprocess module import
    from scripts.build_ollama_modelfile import main as _  # may fail if scripts not package

    # Call script functions by exec of argparse-free path:
    import runpy
    import sys

    sys.argv = [
        "build_ollama_modelfile.py",
        "--data",
        str(data),
        "--out",
        str(out),
        "--examples",
        "1",
        "--base-model",
        "dolphin-llama3",
    ]
    runpy.run_path("scripts/build_ollama_modelfile.py", run_name="__main__")
    text = out.read_text(encoding="utf-8")
    assert "FROM dolphin-llama3" in text
    assert "attack" in text
