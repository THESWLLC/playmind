"""Unit tests for demonstration recorder, BC stubs, dataset, and replay env."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from playmind.demonstrations import (
    SCHEMA_VERSION,
    DemonstrationRecorder,
    list_sessions,
    load_session_samples,
)
from playmind.models.policy_v2 import TORCH_AVAILABLE, SkillPolicyV2, UNTRAINED_CONFIDENCE
from playmind.observations import Observation
from playmind.policies.scripted import ScriptedPolicy
from playmind.replay_env import ReplayEnv
from playmind.training.dataset import DemonstrationDataset


def test_policy_v2_imports_without_requiring_torch() -> None:
    # Module must be importable regardless of TORCH_AVAILABLE.
    assert isinstance(TORCH_AVAILABLE, bool)
    policy = SkillPolicyV2()
    skill, conf = policy.predict_skill({"vision_player_hp": 0.9, "has_target": False})
    assert isinstance(skill, str) and skill
    assert conf <= UNTRAINED_CONFIDENCE + 1e-9
    assert policy.trained is False


def test_policy_v2_save_load_metadata(tmp_path: Path) -> None:
    policy = SkillPolicyV2(skill_names=["explore", "wait", "acquire_target"])
    path = tmp_path / "ckpt.json"
    policy.save(path)
    assert path.exists()
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["model_version"]
    assert meta["skill_names"] == ["explore", "wait", "acquire_target"]
    assert meta["trained"] is False
    assert "torch" in meta["note"].lower() or "CNN" in meta["note"]

    loaded = SkillPolicyV2.load(path)
    assert loaded.skill_names == policy.skill_names
    skill, conf = loaded.predict_skill([0.1, 0.2, 0.3])
    assert skill in loaded.skill_names
    assert conf < 0.5


def test_demonstration_recorder_jsonl_and_mark(tmp_path: Path) -> None:
    root = tmp_path / "demonstrations"
    rec = DemonstrationRecorder(root=root)
    sid = rec.start(goal="farm", profile="test-profile", episode_id="ep-1")
    assert rec.recording is True
    assert sid

    frame = rec.session_dir / "frames" / "000.png"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"fake")

    row = rec.append(
        frame_path=frame,
        observation={"vision_player_hp": 0.8, "has_target": True, "life_phase": "alive"},
        key_events=["key:1", {"type": "click", "x": 10}],
        notes="first strike",
        skill="basic_combat_rotation",
        timestamp=1000.0,
        episode_id="ep-1",
    )
    assert row["schema_version"] == SCHEMA_VERSION
    assert row["frame_path"] == "frames/000.png"
    assert row["skill"] == "basic_combat_rotation"
    assert row["goal"] == "farm"

    # Observation object path
    obs = Observation.from_legacy_dict(
        {"vision_player_hp": 0.5, "is_dead": False, "life_phase": "alive", "has_target": False}
    )
    row2 = rec.append(
        observation=obs,
        key_events=[],
        skill="explore",
        episode_id="ep-1",
        timestamp=1001.0,
    )
    assert "vision_player_hp" in row2["observation"]

    rec.mark("success", notes="good run")
    session_json = json.loads((rec.session_dir / "session.json").read_text(encoding="utf-8"))
    assert session_json["outcome"] == "success"
    assert session_json["schema_version"] == SCHEMA_VERSION

    # Per-sample bad mark
    rec.mark("bad", sample_id=row["sample_id"], notes="misclick")
    samples = load_session_samples(rec.session_dir)
    assert len(samples) == 2
    by_id = {s["sample_id"]: s for s in samples}
    assert by_id[row["sample_id"]]["label"] == "bad"

    out_dir = rec.stop()
    assert out_dir == rec.session_dir
    assert not rec.recording
    assert (out_dir / "meta.jsonl").exists()
    assert list_sessions(root)


def test_demonstration_recorder_requires_start(tmp_path: Path) -> None:
    rec = DemonstrationRecorder(root=tmp_path)
    with pytest.raises(RuntimeError):
        rec.append(observation={"vision_player_hp": 1.0})


def test_dataset_episode_split_and_windows(tmp_path: Path) -> None:
    root = tmp_path / "demonstrations"
    # Two episodes across one session
    rec = DemonstrationRecorder(root=root)
    rec.start(episode_id="ep-a")
    for i in range(5):
        rec.append(
            observation={"vision_player_hp": 0.9 - i * 0.01, "has_target": False},
            key_events=["explore"],
            skill="explore",
            episode_id="ep-a",
            timestamp=float(i),
        )
    # Switch episode mid-session
    for i in range(3):
        rec.append(
            observation={"vision_player_hp": 0.4, "has_target": True, "in_combat": True},
            key_events=["attack"],
            skill="basic_combat_rotation",
            episode_id="ep-b",
            timestamp=float(10 + i),
        )
    rec.mark("success")
    rec.stop()

    train = DemonstrationDataset(root, window_size=3, split="train", seed=0)
    val = DemonstrationDataset(root, window_size=3, split="val", seed=0)
    test = DemonstrationDataset(root, window_size=3, split="test", seed=0)
    all_ds = DemonstrationDataset(root, window_size=3, split="all", seed=0)

    assert len(all_ds) == 8  # 5 + 3 windows (one per sample)
    # Episode-based: each episode appears in exactly one split.
    split_map = all_ds.episode_split_map()
    assert set(split_map) == {"ep-a", "ep-b"}
    assert split_map["ep-a"] != "" and split_map["ep-b"] != ""
    assert len(train) + len(val) + len(test) == len(all_ds)

    item = all_ds[0]
    assert "features" in item and len(item["features"]) == 3
    assert isinstance(item, dict)
    # No torch tensors
    assert not type(item["features"][0]).__module__.startswith("torch")

    batches = list(all_ds.iter_batches(batch_size=4))
    assert batches
    assert "feature" in batches[0]
    summary = all_ds.validate()
    assert summary["windows"] == 8
    assert "explore" in summary["skills"]


def test_replay_env_no_actuators(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "demonstrations"
    rec = DemonstrationRecorder(root=root)
    rec.start(episode_id="ep-replay")
    rec.append(
        observation={
            "vision_player_hp": 0.0,
            "is_dead": True,
            "life_phase": "dead_dialog",
            "has_target": False,
        },
        skill="death_recovery",
        key_events=[],
    )
    rec.append(
        observation={
            "vision_player_hp": 0.9,
            "is_dead": False,
            "is_ghost": False,
            "life_phase": "alive",
            "has_target": False,
            "hostiles_near": True,
        },
        skill="acquire_target",
        key_events=[],
    )
    session_dir = rec.stop()

    # Fail loudly if any actuator import side-effect is invoked.
    def _boom(*_a, **_k):
        raise AssertionError("actuators must not be called during replay")

    monkeypatch.setattr("playmind.actuators.press_key", _boom, raising=False)

    env = ReplayEnv.from_session(session_dir, policy=ScriptedPolicy())
    assert len(env) == 2
    obs0 = env.reset()
    assert obs0 is not None
    results = env.run()
    assert len(results) == 2
    assert results[0].decision.skill in {"death_recovery", "ghost_runback", "wait"}
    assert results[-1].done is True
    # Agreement when labels present
    rate = env.agreement_rate(results)
    assert 0.0 <= rate <= 1.0


def test_train_behavior_clone_cli_without_torch(tmp_path: Path) -> None:
    root = tmp_path / "demonstrations"
    rec = DemonstrationRecorder(root=root)
    rec.start(episode_id="ep-cli")
    rec.append(
        observation={"vision_player_hp": 0.7, "has_target": False},
        skill="explore",
        key_events=["w"],
    )
    rec.stop()

    script = Path(__file__).resolve().parents[1] / "scripts" / "train_behavior_clone.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-dir",
            str(root),
            "--dry-validate-only",
            "--window-size",
            "2",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DemonstrationDataset validation" in proc.stdout
    assert "validated_windows=" in proc.stdout
