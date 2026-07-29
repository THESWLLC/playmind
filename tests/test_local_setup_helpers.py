"""Tests for synthetic demo generation and setup helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from playmind.training.dataset import DemonstrationDataset


ROOT = Path(__file__).resolve().parents[1]


def test_generate_synthetic_demos(tmp_path: Path) -> None:
    out = tmp_path / "demos"
    script = ROOT / "scripts" / "generate_synthetic_demos.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-dir",
            str(out),
            "--sessions",
            "4",
            "--episodes-per-session",
            "2",
            "--steps",
            "12",
            "--json",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["total_samples"] >= 4 * 2 * 12
    assert (out / "SYNTHETIC_MANIFEST.json").exists()

    ds = DemonstrationDataset(
        root=out,
        window_size=8,
        split="all",
        include_unlabeled=False,
    )
    assert len(ds) > 0
    sample = ds[0]
    assert "features" in sample
    assert sample.get("skill") or sample.get("skill_target")
    # Episode-separated splits should assign every episode.
    splits = ds.episode_split_map()
    assert splits


def test_setup_owned_game_copies_config(tmp_path: Path) -> None:
    dest = tmp_path / "owned_game.json"
    script = ROOT / "scripts" / "setup_owned_game.py"
    example = ROOT / "config" / "owned_game.example.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--example",
            str(example),
            "--dest",
            str(dest),
            "--window-title",
            "SmokeGame",
            "--bc-checkpoint",
            "models/checkpoints/recurrent_skill_policy_v2.json",
            "--force",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    cfg = json.loads(dest.read_text(encoding="utf-8"))
    assert cfg["capture"]["window_title"] == "SmokeGame"
    assert cfg["learning_v2"]["enabled"] is True
    assert "recurrent_skill_policy_v2.json" in cfg["learning_v2"]["bc_checkpoint"]
