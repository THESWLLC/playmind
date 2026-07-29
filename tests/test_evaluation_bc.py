"""Tests for BC training/eval polish and offline evaluation reports."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from playmind.demonstrations import DemonstrationRecorder
from playmind.evaluation.metrics import (
    aggregate_episode_metrics,
    kills_per_hour,
    skill_success_rates,
    summarize_replay_results,
)
from playmind.evaluation.report import write_evaluation_report
from playmind.evaluation.scenarios import (
    SCENARIO_SPECS,
    build_synthetic_session,
    run_all_scenarios,
    run_scenario,
)
from playmind.history import TemporalSummary
from playmind.models.policy_v2 import (
    AUX_KEYS,
    DEFAULT_FEATURE_DIM,
    TORCH_AVAILABLE,
    SkillPolicyV2,
    structured_feature_vector,
)
from playmind.models.recurrent_policy import DEFAULT_AUX_KEYS, RecurrentSkillPolicyV2
from playmind.observations import Observation
from playmind.policies.scripted import ScriptedPolicy
from playmind.replay_env import ReplayEnv, compare_policies
from playmind.training.evaluate_behavior_clone import evaluate_behavior_clone
from playmind.training.train_behavior_clone import (
    confusion_matrix,
    dry_validate,
    print_confusion_matrix,
    train_behavior_clone,
)


def _make_demo_root(tmp_path: Path) -> Path:
    root = tmp_path / "demonstrations"
    rec = DemonstrationRecorder(root=root)
    rec.start(episode_id="ep-a")
    for i in range(4):
        rec.append(
            observation={
                "vision_player_hp": 0.9,
                "has_target": False,
                "life_phase": "alive",
            },
            skill="explore",
            episode_id="ep-a",
            timestamp=float(i),
        )
    for i in range(3):
        rec.append(
            observation={
                "vision_player_hp": 0.5,
                "has_target": True,
                "in_combat": True,
                "life_phase": "alive",
            },
            skill="basic_combat_rotation",
            episode_id="ep-b",
            timestamp=float(10 + i),
        )
    rec.mark("success")
    rec.stop()
    return root


def test_structured_feature_vector_from_observation_and_summary() -> None:
    obs = Observation.from_legacy_dict(
        {
            "vision_player_hp": 0.8,
            "has_target": True,
            "in_combat": False,
            "is_dead": False,
            "life_phase": "alive",
        }
    )
    summary = TemporalSummary(health_trend=-0.1, target_flicker_count=2)
    vec = structured_feature_vector(obs, summary)
    assert len(vec) == DEFAULT_FEATURE_DIM
    assert abs(vec[0] - 0.8) < 1e-6
    # Temporal block is appended; health_trend should appear.
    assert -0.1 in vec
    assert 2.0 in vec


def test_skill_policy_predict_returns_aux(tmp_path: Path) -> None:
    policy = SkillPolicyV2()
    skill, conf, aux = policy.predict(
        observation={"vision_player_hp": 0.4, "has_target": False, "life_phase": "alive"}
    )
    assert skill in policy.skill_names
    assert conf < 0.5
    assert set(aux.keys()) == set(AUX_KEYS)
    assert all(v == 0.0 for v in aux.values())

    path = tmp_path / "ckpt.json"
    policy.save(path, config_snapshot={"experiment": "unit"})
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["model_version"]
    assert meta["config"]["experiment"] == "unit"
    assert meta["config"]["arch"] == "mlp"
    loaded = SkillPolicyV2.load(path)
    skill2, conf2, aux2 = loaded.predict([0.1] * loaded.feature_dim)
    assert skill2 in loaded.skill_names
    assert set(aux2) == set(AUX_KEYS)


def test_train_behavior_clone_without_torch(tmp_path: Path) -> None:
    root = _make_demo_root(tmp_path)
    metrics = tmp_path / "metrics.csv"
    ckpt = tmp_path / "ckpt.json"
    result = train_behavior_clone(
        root,
        epochs=2,
        patience=1,
        checkpoint=ckpt,
        metrics_csv=metrics,
        dry_validate_only=True,
    )
    assert result["ok"] is True
    assert result["trained"] is False
    summary = dry_validate(root, window_size=2, split="all")
    assert summary["windows"] >= 1


def test_train_cli_dry_validate(tmp_path: Path) -> None:
    root = _make_demo_root(tmp_path)
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


def test_evaluate_behavior_clone_metrics(tmp_path: Path) -> None:
    root = _make_demo_root(tmp_path)
    report = evaluate_behavior_clone(root, split="all", window_size=2, seed=0)
    assert "top1_accuracy" in report
    assert "per_skill" in report
    assert "confusion" in report
    assert report["n_samples"] >= 1
    # Confusion helpers
    labels = report["labels"]
    mat = confusion_matrix(
        ["explore", "explore", "basic_combat_rotation"],
        ["explore", "wait", "basic_combat_rotation"],
        labels,
    )
    assert len(mat) == len(labels)
    print_confusion_matrix(mat, labels)


def test_evaluate_cli(tmp_path: Path) -> None:
    root = _make_demo_root(tmp_path)
    out = tmp_path / "eval.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_behavior_clone.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-dir",
            str(root),
            "--split",
            "all",
            "--json-out",
            str(out),
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "top-1 accuracy" in proc.stdout
    assert out.exists()


def test_evaluation_metrics_stubs() -> None:
    episodes = [
        {
            "duration_s": 1800,
            "death_count": 1,
            "total_reward": 2.5,
            "skill_attempts": 10,
            "skill_successes": 7,
            "metadata": {
                "kills": 3,
                "invalid_actions": {"outside_mask": 2},
                "skill_stats": {
                    "explore": {"attempts": 4, "successes": 3},
                    "basic_combat_rotation": {"attempts": 6, "successes": 4},
                },
            },
        },
        {
            "duration_s": 1800,
            "death_count": 0,
            "total_reward": 1.0,
            "skill_attempts": 5,
            "skill_successes": 5,
            "metadata": {"kills": 1},
        },
    ]
    assert kills_per_hour(episodes) == pytest.approx(4.0)  # 4 kills / 1 hour
    rates = skill_success_rates(episodes)
    assert "overall" in rates
    assert rates["explore"] == pytest.approx(0.75)
    agg = aggregate_episode_metrics(episodes)
    assert agg["n_episodes"] == 2
    assert agg["invalid_action_counts"]["total"] == 2


def test_scenarios_and_report(tmp_path: Path) -> None:
    assert "death_recovery" in SCENARIO_SPECS
    scripted = ScriptedPolicy()
    result = run_scenario("death_recovery", scripted)
    assert result["n_steps"] == len(SCENARIO_SPECS["death_recovery"])
    assert 0.0 <= result["agreement_rate"] <= 1.0

    session = build_synthetic_session("combat_basic", tmp_path / "sess")
    env = ReplayEnv.from_session(session, policy=scripted)
    results = env.run()
    summary = summarize_replay_results(results, policy_name="scripted")
    assert summary["n_steps"] == len(results)

    all_res = run_all_scenarios(scripted)
    assert all_res["n_scenarios"] == len(SCENARIO_SPECS)

    compared = compare_policies(
        [
            {
                "observation": {"vision_player_hp": 0.0, "is_dead": True, "life_phase": "dead_dialog"},
                "skill": "death_recovery",
            }
        ],
        {"scripted": scripted},
    )
    assert "scripted" in compared

    payload = {
        "title": "unit",
        "summary": {"ok": True},
        "comparisons": {
            "scripted": {"mean_agreement": all_res["mean_agreement"], "n_scenarios": all_res["n_scenarios"]}
        },
        "episode_metrics": aggregate_episode_metrics([]),
        "notes": ["synthetic"],
    }
    paths = write_evaluation_report(tmp_path / "reports", payload, run_id="t1", csv_rows=[
        {"policy": "scripted", "scenario": "death_recovery", "agreement_rate": result["agreement_rate"]}
    ])
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()
    assert Path(paths["csv"]).exists()
    with Path(paths["csv"]).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows


def test_run_evaluation_cli(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_evaluation.py"
    out = tmp_path / "reports"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-dir",
            str(out),
            "--run-id",
            "unit",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "unit" / "report.json").exists()
    assert (out / "unit" / "report.md").exists()
    assert "scripted:" in proc.stdout
    assert "legacy_stub:" in proc.stdout


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_train_with_torch_if_available(tmp_path: Path) -> None:
    root = _make_demo_root(tmp_path)
    metrics = tmp_path / "metrics.csv"
    ckpt = tmp_path / "ckpt.json"
    result = train_behavior_clone(
        root,
        epochs=3,
        patience=2,
        checkpoint=ckpt,
        metrics_csv=metrics,
        dry_validate_only=False,
    )
    assert result["trained"] is True
    assert ckpt.exists()
    assert metrics.exists()
    loaded = RecurrentSkillPolicyV2.load(ckpt)
    skill, conf, aux = loaded.predict(
        observation={"vision_player_hp": 0.9, "has_target": False}
    )
    assert skill in loaded.skill_names
    assert set(aux) == set(DEFAULT_AUX_KEYS)
