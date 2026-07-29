"""Tests for Learning V2 config validation, migration, and diagnostics (Phases 16/17/19)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from playmind.config_v2 import (
    LearningV2Settings,
    safe_defaults,
)
from playmind.diagnostics import (
    DiagnosticsBundle,
    export_diagnostics,
    redact_obj,
    redact_text,
)
from playmind.migration import (
    SCHEMA_VERSION,
    atomic_write_json,
    backup_corrupt,
    ensure_experience_sidecar,
    mark_policy_legacy,
    migrate_owned_data,
)


def test_safe_defaults_validate() -> None:
    s = safe_defaults()
    assert s.history_length == 16
    assert s.policy_mode == "hybrid"
    assert s.device == "cpu"
    assert s.seed == 0
    assert "kill_confirmed" in s.rewards
    assert s.skill_timeouts["death_recovery"] > 0
    assert 0.0 <= s.sensor_thresholds["has_target"] <= 1.0
    s.validate()


def test_load_from_owned_config_merges_and_aliases(tmp_path: Path) -> None:
    owned = {
        "learning_v2": {
            "enabled": True,
            "policy_mode": "scripted",
            "history_length": 8,
            "model_checkpoint": "models/ck.json",
            "sensor_confidence_thresholds": {"has_target": 0.7},
            "reward_values": {"death": -9.0},
            "demo": {"enabled": True, "root": str(tmp_path / "demos")},
            "eval": {"enabled": True, "scenarios": ["acquire"]},
            "device": "cpu",
            "seed": 42,
        }
    }
    s = LearningV2Settings.load_from_owned_config(owned)
    s.validate()
    assert s.enabled is True
    assert s.policy_mode == "scripted"
    assert s.history_length == 8
    assert s.bc_checkpoint == "models/ck.json"
    assert s.sensor_thresholds["has_target"] == pytest.approx(0.7)
    assert s.rewards["death"] == pytest.approx(-9.0)
    assert s.demonstration.enabled is True
    assert s.evaluation.scenarios == ["acquire"]
    assert s.seed == 42


def test_validate_rejects_bad_combos() -> None:
    s = LearningV2Settings(policy_mode="nope")
    with pytest.raises(ValueError, match="policy_mode"):
        s.validate()

    s = LearningV2Settings(history_length=0)
    with pytest.raises(ValueError, match="history_length"):
        s.validate()

    s = LearningV2Settings(confidence_threshold=1.5)
    with pytest.raises(ValueError, match="confidence_threshold"):
        s.validate()

    s = LearningV2Settings(device="tpu")
    with pytest.raises(ValueError, match="device"):
        s.validate()

    s = LearningV2Settings(
        enabled=True, policy_mode="scripted", legacy_q_fallback=True
    )
    with pytest.raises(ValueError, match="legacy_q_fallback"):
        s.validate()

    s = LearningV2Settings(
        enabled=True, policy_mode="legacy_q", bc_checkpoint="x.json"
    )
    with pytest.raises(ValueError, match="bc_checkpoint"):
        s.validate()

    s = LearningV2Settings(skill_timeouts={"wait": -1.0})
    with pytest.raises(ValueError, match="skill_timeouts"):
        s.validate()

    s = LearningV2Settings(sensor_thresholds={"motion": 2.0})
    with pytest.raises(ValueError, match="sensor_thresholds"):
        s.validate()


def test_atomic_write_and_backup_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    atomic_write_json(path, {"a": 1, "schema_version": 1})
    assert json.loads(path.read_text(encoding="utf-8"))["a"] == 1

    path.write_text("{not-json", encoding="utf-8")
    bak = backup_corrupt(path, reason="test")
    assert bak is not None and bak.exists()
    assert bak.read_text(encoding="utf-8") == "{not-json"


def test_migrate_marks_policy_and_stamps_schema(tmp_path: Path) -> None:
    data_dir = tmp_path / "owned"
    data_dir.mkdir()
    policy = {"q": {"s|a": 1.0}, "visits": {}}
    (data_dir / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    (data_dir / "process_memory.json").write_text(
        json.dumps({"death_pipeline": {}, "preventions": []}), encoding="utf-8"
    )
    (data_dir / "ui_memory.json").write_text(json.dumps({"labels": {}}), encoding="utf-8")
    (data_dir / "experience.jsonl").write_text(
        json.dumps({"s": 1}) + "\n", encoding="utf-8"
    )
    # Corrupt ability memory → .bak
    (data_dir / "ability_memory.json").write_text("NOT JSON{{{", encoding="utf-8")

    report = migrate_owned_data(data_dir)
    assert report.policy_legacy_path is not None
    legacy = Path(report.policy_legacy_path)
    assert legacy.exists()
    legacy_raw = json.loads(legacy.read_text(encoding="utf-8"))
    assert legacy_raw["legacy"] is True
    assert legacy_raw["schema_version"] == SCHEMA_VERSION
    assert "q" in legacy_raw

    live = json.loads((data_dir / "policy.json").read_text(encoding="utf-8"))
    assert live["legacy"] is True

    proc = json.loads((data_dir / "process_memory.json").read_text(encoding="utf-8"))
    assert proc["schema_version"] == SCHEMA_VERSION

    side = json.loads((data_dir / "experience.meta.json").read_text(encoding="utf-8"))
    assert side["schema_version"] == SCHEMA_VERSION
    assert side["legacy"] is True

    baks = list(data_dir.glob("ability_memory.json.bak*"))
    assert baks, "corrupt ability_memory should be backed up"

    # Idempotent second run
    report2 = migrate_owned_data(data_dir)
    assert report2.policy_legacy_path is not None


def test_mark_policy_legacy_missing(tmp_path: Path) -> None:
    assert mark_policy_legacy(tmp_path) is None
    side = ensure_experience_sidecar(tmp_path)
    assert side.exists()


def test_redact_home_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "userhome"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))

    secret = str(fake_home / "secret" / "policy.json")
    assert "<HOME>" in redact_text(secret)
    assert str(fake_home) not in redact_text(secret)

    obj = redact_obj({"path": secret, "nested": [secret]})
    assert obj["path"].startswith("<HOME>")
    assert fake_home.name not in json.dumps(obj) or "<HOME>" in obj["path"]


def test_export_diagnostics_bundle_and_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / "experience.jsonl").write_text(
        json.dumps({"obs": {"vision_player_hp": 0.5}, "action": "wait"}) + "\n",
        encoding="utf-8",
    )
    (owned / "policy_decisions.jsonl").write_text(
        json.dumps({"skill": "explore", "confidence": 0.9}) + "\n", encoding="utf-8"
    )
    (owned / "latest.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (owned / "exceptions.txt").write_text("boom\n", encoding="utf-8")

    cfg = tmp_path / "cfg.json"
    home = Path.home()
    cfg.write_text(
        json.dumps(
            {
                "learning_v2": {"enabled": True, "policy_mode": "hybrid"},
                "note": str(home / "hidden"),
            }
        ),
        encoding="utf-8",
    )

    out_root = tmp_path / "diagnostics"
    bundle = DiagnosticsBundle(
        temporal_summary={"len": 3},
        skill_state={"active": "explore"},
        reward_breakdown={"total": 0.1},
        episode_summary={"episode_id": "ep-1"},
        model_metadata={"model_version": "v2"},
        sensor_warnings=["low_hp_conf"],
        exceptions=["unit-test"],
    )
    dest = export_diagnostics(
        out_root=out_root,
        owned_dir=owned,
        config_path=cfg,
        bundle=bundle,
        screenshots=[owned / "latest.png"],
        make_zip=True,
        timestamp="teststamp",
    )
    assert dest.name == "teststamp"
    assert (dest / "recent_observations.json").exists()
    assert (dest / "temporal_summary.json").exists()
    assert (dest / "policy_decisions.json").exists()
    assert (dest / "skill_state.json").exists()
    assert (dest / "reward_breakdown.json").exists()
    assert (dest / "config_snapshot.json").exists()
    assert (dest / "model_metadata.json").exists()
    assert (dest / "episode_summary.json").exists()
    assert (dest / "exceptions.txt").exists()
    assert (dest / "screenshots" / "latest.png").exists()
    assert (dest / "manifest.json").exists()

    cfg_snap = json.loads((dest / "config_snapshot.json").read_text(encoding="utf-8"))
    note = str(cfg_snap.get("note") or "")
    assert "<HOME>" in note or str(home) not in note

    zpath = dest.with_suffix(".zip")
    assert zpath.exists()
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        assert any(n.endswith("manifest.json") for n in names)


def test_migrate_and_diagnostics_scripts(tmp_path: Path) -> None:
    data_dir = tmp_path / "owned"
    data_dir.mkdir()
    atomic_write_json(data_dir / "policy.json", {"q": {"x": 0.0}})
    (data_dir / "experience.jsonl").write_text("{}\n", encoding="utf-8")

    import scripts.migrate_legacy_learning as mig
    import scripts.export_diagnostics as exp

    rc = mig.main(["--data-dir", str(data_dir), "--json"])
    assert rc == 0
    assert (data_dir / "policy.legacy.json").exists()

    out = tmp_path / "diag"
    rc2 = exp.main(
        [
            "--owned-dir",
            str(data_dir),
            "--out-root",
            str(out),
            "--config",
            str(tmp_path / "missing.json"),
            "--no-zip",
            "--json",
        ]
    )
    assert rc2 == 0
    assert any(out.iterdir())
