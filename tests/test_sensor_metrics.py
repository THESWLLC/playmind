"""Tests for sensor metrics, labeling helpers, and ROI overlays."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from playmind.sensor_metrics import (
    SensorMetrics,
    append_label_jsonl,
    compute_report,
    load_labels_jsonl,
    load_report,
    metrics_from_labels_and_predictions,
    report_to_markdown,
    save_report,
    update_from_prediction_vs_label,
)
from playmind.vision_overlay import draw_rois


def test_binary_perfect_scores() -> None:
    m = SensorMetrics()
    for _ in range(4):
        m.update(
            {
                "has_target": True,
                "in_combat": False,
                "is_dead": False,
                "is_ghost": False,
                "moving": True,
                "modal": False,
                "hostiles_near": True,
            },
            {
                "has_target": True,
                "in_combat": False,
                "is_dead": False,
                "is_ghost": False,
                "moving": True,
                "modal": False,
                "hostiles_near": True,
            },
        )
    report = m.compute_report()
    for name in ("target", "combat", "death", "ghost", "movement", "modal", "hostile"):
        metrics = report["binary"][name]
        assert metrics["precision"] == 1.0 or metrics["tp"] + metrics["fp"] == 0
        assert metrics["fnr"] == 0.0 or metrics["fn"] + metrics["tp"] == 0
        if metrics["tp"] > 0:
            assert metrics["recall"] == 1.0
            assert metrics["f1"] == 1.0
        if metrics["fp"] + metrics["tn"] > 0:
            assert metrics["fpr"] == 0.0


def test_binary_fp_fn_and_fpr_fnr() -> None:
    m = SensorMetrics()
    # FP: pred True, label False
    m.update({"has_target": True}, {"has_target": False})
    # FN: pred False, label True
    m.update({"has_target": False}, {"has_target": True})
    # TN
    m.update({"has_target": False}, {"has_target": False})
    # TP
    m.update({"has_target": True}, {"has_target": True})
    t = m.compute_report()["binary"]["target"]
    assert t["tp"] == 1 and t["fp"] == 1 and t["tn"] == 1 and t["fn"] == 1
    assert t["precision"] == pytest.approx(0.5)
    assert t["recall"] == pytest.approx(0.5)
    assert t["f1"] == pytest.approx(0.5)
    assert t["fpr"] == pytest.approx(0.5)
    assert t["fnr"] == pytest.approx(0.5)


def test_health_mae_and_objective_progress() -> None:
    m = SensorMetrics()
    m.update(
        {"player_hp": 0.8, "target_hp": 0.4, "objective_progress": 0.5},
        {"player_hp": 1.0, "target_hp": 0.2, "objective_progress": 0.0},
    )
    m.update(
        {"vision_player_hp": 0.5, "target_hp_est": 0.5, "objective_progress": 1.0},
        {"player_hp": 0.5, "target_hp": 0.5, "objective_progress": 1.0},
    )
    report = compute_report(m)
    assert report["health_mae"]["player_hp"]["mae"] == pytest.approx(0.1)
    assert report["health_mae"]["target_hp"]["mae"] == pytest.approx(0.1)
    assert report["continuous"]["objective_progress"]["mae"] == pytest.approx(0.25)


def test_life_phase_confusion_matrix() -> None:
    m = SensorMetrics()
    m.update({"life_phase": "alive"}, {"life_phase": "alive"})
    m.update({"life_phase": "ghost"}, {"life_phase": "alive"})
    m.update({"life_phase": "ghost"}, {"life_phase": "ghost"})
    report = m.compute_report()
    cm = report["life_phase_confusion"]
    assert cm["alive"]["alive"] == 1
    assert cm["alive"]["ghost"] == 1
    assert cm["ghost"]["ghost"] == 1


def test_movement_from_motion_threshold() -> None:
    m = SensorMetrics()
    m.update({"motion": 3.0}, {"motion": 0.0})
    report = m.compute_report()["binary"]["movement"]
    assert report["fp"] == 1


def test_update_from_prediction_vs_label_helper() -> None:
    m = SensorMetrics()
    update_from_prediction_vs_label(
        m, {"is_dead": True, "life_phase": "dead_dialog"}, {"is_dead": True, "life_phase": "dead_dialog"}
    )
    assert m.n_compared == 1
    assert m.compute_report()["binary"]["death"]["tp"] == 1


def test_save_load_report_and_markdown(tmp_path: Path) -> None:
    m = SensorMetrics()
    m.note_label()
    m.note_label()
    m.update({"has_target": True, "player_hp": 0.9}, {"has_target": True, "player_hp": 0.8})
    report = m.compute_report()
    assert report["n_labeled"] == 2
    out = tmp_path / "labels" / "sensor_metrics_report.json"
    save_report(report, path=out)
    loaded = load_report(out)
    assert loaded["n_labeled"] == 2
    assert loaded["binary"]["target"]["tp"] == 1
    md = report_to_markdown(loaded)
    assert "Sensor metrics report" in md
    assert "target" in md
    assert "Health MAE" in md


def test_append_and_load_labels_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sensor_labels.jsonl"
    append_label_jsonl({"frame": "a.png", "has_target": True, "life_phase": "alive"}, path=path)
    append_label_jsonl({"frame": "b.png", "is_dead": True, "life_phase": "dead_dialog"}, path=path)
    rows = load_labels_jsonl(path)
    assert len(rows) == 2
    assert rows[0]["frame"] == "a.png"


def test_metrics_from_labels_and_predictions_join() -> None:
    labels = [
        {"frame": "f1.png", "has_target": True, "player_hp": 1.0, "life_phase": "alive"},
        {"frame": "f2.png", "has_target": False, "player_hp": 0.2, "life_phase": "ghost"},
        {"frame": "f3.png", "has_target": True, "life_phase": "alive"},
    ]
    preds = [
        {"frame": "f1.png", "has_target": True, "player_hp": 0.9, "life_phase": "alive"},
        {"frame": "f2.png", "has_target": True, "player_hp": 0.1, "life_phase": "alive"},
    ]
    m = metrics_from_labels_and_predictions(labels, preds)
    report = m.compute_report()
    assert report["n_labeled"] == 3
    assert report["n_compared"] == 2
    assert report["binary"]["target"]["tp"] == 1
    assert report["binary"]["target"]["fp"] == 1
    assert report["health_mae"]["player_hp"]["n"] == 2


def test_review_sensor_frames_list_and_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "one.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (frames / "two.jpg").write_bytes(b"JPEG")
    (frames / "skip.txt").write_text("nope")

    from scripts import review_sensor_frames as rsf

    listed = rsf.list_frame_paths(frames)
    assert [p.name for p in listed] == ["one.png", "two.jpg"]

    labels_path = tmp_path / "labels" / "sensor_labels.jsonl"
    record = rsf.label_from_form(
        {
            "frame": [str(frames / "one.png")],
            "is_dead": ["1"],
            "has_target": [],
            "player_hp": ["0.75"],
            "target_hp": [""],
            "life_phase": ["dead_dialog"],
        }
    )
    assert record["is_dead"] is True
    assert record["has_target"] is False
    assert record["player_hp"] == 0.75
    assert "target_hp" not in record
    assert record["life_phase"] == "dead_dialog"

    out = rsf.append_label(record, path=labels_path)
    assert out.exists()
    rows = load_labels_jsonl(out)
    assert rows[0]["is_dead"] is True

    # Non-interactive --label from stdin
    monkeypatch.setattr(
        "sys.stdin",
        __import__("io").StringIO(
            json.dumps({"frame": "two.jpg", "has_target": True, "life_phase": "alive"})
        ),
    )
    rc = rsf.main(["--label", "--labels", str(labels_path)])
    assert rc == 0
    assert len(load_labels_jsonl(labels_path)) == 2

    # --list
    rc = rsf.main(["--list", str(frames)])
    assert rc == 0


def test_sensor_metrics_report_script(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    labels_path = tmp_path / "sensor_labels.jsonl"
    preds_path = tmp_path / "preds.jsonl"
    out_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    labels_path.write_text(
        json.dumps(
            {"frame": "a.png", "has_target": True, "player_hp": 1.0, "life_phase": "alive"}
        )
        + "\n",
        encoding="utf-8",
    )
    preds_path.write_text(
        json.dumps(
            {"frame": "a.png", "has_target": False, "player_hp": 0.5, "life_phase": "alive"}
        )
        + "\n",
        encoding="utf-8",
    )

    from scripts import sensor_metrics_report as smr

    rc = smr.main(
        [
            "--labels",
            str(labels_path),
            "--predictions",
            str(preds_path),
            "--out",
            str(out_path),
            "--markdown-out",
            str(md_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["n_labeled"] == 1
    assert report["binary"]["target"]["fn"] == 1
    printed = capsys.readouterr().out
    assert "Sensor metrics report" in printed
    assert md_path.read_text(encoding="utf-8").startswith("# Sensor metrics")


def test_vision_overlay_draws_with_pillow(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    frame = tmp_path / "frame.png"
    Image.new("RGB", (100, 80), color=(30, 30, 30)).save(frame)
    out = tmp_path / "overlay.png"
    result = draw_rois(
        frame,
        {
            "hp_roi": [5, 5, 40, 20],
            "target": {"box": [50, 10, 90, 30], "confidence": 0.87, "label": "target"},
        },
        out,
        confidences={"hp_roi": 0.91},
    )
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_vision_overlay_noop_without_pillow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    out = draw_rois(tmp_path / "missing.png", {"hp_roi": [0, 0, 1, 1]}, tmp_path / "o.png")
    assert out is None
