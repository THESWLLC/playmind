from __future__ import annotations

from pathlib import Path

from scripts.doctor import doctor_report


def test_doctor_returns_structured_report(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "owned_game.json").write_text(
        '{"mode": "shadow", "enable_keyboard": false}\n',
        encoding="utf-8",
    )
    report = doctor_report(tmp_path, timeout=0.01)
    assert isinstance(report["ok"], bool)
    assert report["python"]["supported"] is True
    assert report["config"]["valid"] is True
    assert report["config"]["safe_defaults"]["keyboard_enabled"] is False
    assert "cuda" in report
    assert "torch" in report
    assert "ram" in report
    assert "disk" in report
    assert "ollama" in report
    assert "training_dependencies" in report
    assert "capture_dependencies" in report
    assert "permissions" in report
