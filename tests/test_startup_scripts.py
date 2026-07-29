from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_windows_and_cross_platform_startup_files_exist() -> None:
    required = (
        "setup_windows.ps1",
        "start_playmind.ps1",
        "start_playmind.bat",
        "setup_wsl_training.ps1",
        "requirements-playmind.txt",
        "requirements-playmind-ml.txt",
        "scripts/doctor.py",
        "scripts/start_all.py",
    )
    for relative in required:
        assert (ROOT / relative).is_file(), relative
    legacy = (ROOT / "scripts/PLAY_WITH_GUI.bat").read_text(encoding="utf-8")
    assert "start_playmind.bat" in legacy
    assert "c:\\Users" not in legacy


def test_start_all_dry_run() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/start_all.py"), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "8777" in result.stdout
    assert "mode=shadow" in result.stdout
    assert "keyboard=off" in result.stdout
