from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from playmind.studio.profiles import PROFILE_RETAIL_WOW_OFFLINE_ONLY, get_profile
from playmind.studio.safety import (
    StudioSafetyError,
    assert_studio_safe,
    detect_forbidden_live_context,
    studio_may_not_send_input,
)


def test_offline_profile_and_live_context_guard() -> None:
    profile = get_profile(PROFILE_RETAIL_WOW_OFFLINE_ONLY)
    assert profile.offline_only and profile.live_use_prohibited
    assert not profile.live_capture and not profile.generated_input
    assert assert_studio_safe() and studio_may_not_send_input()
    assert detect_forbidden_live_context(["obs.exe", "ffmpeg"]) is False
    with pytest.raises(StudioSafetyError, match="offline-only"):
        detect_forbidden_live_context([r"C:\Games\World of Warcraft\Wow.exe"])


def test_all_studio_modules_have_static_forbidden_import_boundary() -> None:
    studio = Path(__file__).parents[1] / "playmind" / "studio"
    forbidden = {
        "playmind.actuators",
        "pyautogui",
        "playmind.owned_loop",
    }
    for path in studio.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not imported.intersection(forbidden), path
        importlib.import_module(f"playmind.studio.{path.stem}")


def test_app_import_does_not_load_actuator_module() -> None:
    code = (
        "import sys; from playmind.studio.app import StudioApp; "
        "assert 'playmind.actuators' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
