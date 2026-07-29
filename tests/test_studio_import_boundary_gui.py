from __future__ import annotations

import subprocess
import sys


def test_importing_studio_gui_does_not_load_actuators_or_owned_loop() -> None:
    source = """
import sys
import playmind.studio_gui
for forbidden in ("playmind.actuators", "playmind.owned_loop"):
    assert forbidden not in sys.modules, sorted(
        name for name in sys.modules if name.startswith("playmind.")
    )
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
