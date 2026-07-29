from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.start_studio import build_parser

ROOT = Path(__file__).resolve().parents[1]


def test_default_studio_port_is_separate_from_owned_gui() -> None:
    args = build_parser().parse_args([])
    assert args.port == 8787
    assert args.port != 8777


def test_startup_dry_run_does_not_bind_server(tmp_path: Path) -> None:
    config = tmp_path / "studio.json"
    config.write_text(
        json.dumps(
            {
                "profile": "retail_wow_offline_only",
                "storage_root": str(tmp_path / "storage"),
                "projects_root": str(tmp_path / "projects"),
                "data_root": str(tmp_path / "data"),
                "registry_path": str(tmp_path / "registry.sqlite"),
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/start_studio.py"),
            "--dry-run",
            "--no-browser",
            "--config",
            str(config),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["port"] == 8787
    assert payload["profile"]["live_use_prohibited"] is True
