#!/usr/bin/env python3
"""Export a Learning V2 diagnostic bundle (folder + optional zip).

Usage:
  PYTHONPATH=. python3 scripts/export_diagnostics.py
  PYTHONPATH=. python3 scripts/export_diagnostics.py --no-zip --owned-dir data/playmind/owned
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export PlayMind Learning V2 diagnostics")
    p.add_argument(
        "--out-root",
        default="data/playmind/diagnostics",
        help="Diagnostics root directory",
    )
    p.add_argument(
        "--owned-dir",
        default="data/playmind/owned",
        help="Owned-game data directory to sample",
    )
    p.add_argument(
        "--config",
        default="config/owned_game.json",
        help="Config JSON to snapshot (falls back to example if missing)",
    )
    p.add_argument("--no-zip", action="store_true", help="Skip writing the .zip archive")
    p.add_argument(
        "--json",
        action="store_true",
        help="Print destination path as JSON",
    )
    args = p.parse_args(argv)

    from playmind.diagnostics import export_diagnostics

    config_path = Path(args.config)
    if not config_path.exists():
        alt = Path("config/owned_game.example.json")
        config_path = alt if alt.exists() else None

    dest = export_diagnostics(
        out_root=Path(args.out_root),
        owned_dir=Path(args.owned_dir),
        config_path=config_path,
        make_zip=not args.no_zip,
    )
    zip_path = dest.with_suffix(".zip")
    if args.json:
        print(
            json.dumps(
                {
                    "diagnostics_dir": str(dest),
                    "zip": str(zip_path) if zip_path.exists() else None,
                },
                indent=2,
            )
        )
    else:
        print(f"Wrote diagnostics → {dest}")
        if zip_path.exists():
            print(f"Zip archive → {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
