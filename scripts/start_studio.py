#!/usr/bin/env python3
"""Launch the separate, offline-only PlayMind Studio GUI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playmind.gui.studio_dashboard import StudioGuiState, studio_doctor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--config")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate offline Studio startup without opening a port.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        state = StudioGuiState(args.config)
        print(
            json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "host": args.host,
                    "port": args.port,
                    "url": f"http://{args.host}:{args.port}/",
                    "profile": state.profile.to_dict(),
                    "doctor": studio_doctor(state),
                    "note": (
                        "Offline startup validated. FFmpeg is required for "
                        "video import, but not for opening Studio."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    from playmind.studio_gui import main as run_studio

    run_studio(
        args.host,
        args.port,
        not args.no_browser,
        config_path=args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
