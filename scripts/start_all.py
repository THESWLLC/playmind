#!/usr/bin/env python3
"""Cross-platform PlayMind GUI launcher."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    url = f"http://{args.host}:{args.port}/"
    if args.dry_run:
        print(f"Would start PlayMind owned GUI at {url}")
        print(f"Repository: {ROOT}")
        print("Safe defaults: mode=shadow, keyboard=off")
        return 0

    from playmind.owned_gui import main as gui_main

    gui_main(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
