#!/usr/bin/env python3
"""Migrate legacy PlayMind learning artifacts for Learning Architecture V2.

Usage:
  python3 scripts/migrate_legacy_learning.py
  python3 scripts/migrate_legacy_learning.py --data-dir data/playmind/owned
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Migrate legacy Q / memory files for Learning V2")
    p.add_argument(
        "--data-dir",
        default="data/playmind/owned",
        help="Owned-game data directory containing policy.json etc.",
    )
    p.add_argument(
        "--overwrite-legacy-policy",
        action="store_true",
        help="Overwrite an existing policy.legacy.json copy",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable MigrationReport JSON",
    )
    args = p.parse_args(argv)

    from playmind.migration import migrate_owned_data

    report = migrate_owned_data(
        Path(args.data_dir),
        overwrite_legacy_policy=bool(args.overwrite_legacy_policy),
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"Migrated data_dir={report.data_dir}")
        for action in report.actions:
            print(f"  - {action}")
        if report.policy_legacy_path:
            print(f"  policy_legacy={report.policy_legacy_path}")
        if report.schema_stamped:
            print(f"  schema_stamped={', '.join(report.schema_stamped)}")
        for w in report.warnings:
            print(f"  warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
