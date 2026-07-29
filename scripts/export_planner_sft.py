#!/usr/bin/env python3
"""Export demonstration records to planner SFT JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playmind.planner_data.export_sft import export_sft, load_demonstration_records


def _jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "--data-dir",
        dest="input_path",
        default="data/playmind/demonstrations",
        help="demonstration directory or standalone JSONL",
    )
    parser.add_argument("--output-dir", default="data/playmind/planner/sft")
    parser.add_argument("--manifest-dir", default="data/playmind/planner/manifests")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-ineligible", action="store_true")
    args = parser.parse_args(argv)
    source = Path(args.input_path)
    records = _jsonl(source) if source.is_file() else load_demonstration_records(source)
    manifest = export_sft(
        records,
        args.output_dir,
        manifest_dir=args.manifest_dir,
        seed=args.seed,
        include_ineligible=args.include_ineligible,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
