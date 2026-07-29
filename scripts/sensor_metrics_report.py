#!/usr/bin/env python3
"""Build a sensor metrics report from labels (+ optional predictions).

Reads ``data/playmind/labels/sensor_labels.jsonl`` by default, optionally joins
predictions JSONL on the ``frame`` field, prints Markdown, and writes JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from playmind.sensor_metrics import (
    DEFAULT_LABELS_PATH,
    DEFAULT_REPORT_PATH,
    load_labels_jsonl,
    metrics_from_labels_and_predictions,
    report_to_markdown,
    save_report,
)


def load_predictions_jsonl(path: Path | str) -> list[dict[str, Any]]:
    src = Path(path)
    if not src.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in src.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS_PATH),
        help="Path to sensor_labels.jsonl",
    )
    p.add_argument(
        "--predictions",
        default=None,
        help="Optional predictions JSONL joined on frame path",
    )
    p.add_argument(
        "--out",
        default=str(DEFAULT_REPORT_PATH),
        help="Output JSON report path",
    )
    p.add_argument(
        "--markdown-out",
        default=None,
        help="Optional path to also write the Markdown report",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print Markdown to stdout",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels = load_labels_jsonl(args.labels)
    predictions = load_predictions_jsonl(args.predictions) if args.predictions else None
    metrics = metrics_from_labels_and_predictions(labels, predictions)
    report = metrics.compute_report()
    out = save_report(report, path=args.out)
    md = report_to_markdown(report)
    if args.markdown_out:
        md_path = Path(args.markdown_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
    if not args.quiet:
        print(md)
    print(f"Wrote JSON report -> {out}", file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
