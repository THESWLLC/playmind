#!/usr/bin/env python3
"""Capture one screenshot for ROI calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from playmind.capture import capture_monitor, capture_region
from playmind.vision import read_frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/playmind/capture_sample.png")
    p.add_argument("--monitor", type=int, default=1)
    p.add_argument("--region", nargs=4, type=int, metavar=("L", "T", "W", "H"))
    p.add_argument("--ocr", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    if args.region:
        l, t, w, h = args.region
        result = capture_region(out, l, t, w, h)
    else:
        result = capture_monitor(out, monitor_index=args.monitor)
    print(f"saved {result.path} ({result.width}x{result.height}) via {result.backend}")
    if args.ocr:
        reading = read_frame(result.path)
        print("hp_est=", reading.player_hp)
        print("quest=", reading.quest_text)
        print("notes=", reading.notes)
        if reading.raw_text:
            print("--- OCR ---")
            print(reading.raw_text[:1000])


if __name__ == "__main__":
    main()
