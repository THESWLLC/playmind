#!/usr/bin/env python3
"""Capture one screenshot for ROI calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from playmind.capture import (
    capture_monitor,
    capture_region,
    capture_window,
    list_visible_windows,
)
from playmind.vision import read_frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/playmind/capture_sample.png")
    p.add_argument("--monitor", type=int, default=1)
    p.add_argument("--region", nargs=4, type=int, metavar=("L", "T", "W", "H"))
    p.add_argument("--window", help="Capture window by title substring")
    p.add_argument("--list-windows", action="store_true", help="List visible windows")
    p.add_argument("--ocr", action="store_true")
    args = p.parse_args()

    if args.list_windows:
        windows = list_visible_windows()
        if not windows:
            print("No large visible windows found (or not on Windows).")
            return
        for w in windows:
            line = f"{w.width}x{w.height} @ {w.left},{w.top}  {w.title!r}"
            print(line.encode("utf-8", "replace").decode("utf-8"))
        return

    out = Path(args.out)
    if args.window:
        result = capture_window(out, args.window)
    elif args.region:
        l, t, w, h = args.region
        result = capture_region(out, l, t, w, h)
    else:
        result = capture_monitor(out, monitor_index=args.monitor)
    note = f" {result.note}" if result.note else ""
    print(f"saved {result.path} ({result.width}x{result.height}) via {result.backend}{note}")
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
