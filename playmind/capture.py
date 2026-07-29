"""Screen / window capture for owned-game observation.

Optional deps: mss (fast), Pillow (fallback ImageGrab).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CaptureResult:
    path: Path
    width: int
    height: int
    backend: str
    note: str = ""


def capture_monitor(
    out_path: Path,
    monitor_index: int = 1,
) -> CaptureResult:
    """Capture a full monitor to PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import mss  # type: ignore
        import mss.tools  # type: ignore

        with mss.mss() as sct:
            monitors = sct.monitors
            idx = min(max(monitor_index, 0), len(monitors) - 1)
            shot = sct.grab(monitors[idx])
            mss.tools.to_png(shot.rgb, shot.size, output=str(out_path))
            return CaptureResult(
                path=out_path,
                width=shot.width,
                height=shot.height,
                backend="mss",
            )
    except Exception as exc:  # noqa: BLE001
        pass

    try:
        from PIL import ImageGrab  # type: ignore

        img = ImageGrab.grab()
        img.save(out_path)
        return CaptureResult(
            path=out_path,
            width=img.width,
            height=img.height,
            backend="pil_imagegrab",
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "No capture backend available. Install: pip install mss Pillow"
        ) from exc


def capture_region(
    out_path: Path,
    left: int,
    top: int,
    width: int,
    height: int,
) -> CaptureResult:
    """Capture a rectangular screen region."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import mss  # type: ignore
        import mss.tools  # type: ignore

        region = {"left": left, "top": top, "width": width, "height": height}
        with mss.mss() as sct:
            shot = sct.grab(region)
            mss.tools.to_png(shot.rgb, shot.size, output=str(out_path))
            return CaptureResult(out_path, shot.width, shot.height, "mss_region")
    except Exception:
        from PIL import ImageGrab  # type: ignore

        img = ImageGrab.grab(bbox=(left, top, left + width, top + height))
        img.save(out_path)
        return CaptureResult(out_path, img.width, img.height, "pil_region")


def timed_capture_burst(
    out_dir: Path,
    count: int = 3,
    interval_s: float = 0.2,
    monitor_index: int = 1,
) -> list[CaptureResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(count):
        path = out_dir / f"frame_{int(time.time() * 1000)}_{i}.png"
        results.append(capture_monitor(path, monitor_index=monitor_index))
        if i + 1 < count:
            time.sleep(interval_s)
    return results


def capture_config_from_dict(cfg: dict[str, Any], out_path: Path) -> CaptureResult:
    mode = cfg.get("mode", "monitor")
    if mode == "region":
        r = cfg["region"]
        return capture_region(
            out_path,
            int(r["left"]),
            int(r["top"]),
            int(r["width"]),
            int(r["height"]),
        )
    return capture_monitor(out_path, monitor_index=int(cfg.get("monitor_index", 1)))
