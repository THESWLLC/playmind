"""Screen / window capture for owned-game observation.

Optional deps: mss (fast), Pillow (fallback ImageGrab).
Window-title capture uses Win32 APIs when available (Windows).
"""

from __future__ import annotations

import sys
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


@dataclass
class WindowBounds:
    left: int
    top: int
    width: int
    height: int
    title: str
    hwnd: int


def _ensure_dpi_aware() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # noqa: BLE001
        pass


def list_visible_windows(min_size: int = 200) -> list[WindowBounds]:
    """List visible top-level windows large enough to be a game client."""
    if sys.platform != "win32":
        return []
    _ensure_dpi_aware()
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found: list[WindowBounds] = []

    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        if not title:
            return True
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width < min_size or height < min_size:
            return True
        found.append(
            WindowBounds(
                left=int(rect.left),
                top=int(rect.top),
                width=width,
                height=height,
                title=title,
                hwnd=int(hwnd),
            )
        )
        return True

    user32.EnumWindows(EnumWindowsProc(_callback), 0)
    return found


def find_window_bounds(title_substr: str, *, client_area: bool = True) -> WindowBounds:
    """Find first visible window whose title contains title_substr (case-insensitive)."""
    needle = title_substr.strip().lower()
    if not needle:
        raise ValueError("window title_substr must be non-empty")
    if sys.platform != "win32":
        raise RuntimeError("window capture requires Windows")

    matches = [w for w in list_visible_windows() if needle in w.title.lower()]
    if not matches:
        titles = ", ".join(repr(w.title) for w in list_visible_windows()[:12])
        raise RuntimeError(
            f"No visible window matching {title_substr!r}. Visible: {titles or '(none)'}"
        )
    win = matches[0]
    if not client_area:
        return win

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    hwnd = wintypes.HWND(win.hwnd)
    client = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(client)):
        return win
    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return win
    width = int(client.right - client.left)
    height = int(client.bottom - client.top)
    if width < 32 or height < 32:
        return win
    return WindowBounds(
        left=int(pt.x),
        top=int(pt.y),
        width=width,
        height=height,
        title=win.title,
        hwnd=win.hwnd,
    )


def focus_window(hwnd: int) -> bool:
    """Bring a window to the foreground (best-effort)."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32
        # Allow SetForegroundWindow from this process more reliably
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        return bool(user32.SetForegroundWindow(hwnd))
    except Exception:  # noqa: BLE001
        return False


def _capture_window_printwindow(
    out_path: Path,
    bounds: WindowBounds,
) -> CaptureResult | None:
    """Capture window pixels via PrintWindow (works when another app occludes it)."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        from PIL import Image  # type: ignore
    except Exception:
        return None

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    hwnd = wintypes.HWND(bounds.hwnd)
    width, height = bounds.width, bounds.height
    if width < 32 or height < 32:
        return None

    hwnd_dc = user32.GetWindowDC(hwnd)
    if not hwnd_dc:
        return None
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
    old = gdi32.SelectObject(mem_dc, bmp)
    # PW_RENDERFULLCONTENT = 2 — better for DirectX/layered clients when supported
    ok = user32.PrintWindow(hwnd, mem_dc, 2)
    if not ok:
        ok = user32.PrintWindow(hwnd, mem_dc, 0)

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = width
    bmi.biHeight = -height  # top-down
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0

    buf_len = width * height * 4
    buf = (ctypes.c_char * buf_len)()
    gdi32.GetDIBits(mem_dc, bmp, 0, height, buf, ctypes.byref(bmi), 0)

    gdi32.SelectObject(mem_dc, old)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem_dc)
    user32.ReleaseDC(hwnd, hwnd_dc)

    if not ok:
        return None

    img = Image.frombuffer("RGBA", (width, height), bytes(buf), "raw", "BGRA", 0, 1).convert("RGB")
    # Reject near-black frames (PrintWindow often fails on exclusive fullscreen DX)
    extrema = img.convert("L").getextrema()
    if extrema[1] < 8:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return CaptureResult(
        path=out_path,
        width=width,
        height=height,
        backend="printwindow",
        note=f"title={bounds.title!r} hwnd={bounds.hwnd}",
    )


def capture_window(
    out_path: Path,
    title_substr: str,
    *,
    client_area: bool = True,
    prefer_printwindow: bool = True,
    focus_before_capture: bool = True,
) -> CaptureResult:
    """Capture a window matched by title substring.

    Tries PrintWindow first (occlusion-resistant). Many DirectX games return a
    black frame there, so we fall back to a screen-region grab — optionally
    focusing the window first so overlays are not included.
    """
    if prefer_printwindow:
        # PrintWindow paints the whole HWND; use outer bounds for a matching bitmap size.
        outer = find_window_bounds(title_substr, client_area=False)
        pw = _capture_window_printwindow(out_path, outer)
        if pw is not None:
            if client_area:
                try:
                    from PIL import Image  # type: ignore

                    client = find_window_bounds(title_substr, client_area=True)
                    # Client origin relative to outer window rect
                    ox = max(0, client.left - outer.left)
                    oy = max(0, client.top - outer.top)
                    img = Image.open(out_path).convert("RGB")
                    crop = img.crop((ox, oy, ox + client.width, oy + client.height))
                    if crop.width >= 32 and crop.height >= 32:
                        crop.save(out_path)
                        pw.width = crop.width
                        pw.height = crop.height
                        pw.note += " client_crop"
                except Exception:  # noqa: BLE001
                    pass
            return pw

    bounds = find_window_bounds(title_substr, client_area=client_area)
    focused = False
    if focus_before_capture:
        focused = focus_window(bounds.hwnd)
        if focused:
            time.sleep(0.15)
    result = capture_region(out_path, bounds.left, bounds.top, bounds.width, bounds.height)
    result.backend = f"{result.backend}_window"
    focus_note = " focused" if focused else " unfocused"
    result.note = (
        f"title={bounds.title!r} hwnd={bounds.hwnd} "
        f"(screen_region_fallback{focus_note})"
    )
    return result


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
    if mode == "window":
        return capture_window(
            out_path,
            str(cfg.get("window_title") or cfg.get("title_substr") or ""),
            client_area=bool(cfg.get("client_area", True)),
            prefer_printwindow=bool(cfg.get("prefer_printwindow", True)),
            focus_before_capture=bool(cfg.get("focus_before_capture", True)),
        )
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
