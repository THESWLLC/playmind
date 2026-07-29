"""Action actuators for demo and owned-game keyboard control.

Wire only to games you own / are allowed to automate.
Do NOT use with World of Warcraft or other restricted live MMO clients.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


DEFAULT_KEYMAP = {
    "move_north": "w",
    "move_south": "s",
    "move_west": "a",
    "move_east": "d",
    "attack": "1",
    "loot": "f",
    "interact": "e",
    "target_nearest": "tab",
    "open_quest_log": "l",
    "logout": "f12",
    "release_spirit": "__click_release__",
    "wait": None,
}

# Virtual-key codes for SendInput
_VK = {
    "w": 0x57,
    "a": 0x41,
    "s": 0x53,
    "d": 0x44,
    "e": 0x45,
    "f": 0x46,
    "g": 0x47,
    "h": 0x48,
    "i": 0x49,
    "j": 0x4A,
    "k": 0x4B,
    "l": 0x4C,
    "m": 0x4D,
    "n": 0x4E,
    "o": 0x4F,
    "p": 0x50,
    "q": 0x51,
    "r": 0x52,
    "t": 0x54,
    "u": 0x55,
    "v": 0x56,
    "x": 0x58,
    "y": 0x59,
    "z": 0x5A,
    "0": 0x30,
    "1": 0x31,
    "2": 0x32,
    "3": 0x33,
    "4": 0x34,
    "5": 0x35,
    "6": 0x36,
    "7": 0x37,
    "8": 0x38,
    "9": 0x39,
    "tab": 0x09,
    "space": 0x20,
    "enter": 0x0D,
    "esc": 0x1B,
    "escape": 0x1B,
    "shift": 0x10,
    "ctrl": 0x11,
    "control": 0x11,
    "alt": 0x12,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
}


class Actuator(Protocol):
    def send(self, action: str) -> None: ...


@dataclass
class DemoActuator:
    def send(self, action: str) -> None:
        return None


@dataclass
class DryRunKeyboardActuator:
    keymap: dict[str, str | None] = field(default_factory=lambda: dict(DEFAULT_KEYMAP))
    log_path: Path = Path("data/playmind/actuator_dryrun.jsonl")

    def send(self, action: str) -> None:
        key = self.keymap.get(action)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"t": time.time(), "action": action, "key": key, "mode": "dry_run"}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def _force_foreground(hwnd: int) -> bool:
    """Force a window into the foreground (AttachThreadInput trick)."""
    if sys.platform != "win32" or not hwnd:
        return False
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    hwnd_t = wintypes.HWND(hwnd)

    if user32.GetForegroundWindow() == hwnd_t:
        return True

    fg = user32.GetForegroundWindow()
    pid = wintypes.DWORD()
    tid_fg = user32.GetWindowThreadProcessId(fg, ctypes.byref(pid))
    tid_target = user32.GetWindowThreadProcessId(hwnd_t, ctypes.byref(pid))
    tid_current = kernel32.GetCurrentThreadId()

    attached_fg = False
    attached_tg = False
    try:
        if tid_fg and tid_fg != tid_current:
            attached_fg = bool(user32.AttachThreadInput(tid_current, tid_fg, True))
        if tid_target and tid_target != tid_current:
            attached_tg = bool(user32.AttachThreadInput(tid_current, tid_target, True))

        user32.ShowWindow(hwnd_t, 9)  # SW_RESTORE
        user32.BringWindowToTop(hwnd_t)
        user32.SetWindowPos(hwnd_t, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040)  # HWND_TOPMOST
        user32.SetWindowPos(hwnd_t, -2, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040)  # HWND_NOTOPMOST
        user32.SetForegroundWindow(hwnd_t)
        user32.SetFocus(hwnd_t)
        time.sleep(0.05)
    finally:
        if attached_tg:
            user32.AttachThreadInput(tid_current, tid_target, False)
        if attached_fg:
            user32.AttachThreadInput(tid_current, tid_fg, False)

    return int(user32.GetForegroundWindow()) == int(hwnd)


def _sendinput_key(vk: int, down: bool) -> None:
    import ctypes
    from ctypes import wintypes

    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_SCANCODE = 0x0008

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong),
        ]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class INPUT_UNION(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("union", INPUT_UNION)]

    scan = ctypes.windll.user32.MapVirtualKeyW(vk, 0)
    flags = KEYEVENTF_SCANCODE | (0 if down else KEYEVENTF_KEYUP)
    inp = INPUT(type=INPUT_KEYBOARD, union=INPUT_UNION(ki=KEYBDINPUT(0, scan, flags, 0, 0)))
    sent = ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if sent != 1:
        # Fallback to virtual-key form
        flags = 0 if down else KEYEVENTF_KEYUP
        inp = INPUT(type=INPUT_KEYBOARD, union=INPUT_UNION(ki=KEYBDINPUT(vk, scan, flags, 0, 0)))
        sent = ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if sent != 1:
            raise RuntimeError(f"SendInput failed for vk={vk} down={down}")


def _process_elevated(pid: int) -> bool | None:
    if sys.platform != "win32" or not pid:
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    advapi32 = ctypes.windll.advapi32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    TOKEN_QUERY = 0x0008
    TokenElevation = 20
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        token = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(handle, TOKEN_QUERY, ctypes.byref(token)):
            return None
        try:
            elev = wintypes.DWORD()
            ret = wintypes.DWORD()
            if not advapi32.GetTokenInformation(
                token, TokenElevation, ctypes.byref(elev), ctypes.sizeof(elev), ctypes.byref(ret)
            ):
                return None
            return bool(elev.value)
        finally:
            kernel32.CloseHandle(token)
    finally:
        kernel32.CloseHandle(handle)


def _self_elevated() -> bool:
    if sys.platform != "win32":
        return False
    try:
        return bool(ctypes_is_admin())
    except Exception:  # noqa: BLE001
        return False


def ctypes_is_admin() -> bool:
    import ctypes

    return bool(ctypes.windll.shell32.IsUserAnAdmin())


def _hwnd_pid(hwnd: int) -> int:
    import ctypes
    from ctypes import wintypes

    pid = wintypes.DWORD()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def _resolve_hwnd(window_title_substr: str) -> int | None:
    if not window_title_substr or sys.platform != "win32":
        return None
    from playmind.capture import find_window_bounds

    try:
        return find_window_bounds(window_title_substr, client_area=False).hwnd
    except Exception:  # noqa: BLE001
        return None


def assert_can_control_window(window_title_substr: str) -> None:
    """Raise a clear error if UIPI will block our keystrokes."""
    hwnd = _resolve_hwnd(window_title_substr)
    if not hwnd:
        raise RuntimeError(f"Game window not found: {window_title_substr!r}")
    game_elev = _process_elevated(_hwnd_pid(hwnd))
    self_elev = _self_elevated()
    if game_elev and not self_elev:
        raise SystemExit(
            "Ascension is running as Administrator, but PlayMind is not.\n"
            "Windows blocks keyboard input from a non-admin process into an admin game.\n"
            "Re-run elevated, e.g.:\n"
            "  powershell -Verb RunAs -File scripts/run_owned_live_admin.ps1"
        )


def _foreground_title() -> str:
    if sys.platform != "win32":
        return ""
    import ctypes

    hwnd = ctypes.windll.user32.GetForegroundWindow()
    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _click_screen(x: int, y: int) -> None:
    """Absolute screen click via SendInput (more reliable than mouse_event under DPI)."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    # Process is DPI-aware so SetCursorPos matches physical pixels.
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = (
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        )

    class INPUT(ctypes.Structure):
        _fields_ = (("type", wintypes.DWORD), ("mi", MOUSEINPUT))

    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    # Absolute SendInput uses 0..65535 normalized coords.
    abs_x = int(x * 65535 / max(1, screen_w - 1))
    abs_y = int(y * 65535 / max(1, screen_h - 1))
    extra = ctypes.pointer(ctypes.c_ulong(0))

    def _send(flags: int, ax: int = 0, ay: int = 0) -> None:
        inp = INPUT(type=0, mi=MOUSEINPUT(ax, ay, 0, flags, 0, extra))
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    MOVE = 0x0001 | 0x8000  # MOVE | ABSOLUTE
    LEFTDOWN, LEFTUP = 0x0002, 0x0004
    _send(MOVE, abs_x, abs_y)
    time.sleep(0.04)
    _send(LEFTDOWN)
    time.sleep(0.06)
    _send(LEFTUP)


@dataclass
class OwnedGameKeyboardActuator:
    """Opt-in OS keyboard sender for games you own.

    Uses Win32 SendInput and forces the target window foreground so keys
    are not swallowed by the IDE.
    """

    keymap: dict[str, str | None] = field(default_factory=lambda: dict(DEFAULT_KEYMAP))
    enabled: bool = False
    i_own_this_game: bool = False
    hold_seconds: float = 0.12
    move_hold_seconds: float = 1.0
    window_title_substr: str = ""
    # Fraction of client size for Release Spirit button (WoW-like death dialog).
    release_click_frac: tuple[float, float] = (0.47, 0.14)
    ui_memory: Any = None
    ability_memory: Any = None
    log_path: Path = Path("data/playmind/actuator_owned.jsonl")
    last_frame: Path | None = None  # for live OCR discovery on click_label

    def send(self, action: str) -> None:
        from playmind.ability_memory import parse_dynamic_action

        key = self.keymap.get(action)
        focused = False
        fg_title = ""
        hwnd = None
        dynamic = parse_dynamic_action(action)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        should_act = self.enabled and self.i_own_this_game and (
            key is not None or action == "release_spirit" or dynamic is not None
        )
        if should_act:
            hwnd = _resolve_hwnd(self.window_title_substr) if self.window_title_substr else None
            if hwnd:
                focused = _force_foreground(hwnd)
            fg_title = _foreground_title()
            if dynamic is not None:
                self._exec_dynamic(hwnd, dynamic)
            elif action == "release_spirit" or key == "__click_release__":
                self._click_release(hwnd)
            elif key is not None:
                self._tap(str(key), action)

        row = {
            "t": time.time(),
            "action": action,
            "key": key,
            "dynamic": dynamic,
            "mode": "owned_keyboard",
            "enabled": self.enabled,
            "i_own_this_game": self.i_own_this_game,
            "focused": focused,
            "fg": fg_title,
            "hwnd": hwnd,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _exec_dynamic(self, hwnd: int | None, dynamic: dict) -> None:
        dtype = dynamic.get("type")
        if dtype in {"click_frac", "click_label"}:
            self._click_dynamic(hwnd, dynamic)
            return
        if dtype == "bind":
            name = str(dynamic.get("name", ""))
            k = str(dynamic.get("key", ""))
            if self.ability_memory is not None and name and k:
                self.ability_memory.bind(name, k, source="llm")
            self._press_chord(k, self.hold_seconds)
            return
        if dtype == "ability":
            name = str(dynamic.get("name", ""))
            row = None
            if self.ability_memory is not None:
                row = self.ability_memory.lookup(name)
            if row and row.get("key"):
                hold = float(row.get("hold") or self.hold_seconds)
                self._press_chord(str(row["key"]), hold)
                return
            # Unknown ability → try digit if name ends with number, else primary attack key
            m = re.search(r"(\d+)$", name)
            self._press_chord(m.group(1) if m else "1", self.hold_seconds)
            return
        if dtype == "key":
            self._press_chord(str(dynamic.get("key", "")), self.hold_seconds)
            return
        if dtype == "hold":
            self._press_chord(
                str(dynamic.get("key", "")),
                float(dynamic.get("seconds") or self.hold_seconds),
            )
            return

    def _resolve_vk(self, name: str) -> int | None:
        name = name.lower().strip()
        if name in _VK:
            return _VK[name]
        if len(name) == 1:
            return ord(name.upper())
        return None

    def _press_chord(self, chord: str, hold: float) -> None:
        """Press key or modifier+key chord (e.g. shift+2)."""
        if not chord:
            return
        parts = [p for p in chord.lower().replace(" ", "").split("+") if p]
        if not parts:
            return
        mod_names = {"shift", "ctrl", "control", "alt"}
        mods = [p for p in parts[:-1] if p in mod_names]
        main = parts[-1]
        mod_vks = [self._resolve_vk(m) for m in mods]
        main_vk = self._resolve_vk(main)
        if main_vk is None:
            return
        for vk in mod_vks:
            if vk is not None:
                _sendinput_key(vk, True)
        _sendinput_key(main_vk, True)
        time.sleep(max(0.02, hold))
        _sendinput_key(main_vk, False)
        for vk in reversed(mod_vks):
            if vk is not None:
                _sendinput_key(vk, False)

    def _click_dynamic(self, hwnd: int | None, dynamic: dict) -> None:
        if hwnd is None:
            return
        from playmind.capture import find_window_bounds
        from playmind.ui_memory import resolve_click_target

        try:
            bounds = find_window_bounds(self.window_title_substr or "Ascension", client_area=True)
        except Exception:  # noqa: BLE001
            return
        fx = fy = None
        source = "frac"
        if dynamic.get("type") == "click_frac":
            fx, fy = float(dynamic["fx"]), float(dynamic["fy"])
        elif dynamic.get("type") == "click_label":
            label = str(dynamic.get("label", ""))
            resolved = resolve_click_target(self.last_frame, self.ui_memory, label)
            if resolved:
                fx, fy, source = resolved
            elif self.ui_memory is not None:
                hit = self.ui_memory.lookup(label)
                if hit:
                    fx, fy = hit
                    source = "memory"
        if fx is None or fy is None:
            # Discover from death-related OCR on the live frame — no hardcoded map.
            self._click_release(hwnd)
            return
        x = int(bounds.left + bounds.width * fx)
        y = int(bounds.top + bounds.height * fy)
        _click_screen(x, y)
        if self.ui_memory is not None and dynamic.get("type") == "click_label":
            self.ui_memory.remember(
                str(dynamic.get("label")), fx, fy, source=f"click:{source}"
            )
        time.sleep(0.1)

    def _click_release(self, hwnd: int | None) -> None:
        """Discover death UI via memory + live OCR; click candidates (no fixed map)."""
        if hwnd is None:
            return
        from playmind.capture import find_window_bounds
        from playmind.ui_memory import find_label_on_frame, resolve_click_target

        try:
            bounds = find_window_bounds(self.window_title_substr or "Ascension", client_area=True)
        except Exception:  # noqa: BLE001
            return

        labels = (
            "yes",
            "accept",
            "return to graveyard",
            "release spirit",
            "resurrect in a safe zone",
            "resurrect now",
            "closest town",
        )
        fracs: list[tuple[float, float]] = []
        for label in labels:
            resolved = resolve_click_target(self.last_frame, self.ui_memory, label)
            if resolved:
                fracs.append((resolved[0], resolved[1]))
                break  # one good hit is enough — don't spray 6 clicks
        if not fracs and self.last_frame is not None:
            for label in labels:
                hits = find_label_on_frame(self.last_frame, label)[:1]
                if hits:
                    fracs.append((hits[0].fx, hits[0].fy))
                    break
        if not fracs:
            return  # nothing discovered — don't spray random clicks
        fx, fy = fracs[0]
        x = int(bounds.left + bounds.width * fx)
        y = int(bounds.top + bounds.height * fy)
        _click_screen(x, y)
        time.sleep(0.06)

    def _tap(self, key: str, action: str) -> None:
        name = key.lower()
        vk = _VK.get(name)
        if vk is None and len(name) == 1:
            vk = ord(name.upper())
        if vk is None:
            raise RuntimeError(f"No VK mapping for key {key!r}")

        hold = self.move_hold_seconds if action.startswith("move_") else self.hold_seconds
        _sendinput_key(vk, True)
        time.sleep(hold)
        _sendinput_key(vk, False)


# Backward-compatible alias used by earlier CLI
@dataclass
class ParsecKeyboardActuator(OwnedGameKeyboardActuator):
    """Keyboard actuator intended for a focused Parsec/game window."""

    log_path: Path = Path("data/playmind/actuator_parsec.jsonl")


def load_keymap(path: Path | None) -> dict[str, str | None]:
    if path is None or not path.exists():
        return dict(DEFAULT_KEYMAP)
    data = json.loads(path.read_text(encoding="utf-8"))
    out = dict(DEFAULT_KEYMAP)
    out.update(data)
    return out
