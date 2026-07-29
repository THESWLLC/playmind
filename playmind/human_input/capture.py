"""Thread-safe, optional physical keyboard and mouse capture."""

from __future__ import annotations

import threading
import time
import warnings
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any, Literal

InputSource = Literal["human", "playmind_generated", "unknown"]
UnfocusedPolicy = Literal["ignore", "label"]

_MODIFIER_NAMES = {"alt", "alt_gr", "ctrl", "ctrl_l", "ctrl_r", "shift", "shift_l", "shift_r", "cmd", "cmd_l", "cmd_r"}
_GAME_KEYS = {"w", "a", "s", "d", "q", "e", "r", "f", "space", "tab"} | {str(i) for i in range(10)}
_MENU_KEYS = {"esc", "escape", "i", "b", "c", "m", "enter"}


def _key_name(key: Any) -> str:
    """Normalize pynput keys and simple test doubles to stable names."""
    char = getattr(key, "char", None)
    if char is not None:
        return str(char).lower()
    name = getattr(key, "name", None)
    if name is not None:
        return str(name).lower()
    text = str(key)
    if text.startswith("Key."):
        text = text[4:]
    return text.strip("'").lower()


def _button_name(button: Any) -> str:
    name = getattr(button, "name", None)
    if name is not None:
        return str(name).lower()
    text = str(button)
    return text[7:] if text.startswith("Button.") else text.lower()


class PhysicalInputCapture:
    """Capture physical input without making ``pynput`` a required dependency.

    Events are ordinary dictionaries so they can be written directly by
    :class:`playmind.demonstrations.DemonstrationRecorder`. Listener callbacks
    only mutate state while holding a lock; ``snapshot_and_clear`` atomically
    swaps the event buffer.
    """

    def __init__(
        self,
        *,
        source: InputSource = "human",
        focus_provider: Callable[[], bool] | None = None,
        unfocused_policy: UnfocusedPolicy = "label",
        ignore_unfocused: bool | None = None,
        normalize_coordinates: bool = False,
        normalize_coords: bool | None = None,
        screen_size: tuple[int, int] | Callable[[], tuple[int, int]] | None = None,
        window_size: tuple[int, int] | Callable[[], tuple[int, int]] | None = None,
        capture_mouse_moves: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ignore_unfocused is not None:
            unfocused_policy = "ignore" if ignore_unfocused else "label"
        if normalize_coords is not None:
            normalize_coordinates = normalize_coords
        if screen_size is None:
            screen_size = window_size
        if source not in {"human", "playmind_generated", "unknown"}:
            raise ValueError(f"unknown input source: {source!r}")
        if unfocused_policy not in {"ignore", "label"}:
            raise ValueError("unfocused_policy must be 'ignore' or 'label'")
        self.source = source
        self.focus_provider = focus_provider
        self.unfocused_policy = unfocused_policy
        self.normalize_coordinates = bool(normalize_coordinates)
        self.screen_size = screen_size
        self.capture_mouse_moves = bool(capture_mouse_moves)
        self.clock = clock

        self._events: deque[dict[str, Any]] = deque()
        self._lock = threading.RLock()
        self._pressed_at: dict[str, float] = {}
        self._modifiers: set[str] = set()
        self._last_mouse_position: tuple[float, float] | None = None
        self._keyboard_listener: Any = None
        self._mouse_listener: Any = None
        self._running = False
        self.available = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> bool:
        """Start pynput listeners, returning False when pynput is unavailable."""
        with self._lock:
            if self._running:
                return self.available
            try:
                from pynput import keyboard, mouse  # type: ignore[import-not-found]
            except ImportError:
                warnings.warn(
                    "pynput is not installed; physical input capture is disabled",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.available = False
                self._running = False
                return False

            self._keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
            self._mouse_listener = mouse.Listener(
                on_move=self._on_mouse_move,
                on_click=self._on_mouse_click,
                on_scroll=self._on_mouse_scroll,
            )
            self._keyboard_listener.start()
            self._mouse_listener.start()
            self.available = True
            self._running = True
            return True

    def stop(self) -> None:
        """Stop active listeners. Safe to call after a no-op start."""
        with self._lock:
            listeners = (self._keyboard_listener, self._mouse_listener)
            self._keyboard_listener = None
            self._mouse_listener = None
            self._running = False
        for listener in listeners:
            if listener is not None:
                try:
                    listener.stop()
                except (AttributeError, RuntimeError):
                    pass

    def snapshot_and_clear(self) -> list[dict[str, Any]]:
        """Atomically return buffered events and clear the queue."""
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return self._annotate_context(events)

    def record_event(self, event: Mapping[str, Any]) -> bool:
        """Add a synthetic/mock event through the same focus/source path."""
        payload = dict(event)
        payload.setdefault("timestamp", float(self.clock()))
        payload.setdefault("source", self.source)
        payload.setdefault("focused", self._focused())
        return self._enqueue(payload)

    def _focused(self) -> bool:
        if self.focus_provider is None:
            return True
        try:
            return bool(self.focus_provider())
        except Exception:  # focus detection must never stop listener threads
            return False

    def _enqueue(self, event: dict[str, Any]) -> bool:
        if not bool(event.get("focused", True)) and self.unfocused_policy == "ignore":
            return False
        event["source"] = str(event.get("source") or self.source)
        with self._lock:
            self._events.append(event)
        return True

    def _base_event(self, event_type: str) -> dict[str, Any]:
        return {
            "type": event_type,
            "timestamp": float(self.clock()),
            "source": self.source,
            "focused": self._focused(),
        }

    def _on_key_press(self, key: Any) -> None:
        name = _key_name(key)
        event = self._base_event("key_down")
        with self._lock:
            if name in _MODIFIER_NAMES:
                self._modifiers.add(name)
            event["key"] = name
            event["printable"] = len(name) == 1 and name.isprintable()
            event["modifiers"] = sorted(self._modifiers)
            self._pressed_at.setdefault(name, float(event["timestamp"]))
        self._enqueue(event)

    def _on_key_release(self, key: Any) -> None:
        name = _key_name(key)
        event = self._base_event("key_up")
        with self._lock:
            started = self._pressed_at.pop(name, None)
            event["key"] = name
            event["printable"] = len(name) == 1 and name.isprintable()
            event["modifiers"] = sorted(self._modifiers)
            event["duration"] = (
                max(0.0, float(event["timestamp"]) - started) if started is not None else None
            )
            event["duration_s"] = event["duration"]
            if name in _MODIFIER_NAMES:
                self._modifiers.discard(name)
        self._enqueue(event)

    def _coordinates(self, x: float, y: float) -> dict[str, float]:
        values = {"x": float(x), "y": float(y)}
        if not self.normalize_coordinates or self.screen_size is None:
            return values
        size = self.screen_size() if callable(self.screen_size) else self.screen_size
        width, height = size
        if width > 0 and height > 0:
            values["x_normalized"] = min(1.0, max(0.0, float(x) / float(width)))
            values["y_normalized"] = min(1.0, max(0.0, float(y) / float(height)))
            values["normalized_x"] = values["x_normalized"]
            values["normalized_y"] = values["y_normalized"]
        return values

    def _on_mouse_move(self, x: float, y: float) -> None:
        if not self.capture_mouse_moves:
            return
        event = self._base_event("mouse_move")
        with self._lock:
            previous = self._last_mouse_position
            self._last_mouse_position = (float(x), float(y))
        event.update(self._coordinates(x, y))
        event["dx"] = 0.0 if previous is None else float(x) - previous[0]
        event["dy"] = 0.0 if previous is None else float(y) - previous[1]
        self._enqueue(event)

    def _on_mouse_click(self, x: float, y: float, button: Any, pressed: bool) -> None:
        event = self._base_event("mouse_button")
        event.update(self._coordinates(x, y))
        event.update({"button": _button_name(button), "pressed": bool(pressed)})
        self._enqueue(event)

    def _on_mouse_scroll(self, x: float, y: float, dx: float, dy: float) -> None:
        event = self._base_event("mouse_wheel")
        event.update(self._coordinates(x, y))
        event.update({"dx": float(dx), "dy": float(dy)})
        self._enqueue(event)

    @staticmethod
    def _annotate_context(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Label likely typing and menu interactions without dropping evidence."""
        key_down = [e for e in events if e.get("type") == "key_down"]
        printable = [
            e
            for e in key_down
            if e.get("printable") and str(e.get("key", "")).lower() not in _GAME_KEYS
        ]
        game_actions = [
            e
            for e in key_down
            if str(e.get("key", "")).lower() in _GAME_KEYS
        ] + [
            e
            for e in events
            if e.get("type") == "mouse_button" and e.get("pressed")
        ]
        typing_ids: set[int] = set()
        for index, event in enumerate(printable):
            start = float(event.get("timestamp") or 0.0)
            burst = [
                candidate
                for candidate in printable[index:]
                if float(candidate.get("timestamp") or 0.0) - start <= 1.5
            ]
            if len(burst) < 4:
                continue
            end = float(burst[-1].get("timestamp") or start)
            if not any(
                start <= float(action.get("timestamp") or 0.0) <= end
                for action in game_actions
            ):
                typing_ids.update(id(item) for item in burst)
        for event in events:
            heuristics = list(event.get("heuristics") or [])
            if id(event) in typing_ids:
                heuristics.append("likely_chat_or_typing")
            key = str(event.get("key") or "").lower()
            if (
                event.get("type") == "key_down"
                and key in _MENU_KEYS
                or event.get("type") in {"mouse_button", "mouse_wheel"}
                and not bool(event.get("focused", True))
            ):
                heuristics.append("likely_menu_interaction")
            if heuristics:
                event["heuristics"] = sorted(set(heuristics))
        return events


__all__ = ["InputSource", "PhysicalInputCapture", "UnfocusedPolicy"]
