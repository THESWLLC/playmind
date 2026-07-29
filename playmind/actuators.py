"""Action actuators for demo and owned-game keyboard control.

Wire only to games you own / are allowed to automate.
Do NOT use with World of Warcraft or other restricted live MMO clients.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


DEFAULT_KEYMAP = {
    "move_north": "w",
    "move_south": "s",
    "move_west": "a",
    "move_east": "d",
    "attack": "1",
    "loot": "f",
    "interact": "e",
    "open_quest_log": "l",
    "logout": "f12",
    "wait": None,
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


@dataclass
class OwnedGameKeyboardActuator:
    """Opt-in OS keyboard sender for games you own.

    Requires:
      pip install pynput
      i_own_this_game=True
      enabled=True
    """

    keymap: dict[str, str | None] = field(default_factory=lambda: dict(DEFAULT_KEYMAP))
    enabled: bool = False
    i_own_this_game: bool = False
    hold_seconds: float = 0.08
    log_path: Path = Path("data/playmind/actuator_owned.jsonl")

    def send(self, action: str) -> None:
        key = self.keymap.get(action)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "t": time.time(),
            "action": action,
            "key": key,
            "mode": "owned_keyboard",
            "enabled": self.enabled,
            "i_own_this_game": self.i_own_this_game,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if key is None:
            return
        if not self.enabled or not self.i_own_this_game:
            return

        try:
            from pynput.keyboard import Controller, Key  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install pynput: pip install pynput") from exc

        keyboard = Controller()
        special = {
            "enter": Key.enter,
            "esc": Key.esc,
            "space": Key.space,
            "tab": Key.tab,
            "f12": Key.f12,
            "f1": Key.f1,
        }
        press = special.get(key.lower(), key)
        keyboard.press(press)
        time.sleep(self.hold_seconds)
        keyboard.release(press)


# Backward-compatible alias used by earlier CLI
@dataclass
class ParsecKeyboardActuator(OwnedGameKeyboardActuator):
    """Keyboard actuator intended for a focused Parsec/game window."""

    window_title_substr: str = ""
    log_path: Path = Path("data/playmind/actuator_parsec.jsonl")

    def send(self, action: str) -> None:
        # Reuse owned-game sender; window focus is the operator's responsibility.
        super().send(action)


def load_keymap(path: Path | None) -> dict[str, str | None]:
    if path is None or not path.exists():
        return dict(DEFAULT_KEYMAP)
    data = json.loads(path.read_text(encoding="utf-8"))
    out = dict(DEFAULT_KEYMAP)
    out.update(data)
    return out
