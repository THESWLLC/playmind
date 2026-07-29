"""Action actuators: demo (in-process), dry-run keyboard, Parsec-oriented stub.

These modules intentionally do NOT target World of Warcraft or any
restricted live MMO client. Wire them only to games you own.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


# Default demo keymap (logical action -> key name)
DEFAULT_KEYMAP = {
    "move_north": "w",
    "move_south": "s",
    "move_west": "a",
    "move_east": "d",
    "attack": "1",
    "loot": "f",
    "interact": "e",
    "open_quest_log": "l",
    "wait": None,
}


class Actuator(Protocol):
    def send(self, action: str) -> None: ...


@dataclass
class DemoActuator:
    """No-op for in-process demo world (world.step handles actions)."""

    def send(self, action: str) -> None:
        return None


@dataclass
class DryRunKeyboardActuator:
    """Logs intended keypresses without sending OS input."""

    keymap: dict[str, str | None] = field(default_factory=lambda: dict(DEFAULT_KEYMAP))
    log_path: Path = Path("data/playmind/actuator_dryrun.jsonl")

    def send(self, action: str) -> None:
        key = self.keymap.get(action)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"t": time.time(), "action": action, "key": key, "mode": "dry_run"}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


@dataclass
class ParsecKeyboardActuator:
    """Placeholder for Parsec/window keyboard control.

    Disabled by default. Enable only for games you own after installing
    a local input backend and confirming window focus rules.
    """

    keymap: dict[str, str | None] = field(default_factory=lambda: dict(DEFAULT_KEYMAP))
    enabled: bool = False
    window_title_substr: str = ""
    log_path: Path = Path("data/playmind/actuator_parsec.jsonl")

    def send(self, action: str) -> None:
        key = self.keymap.get(action)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "t": time.time(),
            "action": action,
            "key": key,
            "mode": "parsec_stub",
            "enabled": self.enabled,
            "window": self.window_title_substr,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if not self.enabled:
            return
        # Real OS key injection is opt-in and not bundled by default.
        raise RuntimeError(
            "ParsecKeyboardActuator.enabled is True, but no OS key backend is "
            "bundled. Add a local backend intentionally for your owned game."
        )


def load_keymap(path: Path | None) -> dict[str, str | None]:
    if path is None or not path.exists():
        return dict(DEFAULT_KEYMAP)
    data = json.loads(path.read_text(encoding="utf-8"))
    out = dict(DEFAULT_KEYMAP)
    out.update(data)
    return out
