"""Owned-game observe→plan→act loop using capture + vision + keyboard.

Demo mode remains default. This loop is for your own game window / Parsec.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from playmind.actuators import (
    Actuator,
    DryRunKeyboardActuator,
    OwnedGameKeyboardActuator,
    load_keymap,
)
from playmind.capture import capture_config_from_dict
from playmind.planner import HeuristicPlanner, OllamaPlanner
from playmind.session import SessionConfig, SessionScheduler
from playmind.vision import read_frame


@dataclass
class OwnedLoopConfig:
    config_path: Path = Path("config/owned_game.json")
    dry_run: bool = True
    use_ollama: bool = False
    ollama_model: str = "dolphin-llama3"
    tick_seconds: float = 0.35
    max_ticks: int = 0  # 0 = until session stop
    data_dir: Path = Path("data/playmind/owned")


def load_owned_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Copy config/owned_game.example.json and edit it."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def vision_obs_from_frame(frame_path: Path, rois: dict[str, Any]) -> dict[str, Any]:
    hp_roi = None
    if "hp_roi" in rois:
        r = rois["hp_roi"]
        hp_roi = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
    reading = read_frame(frame_path, hp_roi=hp_roi)
    obs: dict[str, Any] = {
        "player": {"x": 0, "y": 0, "hp": reading.player_hp if reading.player_hp is not None else 1.0},
        "adjacent_enemies": [],
        "npc_here": False,
        "npc_pos": {"x": 0, "y": 0},
        "quest_kills": 0,
        "quest_kills_needed": 0,
        "quest_complete": False,
        "steps": 0,
        "quest_text": reading.quest_text,
    }
    obs.update(reading.to_obs_patch())
    return obs


@dataclass
class OwnedGameLoop:
    cfg: OwnedLoopConfig = field(default_factory=OwnedLoopConfig)
    directive: str | None = None
    on_status: Callable[[dict[str, Any]], None] | None = None

    def run(self) -> None:
        owned = load_owned_config(self.cfg.config_path)
        if not owned.get("i_own_this_game", False):
            raise SystemExit(
                "Refusing to run: set i_own_this_game=true in owned_game.json "
                "only for games you own / may automate."
            )

        keymap = load_keymap(Path(owned.get("keymap_path", "config/keymap.example.json")))
        capture_cfg = owned.get("capture", {"mode": "monitor", "monitor_index": 1})
        rois = owned.get("rois", {})
        session_cfg = SessionConfig(**owned.get("session", {}))
        scheduler = SessionScheduler(config=session_cfg)

        actuator: Actuator
        if self.cfg.dry_run or not owned.get("enable_keyboard", False):
            actuator = DryRunKeyboardActuator(
                keymap=keymap, log_path=self.cfg.data_dir / "dryrun.jsonl"
            )
            mode = "dry_run"
        else:
            actuator = OwnedGameKeyboardActuator(
                keymap=keymap,
                enabled=True,
                i_own_this_game=True,
                log_path=self.cfg.data_dir / "keys.jsonl",
            )
            mode = "live_keyboard"

        planner = (
            OllamaPlanner(model=self.cfg.ollama_model)
            if self.cfg.use_ollama
            else HeuristicPlanner()
        )

        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        ticks = 0
        print(f"Owned loop starting mode={mode} config={self.cfg.config_path}")

        while True:
            if scheduler.should_stop():
                print("Max session wall time reached; stopping.")
                break
            if self.cfg.max_ticks and ticks >= self.cfg.max_ticks:
                print("Max ticks reached; stopping.")
                break

            if scheduler.should_start_break():
                mins = scheduler.start_break()
                print(f"Break started (~{mins:.1f} min). Sending logout action.")
                actuator.send(owned.get("logout_action", "logout"))
                while not scheduler.break_done():
                    time.sleep(1.0)
                scheduler.end_break()
                print("Break over; resuming.")

            frame_path = self.cfg.data_dir / "latest.png"
            try:
                cap = capture_config_from_dict(capture_cfg, frame_path)
            except Exception as exc:  # noqa: BLE001
                print(f"capture failed: {exc}")
                time.sleep(1.0)
                continue

            obs = vision_obs_from_frame(cap.path, rois)
            # Heuristic planner expects demo fields; for owned games prefer
            # vision quest text + simple directives.
            action = self._plan_owned(planner, obs, self.directive)
            actuator.send(action)
            ticks += 1

            status = {
                "tick": ticks,
                "action": action,
                "capture": cap.backend,
                "vision_hp": obs.get("vision_player_hp"),
                "vision_quest": obs.get("vision_quest_text"),
                "session": scheduler.status(),
                "mode": mode,
            }
            if self.on_status:
                self.on_status(status)
            else:
                print(status)
            time.sleep(self.cfg.tick_seconds)

    def _plan_owned(self, planner, obs: dict[str, Any], directive: str | None) -> str:
        d = (directive or "").strip().lower()
        if d in {"stop", "wait"}:
            return "wait"
        if d == "logout":
            return "logout"
        # If we only have vision text, keep a cautious default policy.
        if obs.get("vision_player_hp") is not None and obs["vision_player_hp"] < 0.25:
            return "wait"
        if "talk" in (obs.get("quest_text") or "").lower():
            return "interact"
        if "kill" in (obs.get("quest_text") or "").lower():
            return "attack"
        # Fall back to heuristic on the synthetic obs (may mostly explore/wait).
        try:
            return planner.plan(obs, directive)
        except Exception:
            return "wait"
