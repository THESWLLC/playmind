"""Planners: heuristic bootstrap + optional Ollama LLM."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol

from playmind.demo_world import ACTIONS


class Planner(Protocol):
    def plan(self, obs: dict[str, Any], directive: str | None = None) -> str: ...


class HeuristicPlanner:
    """Deterministic quest/farm brain for the demo world."""

    def plan(self, obs: dict[str, Any], directive: str | None = None) -> str:
        d = (directive or "").strip().lower()
        if d in {"stop", "wait"}:
            return "wait"
        if d == "questlog":
            return "open_quest_log"

        if obs.get("quest_complete"):
            return "wait"

        # Prefer reading quest text once early.
        if obs.get("quest_kills", 0) == 0 and not obs.get("quest_log_open") and obs.get("steps", 0) < 2:
            return "open_quest_log"

        # Turn in when ready.
        if obs.get("quest_kills", 0) >= obs.get("quest_kills_needed", 99):
            if obs.get("npc_here"):
                return "interact"
            return _step_toward(obs["player"], obs["npc_pos"])

        # Combat if enemy adjacent.
        if obs.get("adjacent_enemies"):
            return "attack"

        if obs.get("herb_here") and d == "loot":
            return "loot"

        # Hunt nearest known wolf; if none, explore toward east camp approach.
        if obs.get("nearest_wolf"):
            return _step_toward(obs["player"], obs["nearest_wolf"])
        return _step_toward(obs["player"], {"x": 8, "y": 3})


def _step_toward(player: dict[str, int], target: dict[str, int]) -> str:
    dx = target["x"] - player["x"]
    dy = target["y"] - player["y"]
    if abs(dx) >= abs(dy) and dx != 0:
        return "move_east" if dx > 0 else "move_west"
    if dy != 0:
        return "move_south" if dy > 0 else "move_north"
    return "wait"


class OllamaPlanner:
    """Optional local LLM planner. Falls back if Ollama is down."""

    def __init__(
        self,
        model: str = "dolphin-llama3",
        host: str = "http://127.0.0.1:11434",
        fallback: Planner | None = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.fallback = fallback or HeuristicPlanner()

    def plan(self, obs: dict[str, Any], directive: str | None = None) -> str:
        prompt = {
            "system": (
                "You control a game character in an owned demo game. "
                f"Choose exactly one action from: {', '.join(ACTIONS)}. "
                "Reply with only the action name."
            ),
            "directive": directive,
            "state": obs,
        }
        try:
            text = self._generate(json.dumps(prompt))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return self.fallback.plan(obs, directive)

        cleaned = text.strip().split()[0].strip("`\"'.,").lower()
        # Allow model to return move_east etc.
        for action in ACTIONS:
            if cleaned == action or cleaned.replace("-", "_") == action:
                return action
        # Fuzzy contains
        for action in ACTIONS:
            if action in cleaned:
                return action
        return self.fallback.plan(obs, directive)

    def _generate(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return str(payload.get("response", ""))
