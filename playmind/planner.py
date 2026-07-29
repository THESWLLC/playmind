"""Planners: heuristic bootstrap + optional Ollama LLM."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol, Sequence

from playmind.demo_world import ACTIONS
from playmind.learning import OWNED_ACTIONS


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
    px = int(player.get("x", 0))
    py = int(player.get("y", 0))
    dx = int(target.get("x", 0)) - px
    dy = int(target.get("y", 0)) - py
    if abs(dx) >= abs(dy) and dx != 0:
        return "move_east" if dx > 0 else "move_west"
    if dy != 0:
        return "move_south" if dy > 0 else "move_north"
    return "wait"


OWNED_SYSTEM = (
    "You are PlayMind controlling an owned game character (Ascension). "
    "You receive sensors AND optional screen_ocr text from the live frame. "
    "Reply with EXACTLY one action name from the allowed list. "
    "If screen_ocr or sensors mention Release Spirit, Return to Graveyard, Resurrect, or Accept while dead → release_spirit. "
    "If has_target and healthy → attack. "
    "If alive with no target → target_nearest then move. "
    "Never invent actions. Reply with only the action name."
)


class OllamaPlanner:
    """Optional local LLM planner. Falls back if Ollama is down."""

    def __init__(
        self,
        model: str = "dolphin-llama3",
        host: str = "http://127.0.0.1:11434",
        fallback: Planner | None = None,
        actions: Sequence[str] | None = None,
        system: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.fallback = fallback or HeuristicPlanner()
        self.actions = tuple(actions) if actions else ACTIONS
        self.system = system or (
            "You control a game character in an owned demo game. "
            f"Choose exactly one action from: {', '.join(self.actions)}. "
            "Reply with only the action name."
        )
        self.timeout_s = timeout_s
        self.last_raw: str = ""
        self.last_error: str = ""

    @classmethod
    def for_owned(
        cls,
        model: str = "llama3.2",
        host: str = "http://127.0.0.1:11434",
        fallback: Planner | None = None,
    ) -> "OllamaPlanner":
        return cls(
            model=model,
            host=host,
            fallback=fallback,
            actions=OWNED_ACTIONS,
            system=OWNED_SYSTEM + f" Allowed: {', '.join(OWNED_ACTIONS)}.",
        )

    def plan(self, obs: dict[str, Any], directive: str | None = None) -> str:
        # Hard safety: death always releases before asking the model.
        if obs.get("is_dead") or obs.get("ghost_buttons"):
            return "release_spirit"

        slim = {
            k: obs.get(k)
            for k in (
                "vision_player_hp",
                "has_target",
                "in_combat",
                "is_dead",
                "is_ghost",
                "desaturated",
                "ghost_buttons",
                "motion",
                "target_hp_est",
                "quest_text",
                "vision_quest_text",
                "screen_ocr",
                "steps",
            )
            if obs.get(k) is not None
        }
        slim["player_hp"] = (obs.get("player") or {}).get("hp")
        prompt = (
            f"{self.system}\n"
            f"directive={directive!r}\n"
            f"state={json.dumps(slim, sort_keys=True)}\n"
            "action:"
        )
        try:
            text = self._generate(prompt)
            self.last_raw = text
            self.last_error = ""
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            self.last_error = str(exc)
            return self.fallback.plan(obs, directive)

        cleaned = text.strip().split()[0].strip("`\"'.,").lower().replace("-", "_")
        for action in self.actions:
            if cleaned == action:
                return action
        for action in self.actions:
            if action in cleaned:
                return action
        return self.fallback.plan(obs, directive)

    def _generate(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 16},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return str(payload.get("response", ""))


def ollama_available(host: str = "http://127.0.0.1:11434", timeout_s: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False
