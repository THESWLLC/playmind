"""Multi-model council: Actor acts fast; Teacher reviews failures and teaches.

They share one policy + lessons file so progress compounds instead of arguing
every tick.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playmind.ability_memory import parse_dynamic_action
from playmind.learning import OWNED_ACTIONS, OnlinePolicy


TEACHER_SYSTEM = """You are the Teacher brain for PlayMind (Ascension / WoW-like MMO).
The Actor just took an action that failed or scored poorly.
Propose a BETTER next action for the same situation.

Reply with EXACTLY one action line:
- known: move_north, attack, target_nearest, interact, wait, ...
- key:tab / key:1 / key:esc / hold:w:0.7
- click_label:Close / click_label:Accept
- bind:Name=key

Obey the GOAL. Prefer closing menus, targeting hostiles, attacking, or moving to find mobs.
No explanation — action line only."""


@dataclass
class Lesson:
    t: float
    goal: str
    bad_action: str
    reward: float
    better: str
    reason: str = ""
    ocr: str = ""


@dataclass
class TeacherBrain:
    """Text (or vision) model that critiques failures and boosts the Actor's Q."""

    model: str = "llama3.2"
    host: str = "http://127.0.0.1:11434"
    path: Path = Path("data/playmind/owned/lessons.jsonl")
    timeout_s: float = 45.0
    lessons: list[Lesson] = field(default_factory=list)
    last_raw: str = ""
    last_better: str = ""
    last_error: str = ""
    teaches: int = 0

    def maybe_teach(
        self,
        *,
        policy: OnlinePolicy,
        obs: dict[str, Any],
        action: str,
        reward: float,
        next_obs: dict[str, Any],
        goal_summary: str,
        actions: list[str],
        force: bool = False,
    ) -> str | None:
        """If the outcome was bad, ask Teacher for a better action and teach Q."""
        if not force and reward > 0.05:
            return None
        if not force and reward > -0.05 and self._stuck_signal(obs) is False:
            return None

        better = self._ask(obs, action, reward, next_obs, goal_summary)
        if not better or better == action:
            return None

        # Strongly boost the better action in this state; punish the bad one.
        policy.teach(obs, better, boost=1.2)
        policy.teach(obs, action, boost=-0.6)
        policy.update(obs, better, max(0.4, abs(reward) + 0.2), next_obs, False, actions)

        lesson = Lesson(
            t=time.time(),
            goal=goal_summary,
            bad_action=action,
            reward=reward,
            better=better,
            reason=self.last_raw[:200],
            ocr=(obs.get("screen_ocr") or "")[:160],
        )
        self.lessons.append(lesson)
        self._save_lesson(lesson)
        self.teaches += 1
        self.last_better = better
        return better

    @staticmethod
    def _stuck_signal(obs: dict[str, Any]) -> bool:
        hint = str(obs.get("stuck_hint") or "")
        return "failed" in hint or bool(obs.get("modal_menu"))

    def _ask(
        self,
        obs: dict[str, Any],
        action: str,
        reward: float,
        next_obs: dict[str, Any],
        goal_summary: str,
    ) -> str | None:
        payload = {
            "goal": goal_summary,
            "bad_action": action,
            "reward": reward,
            "hp": obs.get("vision_player_hp"),
            "has_target": obs.get("has_target"),
            "modal": obs.get("modal_menu"),
            "ocr": (obs.get("screen_ocr") or "")[:240],
            "next_hp": next_obs.get("vision_player_hp"),
            "next_target": next_obs.get("has_target"),
            "allowed_examples": list(OWNED_ACTIONS)[:8]
            + ["key:tab", "key:1", "key:esc", "hold:w:0.7", "click_label:Close"],
        }
        prompt = (
            TEACHER_SYSTEM
            + "\nSituation:\n"
            + json.dumps(payload, sort_keys=True)
            + "\nBetter action:"
        )
        try:
            text = self._generate(prompt)
            self.last_raw = text
            self.last_error = ""
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            self.last_raw = ""
            return None
        return self._parse(text)

    def _generate(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 48},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host.rstrip('/')}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return str(data.get("response", ""))

    def _parse(self, text: str) -> str | None:
        for line in text.strip().splitlines():
            line = line.strip().strip("`\"'")
            if not line:
                continue
            if ":" in line and line.lower().split(":", 1)[0].strip() in {
                "action",
                "better",
                "output",
                "answer",
            }:
                line = line.split(":", 1)[1].strip()
            dyn = parse_dynamic_action(line)
            if dyn:
                t = dyn.get("type")
                if t == "key":
                    return f"key:{dyn['key']}"
                if t == "hold":
                    return f"hold:{dyn['key']}:{dyn['seconds']}"
                if t == "click_label":
                    return f"click_label:{dyn['label']}"
                if t == "click_frac":
                    return f"click:{dyn['fx']:.3f},{dyn['fy']:.3f}"
                if t == "ability":
                    return f"ability:{dyn['name']}"
                if t == "bind":
                    return f"bind:{dyn['name']}={dyn['key']}"
            token = line.lower().replace("-", "_").split()[0].strip("`\"'.,: ")
            for a in OWNED_ACTIONS:
                if token == a:
                    return a
        low = text.lower().replace("-", "_")
        for a in OWNED_ACTIONS:
            if a in low:
                return a
        if "esc" in low:
            return "key:esc"
        if "tab" in low:
            return "key:tab"
        return None

    def _save_lesson(self, lesson: Lesson) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "t": lesson.t,
                        "goal": lesson.goal,
                        "bad_action": lesson.bad_action,
                        "reward": lesson.reward,
                        "better": lesson.better,
                        "reason": lesson.reason,
                        "ocr": lesson.ocr,
                    }
                )
                + "\n"
            )


@dataclass
class CouncilStatus:
    teaches: int = 0
    last_better: str = ""
    last_raw: str = ""
    last_error: str = ""
