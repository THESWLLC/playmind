"""Self-learning helpers: experience log + simple online action values."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def state_key(obs: dict[str, Any]) -> str:
    """Discretize observation into a stable key for tabular learning."""
    p = obs["player"]
    enemy = "none"
    if obs.get("adjacent_enemies"):
        e = obs["adjacent_enemies"][0]
        enemy = f"{e['name']}:{int(e['hp'] * 10)}"
    return "|".join(
        [
            f"pos:{p['x']},{p['y']}",
            f"hp:{int(p['hp'] * 5)}",
            f"enemy:{enemy}",
            f"herb:{int(bool(obs.get('herb_here')))}",
            f"npc:{int(bool(obs.get('npc_here')))}",
            f"qk:{obs.get('quest_kills', 0)}/{obs.get('quest_kills_needed', 0)}",
        ]
    )


@dataclass
class ExperienceBuffer:
    path: Path
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        obs: dict[str, Any],
        action: str,
        reward: float,
        next_obs: dict[str, Any],
        done: bool,
        source: str = "self",
    ) -> None:
        row = {
            "state": state_key(obs),
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_state": state_key(next_obs),
            "done": done,
            "source": source,
        }
        self.rows.append(row)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")

    def export_finetune_jsonl(self, out: Path, min_reward: float = 0.1) -> int:
        """Export successful steps as instruction pairs for later LLM fine-tunes."""
        out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with out.open("w", encoding="utf-8") as f:
            for row in self.rows:
                if row["reward"] < min_reward and row["source"] != "teacher":
                    continue
                sample = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a game agent. Reply with one action name only.",
                        },
                        {
                            "role": "user",
                            "content": json.dumps(row["obs"], sort_keys=True),
                        },
                        {"role": "assistant", "content": row["action"]},
                    ]
                }
                f.write(json.dumps(sample) + "\n")
                n += 1
        return n


@dataclass
class OnlinePolicy:
    """Epsilon-greedy tabular learner (learns alone from rewards)."""

    epsilon: float = 0.15
    alpha: float = 0.25
    gamma: float = 0.95
    q: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )

    def choose(self, obs: dict[str, Any], actions: list[str]) -> str:
        key = state_key(obs)
        if random.random() < self.epsilon:
            return random.choice(actions)
        values = self.q[key]
        best_v = None
        best_actions = []
        for a in actions:
            v = values.get(a, 0.0)
            if best_v is None or v > best_v:
                best_v = v
                best_actions = [a]
            elif v == best_v:
                best_actions.append(a)
        return random.choice(best_actions or actions)

    def update(
        self,
        obs: dict[str, Any],
        action: str,
        reward: float,
        next_obs: dict[str, Any],
        done: bool,
        actions: list[str],
    ) -> None:
        key = state_key(obs)
        next_key = state_key(next_obs)
        old = self.q[key][action]
        if done:
            target = reward
        else:
            next_best = max((self.q[next_key].get(a, 0.0) for a in actions), default=0.0)
            target = reward + self.gamma * next_best
        self.q[key][action] = old + self.alpha * (target - old)

    def teach(self, obs: dict[str, Any], action: str, boost: float = 1.0) -> None:
        """Strongly prefer a human-provided action in this state."""
        key = state_key(obs)
        self.q[key][action] += boost

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: dict(v) for k, v in self.q.items()}
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.q = defaultdict(lambda: defaultdict(float))
        for k, actions in raw.items():
            for a, v in actions.items():
                self.q[k][a] = float(v)
