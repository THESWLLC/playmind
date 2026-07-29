#!/usr/bin/env python3
"""PlayMind one-file shareable demo (stdlib only).

Run:
  python playmind_onefile.py
  python playmind_onefile.py --episodes 5

This is a self-contained owned-game agent demo. It does not interact with
World of Warcraft or any other commercial MMO client.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ACTIONS = [
    "move_north",
    "move_south",
    "move_east",
    "move_west",
    "attack",
    "loot",
    "interact",
    "open_quest_log",
    "wait",
]


@dataclass
class World:
    width: int = 12
    height: int = 8
    player: tuple[int, int] = (1, 1)
    hp: float = 1.0
    monsters: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    npc: tuple[int, int] = (10, 6)
    quest_kills: int = 0
    need: int = 3
    done: bool = False
    steps: int = 0

    def __post_init__(self) -> None:
        if not self.monsters:
            self.monsters = {
                (4, 2): {"name": "Wolf", "hp": 1.0},
                (6, 4): {"name": "Wolf", "hp": 1.0},
                (8, 1): {"name": "Wolf", "hp": 1.0},
            }

    def obs(self) -> dict[str, Any]:
        x, y = self.player
        adj = []
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (x + dx, y + dy)
            if p in self.monsters:
                adj.append(self.monsters[p] | {"pos": p})
        nearest = None
        best = None
        for pos, m in self.monsters.items():
            if m["name"] != "Wolf":
                continue
            dist = abs(pos[0] - x) + abs(pos[1] - y)
            if best is None or dist < best:
                best = dist
                nearest = {"x": pos[0], "y": pos[1]}
        return {
            "player": {"x": x, "y": y, "hp": self.hp},
            "adjacent_enemies": adj,
            "nearest_wolf": nearest,
            "npc_here": self.player == self.npc,
            "npc_pos": {"x": self.npc[0], "y": self.npc[1]},
            "quest_kills": self.quest_kills,
            "quest_kills_needed": self.need,
            "quest_complete": self.done,
            "steps": self.steps,
        }

    def render(self) -> str:
        out = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                p = (x, y)
                if p == self.player:
                    row.append("@")
                elif p == self.npc:
                    row.append("N")
                elif p in self.monsters:
                    row.append("W")
                else:
                    row.append(".")
            out.append("".join(row))
        return "\n".join(out)

    def step(self, action: str) -> float:
        self.steps += 1
        reward = -0.01
        x, y = self.player
        if action.startswith("move_"):
            dx, dy = {
                "move_north": (0, -1),
                "move_south": (0, 1),
                "move_east": (1, 0),
                "move_west": (-1, 0),
            }[action]
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in self.monsters:
                self.player = (nx, ny)
        elif action == "attack":
            target = None
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (x + dx, y + dy)
                if p in self.monsters:
                    target = p
                    break
            if target:
                self.monsters[target]["hp"] -= 0.55
                self.hp -= 0.08
                reward = 0.15
                if self.monsters[target]["hp"] <= 0:
                    if self.monsters[target]["name"] == "Wolf":
                        self.quest_kills += 1
                    del self.monsters[target]
                    reward = 1.0
        elif action == "interact" and self.player == self.npc:
            if self.quest_kills >= self.need:
                self.done = True
                reward = 2.0
        return reward


def state_key(o: dict[str, Any]) -> str:
    p = o["player"]
    e = o["adjacent_enemies"][0]["name"] if o["adjacent_enemies"] else "none"
    return f"{p['x']},{p['y']}|{e}|{o['quest_kills']}|{int(o['npc_here'])}"


class Policy:
    def __init__(self) -> None:
        self.q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.eps = 0.15

    def heuristic(self, o: dict[str, Any]) -> str:
        if o["adjacent_enemies"]:
            return "attack"
        if o["quest_kills"] >= o["quest_kills_needed"]:
            if o["npc_here"]:
                return "interact"
            dx = o["npc_pos"]["x"] - o["player"]["x"]
            dy = o["npc_pos"]["y"] - o["player"]["y"]
            if abs(dx) >= abs(dy) and dx:
                return "move_east" if dx > 0 else "move_west"
            if dy:
                return "move_south" if dy > 0 else "move_north"
            return "wait"
        if o.get("nearest_wolf"):
            dx = o["nearest_wolf"]["x"] - o["player"]["x"]
            dy = o["nearest_wolf"]["y"] - o["player"]["y"]
            if abs(dx) >= abs(dy) and dx:
                return "move_east" if dx > 0 else "move_west"
            if dy:
                return "move_south" if dy > 0 else "move_north"
        return "move_east"

    def act(self, o: dict[str, Any], use_learned: bool = False) -> str:
        # Default: reliable heuristic. Learning still updates Q in the background.
        if not use_learned:
            return self.heuristic(o)
        key = state_key(o)
        if random.random() < self.eps:
            return random.choice(ACTIONS)
        return max(ACTIONS, key=lambda a: self.q[key].get(a, 0.0))

    def learn(self, o, a, r, n, done) -> None:
        key, nkey = state_key(o), state_key(n)
        old = self.q[key][a]
        nxt = 0.0 if done else max(self.q[nkey].get(x, 0.0) for x in ACTIONS)
        self.q[key][a] = old + 0.25 * ((r + 0.95 * nxt) - old)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--save", default="data/playmind_onefile_policy.json")
    args = parser.parse_args()
    policy = Policy()
    wins = 0
    for ep in range(1, args.episodes + 1):
        w = World()
        for _ in range(args.max_steps):
            o = w.obs()
            a = policy.act(o)
            r = w.step(a)
            n = w.obs()
            policy.learn(o, a, r, n, w.done)
            if w.done:
                wins += 1
                break
        print(f"Episode {ep}: {'WIN' if w.done else 'fail'} steps={w.steps} kills={w.quest_kills}")
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save).write_text(
        json.dumps({k: dict(v) for k, v in policy.q.items()}), encoding="utf-8"
    )
    print(f"Wins {wins}/{args.episodes}. Saved {args.save}")
    print("Legend: @ you, W wolf, N quest NPC")


if __name__ == "__main__":
    main()
