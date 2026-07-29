"""Minimal owned grid world for agent development (not any commercial MMO client)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ACTIONS = (
    "move_north",
    "move_south",
    "move_east",
    "move_west",
    "attack",
    "loot",
    "interact",
    "open_quest_log",
    "wait",
)


@dataclass
class DemoWorld:
    width: int = 12
    height: int = 8
    player: tuple[int, int] = (1, 1)
    player_hp: float = 1.0
    monsters: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    herbs: set[tuple[int, int]] = field(default_factory=set)
    npc: tuple[int, int] = (10, 6)
    quest_kills_needed: int = 3
    quest_kills: int = 0
    quest_complete: bool = False
    inventory_loot: int = 0
    steps: int = 0
    log: list[str] = field(default_factory=list)
    quest_log_open: bool = False

    def __post_init__(self) -> None:
        if not self.monsters:
            self.monsters = {
                (4, 2): {"id": "wolf_1", "name": "Wolf", "hp": 1.0},
                (6, 4): {"id": "wolf_2", "name": "Wolf", "hp": 1.0},
                (8, 1): {"id": "wolf_3", "name": "Wolf", "hp": 1.0},
                (3, 6): {"id": "boar_1", "name": "Boar", "hp": 1.0},
            }
        if not self.herbs:
            self.herbs = {(2, 3), (7, 5), (9, 3)}

    def observe(self) -> dict[str, Any]:
        x, y = self.player
        adjacent = []
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)):
            pos = (x + dx, y + dy)
            if pos in self.monsters:
                m = self.monsters[pos]
                adjacent.append({"pos": pos, "name": m["name"], "hp": m["hp"]})
        # Nearest living wolf helps planners without exposing full fog-of-war cheats later.
        nearest_wolf = None
        best = None
        for pos, m in self.monsters.items():
            if m["name"] != "Wolf":
                continue
            dist = abs(pos[0] - x) + abs(pos[1] - y)
            if best is None or dist < best:
                best = dist
                nearest_wolf = {"x": pos[0], "y": pos[1], "name": m["name"]}
        quest_text = (
            f"Kill {self.quest_kills_needed} Wolves ({self.quest_kills}/{self.quest_kills_needed}). "
            f"Then talk to Mira at the east camp."
        )
        return {
            "player": {"x": x, "y": y, "hp": round(self.player_hp, 2)},
            "adjacent_enemies": adjacent,
            "nearest_wolf": nearest_wolf,
            "herb_here": self.player in self.herbs,
            "npc_here": self.player == self.npc,
            "npc_pos": {"x": self.npc[0], "y": self.npc[1]},
            "quest_log_open": self.quest_log_open,
            "quest_text": quest_text if self.quest_log_open else None,
            "quest_kills": self.quest_kills,
            "quest_kills_needed": self.quest_kills_needed,
            "quest_complete": self.quest_complete,
            "inventory_loot": self.inventory_loot,
            "steps": self.steps,
            "monsters_alive": len(self.monsters),
            "last_events": self.log[-5:],
        }

    def _move(self, dx: int, dy: int) -> str:
        x, y = self.player
        nx, ny = x + dx, y + dy
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return "bump_edge"
        if (nx, ny) in self.monsters:
            return "blocked_by_enemy"
        self.player = (nx, ny)
        return "moved"

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Apply action. Returns observe, reward, done, info."""
        self.steps += 1
        self.quest_log_open = False
        reward = -0.01
        info: dict[str, Any] = {"action": action, "result": "ok"}

        if action == "move_north":
            info["result"] = self._move(0, -1)
        elif action == "move_south":
            info["result"] = self._move(0, 1)
        elif action == "move_east":
            info["result"] = self._move(1, 0)
        elif action == "move_west":
            info["result"] = self._move(-1, 0)
        elif action == "attack":
            reward, info["result"] = self._attack()
        elif action == "loot":
            if self.player in self.herbs:
                self.herbs.remove(self.player)
                self.inventory_loot += 1
                reward = 0.3
                info["result"] = "looted_herb"
                self.log.append("Picked herb.")
            else:
                info["result"] = "nothing_to_loot"
                reward = -0.05
        elif action == "interact":
            reward, info["result"] = self._interact()
        elif action == "open_quest_log":
            self.quest_log_open = True
            info["result"] = "quest_log_open"
            reward = 0.02
        elif action == "wait":
            info["result"] = "waited"
        else:
            info["result"] = "invalid_action"
            reward = -0.1

        # Light damage if standing on enemy somehow (shouldn't), else ambient.
        if self.player_hp <= 0:
            self.player_hp = 0.0
            self.log.append("Player defeated.")
            return self.observe(), reward - 1.0, True, info

        done = self.quest_complete
        if done:
            reward += 2.0
            self.log.append("Quest complete!")
        return self.observe(), reward, done, info

    def _attack(self) -> tuple[float, str]:
        x, y = self.player
        target_pos = None
        for dx, dy in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)):
            pos = (x + dx, y + dy)
            if pos in self.monsters:
                target_pos = pos
                break
        if target_pos is None:
            return -0.05, "no_target"

        monster = self.monsters[target_pos]
        monster["hp"] -= 0.55
        self.player_hp -= 0.08
        self.log.append(f"Hit {monster['name']} ({monster['hp']:.2f} hp).")
        if monster["hp"] <= 0:
            name = monster["name"]
            del self.monsters[target_pos]
            if name == "Wolf":
                self.quest_kills += 1
            self.inventory_loot += 1
            self.log.append(f"Defeated {name}.")
            return 1.0, "killed"
        return 0.15, "hit"

    def _interact(self) -> tuple[float, str]:
        if self.player != self.npc:
            return -0.05, "no_npc"
        if self.quest_kills >= self.quest_kills_needed:
            self.quest_complete = True
            self.log.append("Mira: Quest turned in. Well done!")
            return 1.5, "quest_turned_in"
        self.log.append("Mira: Still need more Wolf pelts.")
        return 0.05, "npc_talk_incomplete"

    def render_ascii(self) -> str:
        rows = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)
                if pos == self.player:
                    row.append("@")
                elif pos == self.npc:
                    row.append("N")
                elif pos in self.monsters:
                    row.append("W" if self.monsters[pos]["name"] == "Wolf" else "B")
                elif pos in self.herbs:
                    row.append("*")
                else:
                    row.append(".")
            rows.append("".join(row))
        return "\n".join(rows)
