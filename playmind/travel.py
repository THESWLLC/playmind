"""Travel memory: discover a heading, commit forward, remember walls.

Stops east↔west thrash and “attack in place forever” when nothing is dying.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


_DIRS = ("north", "east", "south", "west")
_HOLD = {
    "north": "hold:w:1.2",
    "south": "hold:s:1.2",
    "east": "hold:d:1.2",
    "west": "hold:a:1.2",
}
_MOVE = {
    "north": "move_north",
    "south": "move_south",
    "east": "move_east",
    "west": "move_west",
}
_LEFT = {"north": "west", "west": "south", "south": "east", "east": "north"}
_RIGHT = {"north": "east", "east": "south", "south": "west", "west": "north"}
_BACK = {"north": "south", "south": "north", "east": "west", "west": "east"}


@dataclass
class TravelMemory:
    """Sticky wander — try a direction, measure motion, remember dead ends."""

    heading: str = "east"
    commit_left: int = 0
    low_motion: int = 0
    still_farm: int = 0
    heading_ticks: int = 0
    tab_miss: int = 0
    expect_target: bool = False  # after Tab, swing once even if sensor blind
    blocked: dict[str, float] = field(default_factory=dict)
    successes: dict[str, int] = field(default_factory=dict)
    tick: int = 0

    def note(
        self,
        action: str,
        *,
        motion: float,
        had_target: bool,
        reward: float,
        tick: int,
    ) -> None:
        self.tick = tick
        a = (action or "").lower()
        moved = a.startswith("move_") or a.startswith("hold:")
        combat = a in {"attack", "target_nearest"} or a.startswith("key:")

        if a in {"key:tab", "target_nearest"}:
            self.expect_target = True
            if had_target:
                self.tab_miss = 0
            else:
                self.tab_miss += 1

        if moved:
            self.heading_ticks += 1
            if motion < 2.0:
                self.low_motion += 1
            else:
                self.low_motion = max(0, self.low_motion - 1)
                for d, hold in _HOLD.items():
                    if a.startswith(hold[:6]) or a == _MOVE.get(d, ""):
                        self.successes[d] = self.successes.get(d, 0) + 1
                        break
            # Only declare a wall after several stuck steps — don't spin every 2 ticks.
            if self.low_motion >= 5:
                self.blocked[self.heading] = float(tick + 50)
                self.heading = _LEFT[self.heading]  # always turn left at walls
                self.commit_left = 4
                self.low_motion = 0
                self.heading_ticks = 0

        if combat and had_target and reward < 0.7:
            self.still_farm += 1
        elif moved and motion >= 2.0:
            self.still_farm = 0
        elif reward >= 0.9:
            self.still_farm = 0

        if had_target and reward >= 0.5:
            self.heading_ticks = 0
            self.tab_miss = 0

    def needs_travel(self, obs: dict[str, Any]) -> bool:
        if obs.get("is_dead") or obs.get("is_ghost") or obs.get("modal_menu"):
            return False
        if obs.get("life_phase") and obs.get("life_phase") != "alive":
            return False
        # Progressive curriculum / false-target escape.
        if obs.get("forced_travel") or obs.get("progress_stage") in {"push", "break_loop"}:
            return True
        if int(obs.get("no_damage_casts") or 0) >= 3:
            return True
        if int(obs.get("stagnant") or 0) >= 5:
            return True
        if not obs.get("has_target") and not obs.get("in_combat"):
            return True
        if self.still_farm >= 5:
            return True
        if self.low_motion >= 5:
            return True
        return False

    def action(self, obs: dict[str, Any]) -> tuple[str, str]:
        """Return (action, reason) — long commits so we leave the bubble."""
        if self.heading_ticks >= 22:
            self.blocked[self.heading] = float(self.tick + 40)
            self.heading = random.choice([_LEFT[self.heading], _RIGHT[self.heading]])
            self.heading_ticks = 0
            self.commit_left = 0

        if self.commit_left > 0:
            self.commit_left -= 1
            act = _HOLD[self.heading]
            return act, f"travel:{self.heading} commit→{act}"

        # Fresh pick only when not mid-corridor.
        if self.heading_ticks < 2:
            self.heading = self._pick_fresh()
        self.commit_left = random.randint(5, 8)
        act = _HOLD[self.heading]
        return act, f"travel:{self.heading} discover→{act}"

    def _pick_fresh(self) -> str:
        now = float(self.tick)
        open_dirs = [d for d in _DIRS if self.blocked.get(d, -1) <= now]
        if not open_dirs:
            self.blocked.clear()
            open_dirs = list(_DIRS)
        scored: list[tuple[float, str]] = []
        for d in open_dirs:
            score = float(self.successes.get(d, 0)) * 0.4
            if d == _BACK.get(self.heading):
                score -= 5.0
            if d == self.heading:
                score += 1.0  # keep going if it was working
            if d in {"north", "east"}:
                score += 0.4
            scored.append((score + random.random() * 0.4, d))
        scored.sort(reverse=True)
        return scored[0][1]
