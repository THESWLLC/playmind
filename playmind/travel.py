"""Travel memory: discover a heading, commit forward, remember walls.

Stops east↔west thrash and “attack in place forever” when nothing is dying.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


_DIRS = ("north", "east", "south", "west")
_HOLD = {
    "north": "hold:w:1.1",
    "south": "hold:s:1.1",
    "east": "hold:d:1.1",
    "west": "hold:a:1.1",
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
    still_farm: int = 0  # combat spam with no progress
    blocked: dict[str, float] = field(default_factory=dict)  # heading -> until_tick
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

        if moved:
            if motion < 2.5:
                self.low_motion += 1
            else:
                self.low_motion = 0
                # Remember this heading worked.
                for d, hold in _HOLD.items():
                    if a.startswith(hold[:6]) or a == _MOVE[d]:
                        self.successes[d] = self.successes.get(d, 0) + 1
                        self.heading = d
                        break
            # Wall: several low-motion tries on this heading → mark blocked.
            if self.low_motion >= 2:
                self.blocked[self.heading] = float(tick + 40)
                self.heading = self._pick_fresh()
                self.commit_left = 0
                self.low_motion = 0

        if combat and had_target and reward < 0.7:
            # Swinging without a strong kill signal — count as idle farm.
            self.still_farm += 1
        elif moved and motion >= 2.5:
            self.still_farm = 0
        elif reward >= 0.9:
            self.still_farm = 0

    def needs_travel(self, obs: dict[str, Any]) -> bool:
        if obs.get("is_dead") or obs.get("is_ghost") or obs.get("modal_menu"):
            return False
        if obs.get("life_phase") and obs.get("life_phase") != "alive":
            return False
        # No target → go find denser ground.
        if not obs.get("has_target") and not obs.get("in_combat"):
            return True
        # Has a "target" but we're just spinning in place.
        if self.still_farm >= 4:
            return True
        if self.low_motion >= 2:
            return True
        return False

    def action(self, obs: dict[str, Any]) -> tuple[str, str]:
        """Return (action, reason) — commit to a heading for several ticks."""
        if self.commit_left > 0:
            self.commit_left -= 1
            act = _HOLD[self.heading]
            return act, f"travel:{self.heading} commit→{act}"

        self.heading = self._pick_fresh()
        # Longer commits so we actually leave the spawn bubble.
        self.commit_left = random.randint(3, 6)
        act = _HOLD[self.heading]
        return act, f"travel:{self.heading} discover→{act}"

    def _pick_fresh(self) -> str:
        now = float(self.tick)
        open_dirs = [d for d in _DIRS if self.blocked.get(d, -1) <= now]
        if not open_dirs:
            self.blocked.clear()
            open_dirs = list(_DIRS)
        # Prefer headings that previously produced motion; avoid immediate reverse.
        scored: list[tuple[float, str]] = []
        for d in open_dirs:
            score = float(self.successes.get(d, 0))
            if d == _BACK.get(self.heading):
                score -= 3.0
            if d == self.heading:
                score += 1.5
            scored.append((score + random.random() * 0.3, d))
        scored.sort(reverse=True)
        return scored[0][1]

    def turn_toward_hostiles(self) -> str | None:
        """Occasional glance left/right while traveling."""
        if random.random() < 0.18:
            self.heading = random.choice([_LEFT[self.heading], _RIGHT[self.heading]])
            return _HOLD[self.heading]
        return None
