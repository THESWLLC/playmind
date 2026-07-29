"""Directive → structured goal for owned-game play.

Examples:
  farm
  farm to level 2
  kill grell
  kill 5 young nightsaber
  go north / move east
  quest / turn in
  stop
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Goal:
    raw: str
    kind: str = "farm"  # farm | kill | go | quest | stop | free
    target_name: str | None = None
    target_count: int | None = None
    target_level: int | None = None
    direction: str | None = None  # north/south/east/west
    kills: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.kind == "stop":
            return "STOP"
        if self.kind == "kill":
            need = f" ({self.kills}/{self.target_count})" if self.target_count else ""
            who = self.target_name or "enemies"
            return f"KILL {who}{need}"
        if self.kind == "farm" and self.target_level:
            return f"FARM to level {self.target_level}"
        if self.kind == "farm":
            return "FARM / grind XP"
        if self.kind == "go" and self.direction:
            return f"GO {self.direction}"
        if self.kind == "quest":
            return "QUEST / turn-in / talk"
        return f"FREE: {self.raw}"

    def prompt_rules(self) -> str:
        """Strong instructions injected into the vision LLM."""
        lines = [
            f"PLAYER DIRECTIVE (obey this): {self.raw}",
            f"Active goal: {self.summary()}",
            "This is an MMO (WoW-like / Ascension): tab-target, action bar 1-5, Esc closes menus,",
            "red nameplates are enemies, quests/NPCs use Interact, death needs Resurrect/Release Spirit.",
        ]
        if self.kind == "stop":
            lines.append("Goal=STOP → reply wait")
        elif self.kind == "kill":
            who = self.target_name or "any hostile"
            lines.append(f"Goal=KILL → find and kill {who}: key:tab to target, then key:1/attack.")
            lines.append("Red ground circle / selected nameplate = already targeted → spam key:1.")
            lines.append("Never click_label on enemy names (boar/grell) — use Tab + ability keys.")
            lines.append("Do not open menus. Prefer combat over sightseeing.")
        elif self.kind == "farm":
            lines.append(
                "Goal=FARM XP → kill nearby hostiles, keep moving if no target, "
                "use abilities 1-5, avoid standing still."
            )
            lines.append(
                "If a mob has a red selection ring, you already have a target: press key:1 "
                "(do not click the nameplate text)."
            )
            if self.target_level:
                lines.append(f"Keep grinding until character level reaches {self.target_level}.")
        elif self.kind == "go" and self.direction:
            lines.append(
                f"Goal=TRAVEL → hold/move {self.direction} (hold:w/a/s/d). "
                "Only fight if something attacks you."
            )
        elif self.kind == "quest":
            lines.append(
                "Goal=QUEST → look for quest UI / NPCs: interact, click_label Accept/Continue/Complete."
            )
        return "\n".join(lines)


def parse_directive(text: str | None) -> Goal:
    raw = (text or "").strip()
    if not raw:
        return Goal(raw="farm", kind="farm", notes=["default"])

    low = raw.lower().strip()
    if low in {"stop", "wait", "halt"}:
        return Goal(raw=raw, kind="stop")

    # farm to level N / level to N / grind to 2
    m = re.search(r"(?:farm|grind|level)\s*(?:to\s*)?(?:level\s*)?(\d{1,2})", low)
    if m or low in {"farm", "grind", "xp"}:
        lvl = int(m.group(1)) if m else None
        return Goal(raw=raw, kind="farm", target_level=lvl)

    # kill 5 grell / kill grells / attack young nightsaber
    m = re.search(
        r"(?:kill|slay|attack|farm)\s+(?:(\d+)\s+)?(.+)$",
        low,
    )
    if m and not low.startswith("farm to"):
        count = int(m.group(1)) if m.group(1) else None
        name = (m.group(2) or "").strip()
        name = re.sub(r"\s+", " ", name)
        # "farm mobs" → farm kind; "kill grell" → kill
        if low.startswith("farm ") and name in {"mobs", "enemies", "xp"}:
            return Goal(raw=raw, kind="farm")
        if name and name not in {"mobs", "enemies"}:
            return Goal(raw=raw, kind="kill", target_name=name, target_count=count)

    # go north / walk east / move south
    m = re.search(r"(?:go|walk|run|move|travel)\s+(north|south|east|west)\b", low)
    if m:
        return Goal(raw=raw, kind="go", direction=m.group(1))
    if low in {"north", "south", "east", "west"}:
        return Goal(raw=raw, kind="go", direction=low)

    if any(k in low for k in ("quest", "turn in", "turnin", "npc", "talk")):
        return Goal(raw=raw, kind="quest")

    # Free-form — still pass through to the LLM strongly
    return Goal(raw=raw, kind="free")


def directive_reward_bonus(
    goal: Goal,
    prev: dict[str, Any],
    action: str,
    nxt: dict[str, Any],
) -> float:
    """Extra reward so Q-learning also pursues the directive."""
    if goal.kind == "stop":
        return 0.3 if action == "wait" else -0.2

    a = (action or "").lower()
    bonus = 0.0
    ocr = f"{nxt.get('screen_ocr') or ''} {prev.get('screen_ocr') or ''}".lower()
    target_name = (goal.target_name or "").lower()

    if goal.kind in {"farm", "kill"}:
        if nxt.get("has_target") and not prev.get("has_target"):
            bonus += 0.25
            if target_name and target_name in ocr:
                bonus += 0.35
        if a in {"attack", "key:1", "key:2", "key:3"} or a.startswith("ability:"):
            if nxt.get("has_target") or prev.get("has_target"):
                bonus += 0.2
                if target_name and target_name in ocr:
                    bonus += 0.25
            else:
                bonus -= 0.05
        if a in {"target_nearest", "key:tab"}:
            bonus += 0.1
        if a.startswith("move_") or a.startswith("hold:w"):
            if not prev.get("has_target"):
                bonus += 0.05  # hunting

    if goal.kind == "go" and goal.direction:
        want = {
            "north": ("move_north", "hold:w"),
            "south": ("move_south", "hold:s"),
            "east": ("move_east", "hold:d"),
            "west": ("move_west", "hold:a"),
        }.get(goal.direction, ())
        if any(a.startswith(w) or a == w for w in want):
            bonus += 0.35
        elif a.startswith("move_") or a.startswith("hold:"):
            bonus -= 0.1

    if goal.kind == "quest":
        if a == "interact" or a.startswith("click_label:"):
            bonus += 0.25
        if any(k in ocr for k in ("accept", "complete", "continue", "quest")):
            if a.startswith("click_label:") or a == "interact":
                bonus += 0.3

    return bonus


def goal_action_hint(goal: Goal, obs: dict[str, Any], tick: int) -> str | None:
    """Cheap heuristic action aligned with the goal (used between VLM ticks)."""
    if goal.kind == "stop":
        return "wait"
    if obs.get("is_dead") or (obs.get("vision_player_hp") is not None and float(obs["vision_player_hp"]) < 0.05):
        return None  # let other systems handle
    if obs.get("modal_menu"):
        return "key:esc"

    if goal.kind == "go" and goal.direction:
        return {
            "north": "hold:w:0.8",
            "south": "hold:s:0.8",
            "east": "hold:d:0.8",
            "west": "hold:a:0.8",
        }.get(goal.direction)

    if goal.kind in {"farm", "kill"}:
        # Commit to killing once anything looks targeted — do not click nameplates.
        if obs.get("has_target") or obs.get("in_combat"):
            # Don't stand still forever — mix in a short step every few casts.
            if tick % 7 == 0:
                return "hold:w:0.6"
            return "key:1" if tick % 3 else "attack"
        phase = tick % 8
        if phase in {0, 4}:
            return "key:tab"
        if phase in {1, 2, 3, 5, 6}:
            return "hold:w:1.0"
        return "move_east"

    if goal.kind == "quest":
        phase = tick % 4
        if phase == 0:
            return "interact"
        if phase == 1:
            return "click_label:Accept"
        return "hold:w:0.5"

    return None
