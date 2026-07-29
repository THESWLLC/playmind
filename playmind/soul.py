"""Soul: first-person self-model for the owned-game agent.

Not a cheat script — a coherent awareness layer the learner and VLM share:
who I am (alive / dead / ghost), what I can cast, whether I have a target,
and what I should care about right now.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SoulState:
    """What it feels like to be this character right now."""

    body: str = "alive"  # alive | dead | ghost
    hp: float = 1.0
    has_target: bool = False
    hostiles_near: bool = False
    modal: bool = False
    confirm_pending: bool = False  # "are you sure?" style
    spells: list[str] = field(default_factory=list)
    bar_slots: list[bool] = field(default_factory=list)  # True = icon present
    feeling: str = ""
    intent: str = ""

    def summary(self) -> str:
        lines = [
            f"I am {self.body.upper()}.",
            f"My health is about {int(self.hp * 100)}%.",
        ]
        if self.body == "dead":
            lines.append("I see death UI — I must return / resurrect before I can fight.")
            if self.confirm_pending:
                lines.append("A confirm dialog is open — I must answer Yes/Accept.")
            if self.modal:
                lines.append("A menu is blocking me — close it first.")
        elif self.body == "ghost":
            lines.append("I am a spirit — run to my corpse or a spirit healer.")
        else:
            if self.has_target:
                lines.append("I have a target locked — fight with my spells.")
            elif self.hostiles_near:
                lines.append("Hostiles are nearby — Tab to target, then cast.")
            else:
                lines.append("No target — walk and look for enemies.")
            if self.spells:
                lines.append("My known spells: " + ", ".join(self.spells[:8]) + ".")
            filled = sum(1 for s in self.bar_slots if s)
            if self.bar_slots:
                lines.append(f"Action bar shows ~{filled} filled slots.")
        if self.feeling:
            lines.append(self.feeling)
        if self.intent:
            lines.append(f"Right now I intend to: {self.intent}")
        return " ".join(lines)

    def to_obs(self) -> dict[str, Any]:
        return {
            "soul_body": self.body,
            "soul_feeling": self.feeling,
            "soul_intent": self.intent,
            "soul_summary": self.summary(),
            "soul_spells": list(self.spells),
            "bar_slots_filled": sum(1 for s in self.bar_slots if s),
            "confirm_pending": self.confirm_pending,
        }


def read_action_bar_slots(path: Path, n: int = 10) -> list[bool]:
    """Heuristic: bottom-center action bar slots with non-empty icons."""
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return []
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return []
    w, h = img.size
    # Ascension main bar sits bottom-center above micro buttons.
    y0, y1 = int(h * 0.88), int(h * 0.94)
    # Center cluster of slots
    x_mid = w * 0.50
    slot_w = int(w * 0.028)
    gap = int(w * 0.004)
    start = int(x_mid - (n / 2) * (slot_w + gap))
    filled: list[bool] = []
    for i in range(n):
        x0 = start + i * (slot_w + gap)
        x1 = x0 + slot_w
        if x0 < 0 or x1 > w:
            filled.append(False)
            continue
        crop = img.crop((x0, y0, x1, y1))
        px = list(crop.getdata())
        if not px:
            filled.append(False)
            continue
        # Empty slots are dark/grey; icons are colorful or bright.
        vivid = 0
        for r, g, b in px:
            mx, mn = max(r, g, b), min(r, g, b)
            if mx > 70 and (mx - mn) > 25:
                vivid += 1
            elif mx > 140:
                vivid += 1
        filled.append((vivid / len(px)) > 0.12)
    return filled


def feel_soul(obs: dict[str, Any], frame_path: Path | None = None) -> SoulState:
    """Build soul state from sensors + optional action-bar glance."""
    hp = float(obs.get("vision_player_hp") or obs.get("player", {}).get("hp") or 0.5)
    ocr = (obs.get("screen_ocr") or "").lower()
    phase = obs.get("life_phase")
    # Corpse-run ghost wins over death-dialog heuristics.
    if phase in {"dead_dialog", "confirm", "rez_picker"}:
        dead, ghost = True, False
    elif phase == "ghost" or re.search(r"\b\d+\s*yds?\b", ocr) or "spirit healer" in ocr:
        dead, ghost = False, True
    elif phase == "alive":
        dead, ghost = False, False
    else:
        if re.search(r"\b\d+\s*yds?\b", ocr) or "spirit healer" in ocr:
            dead, ghost = False, True
        else:
            dead = bool(obs.get("is_dead")) or "you are dead" in ocr
            ghost = bool(obs.get("is_ghost")) and not dead
    if dead:
        body = "dead"
        hp = 0.0
    elif ghost:
        body = "ghost"
    else:
        body = "alive"

    confirm = "are you sure" in ocr or "want to return" in ocr
    modal = bool(obs.get("modal_menu"))

    spells: list[str] = []
    for name in obs.get("abilities_known") or []:
        spells.append(str(name))
    # Prefer named combat abilities over generic key seeds
    summary = obs.get("ability_summary") or ""
    if summary:
        for part in summary.split(","):
            part = part.strip()
            if "=" in part:
                n = part.split("=", 1)[0].strip()
                if n and n not in spells:
                    spells.append(n)

    bar: list[bool] = []
    if frame_path is not None and frame_path.exists():
        bar = read_action_bar_slots(frame_path)

    feeling = ""
    intent = ""
    if body == "dead":
        feeling = "Cold. The world is grey. I cannot fight."
        if modal:
            intent = "close the Options menu"
        elif confirm:
            intent = "confirm — Yes / Accept"
        else:
            intent = "Return to Graveyard or Resurrect in a Safe Zone"
    elif body == "ghost":
        feeling = "I am only a spirit."
        intent = "find my corpse or spirit healer"
    elif hp < 0.25:
        feeling = "I am nearly dead — this fight is killing me."
        intent = "move away or finish carefully"
    elif obs.get("has_target"):
        feeling = "Locked on. Ready to cast."
        intent = "press my damage spells (1 / 2)"
    elif obs.get("hostiles_near"):
        feeling = "I smell enemies."
        intent = "Tab target, then cast"
    else:
        feeling = "The clearing is quiet."
        intent = "walk and hunt"

    # Nudge spell literacy when bar looks full but we only know generic keys.
    named = [s for s in spells if s not in {"attack", "primary", "secondary", "target nearest", "loot", "interact"}
             and not s.startswith("key ") and not s.startswith("ability ")]
    if body == "alive" and sum(1 for b in bar if b) >= 2 and len(named) < 1:
        intent = "look at action bar icons and bind:SpellName=1 (and =2)"
        feeling = (feeling + " My spells feel unnamed — I should learn them.").strip()

    return SoulState(
        body=body,
        hp=hp,
        has_target=bool(obs.get("has_target")),
        hostiles_near=bool(obs.get("hostiles_near")),
        modal=modal,
        confirm_pending=confirm,
        spells=spells[:12],
        bar_slots=bar,
        feeling=feeling,
        intent=intent,
    )
