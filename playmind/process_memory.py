"""Persistent process memory: death pipelines, death causes, preventions, travel.

Once the agent clears a death dialog or learns why it died, that knowledge
survives restarts and guides the next attempt.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _now() -> float:
    return time.time()


@dataclass
class ProcessMemory:
    """Long-term memory of procedures and failure → prevention."""

    path: Path = Path("data/playmind/owned/process_memory.json")
    # phase -> action -> {successes, fails, last}
    death_pipeline: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    # cause_id -> {count, last_hp, prevention, detail}
    death_causes: dict[str, dict[str, Any]] = field(default_factory=dict)
    # active prevention rule ids
    preventions: list[str] = field(default_factory=list)
    # travel snapshot
    travel: dict[str, Any] = field(default_factory=dict)
    # named recipes that worked end-to-end
    recipes: dict[str, list[str]] = field(default_factory=dict)
    updated_at: float = 0.0

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return
        self.death_pipeline = raw.get("death_pipeline") or {}
        self.death_causes = raw.get("death_causes") or {}
        self.preventions = list(raw.get("preventions") or [])
        self.travel = raw.get("travel") or {}
        self.recipes = raw.get("recipes") or {}
        self.updated_at = float(raw.get("updated_at") or 0)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now()
        payload = {
            "updated_at": self.updated_at,
            "death_pipeline": self.death_pipeline,
            "death_causes": self.death_causes,
            "preventions": self.preventions,
            "travel": self.travel,
            "recipes": self.recipes,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- death pipeline (what to click next time) ---

    def note_pipeline(self, phase: str, action: str, *, success: bool) -> None:
        if not phase or not action:
            return
        bucket = self.death_pipeline.setdefault(phase, {})
        row = bucket.setdefault(action, {"successes": 0, "fails": 0, "last": 0.0})
        if success:
            row["successes"] = int(row.get("successes", 0)) + 1
        else:
            row["fails"] = int(row.get("fails", 0)) + 1
        row["last"] = _now()
        # Remember full rez recipe when we finish to alive/ghost.
        if success and phase in {"rez_picker", "confirm", "dead_dialog"}:
            recipe = self.recipes.setdefault("death_to_alive", [])
            if action not in recipe[-3:]:
                recipe.append(action)
                self.recipes["death_to_alive"] = recipe[-12:]

    def best_pipeline_action(self, phase: str, candidates: list[str] | None = None) -> str | None:
        bucket = self.death_pipeline.get(phase) or {}
        if not bucket:
            return None
        scored: list[tuple[float, str]] = []
        for act, row in bucket.items():
            if candidates and act not in candidates and not any(
                act.lower() == c.lower() for c in candidates
            ):
                # still allow semantic relatives
                if not any(c.lower() in act.lower() or act.lower() in c.lower() for c in (candidates or [])):
                    continue
            succ = int(row.get("successes", 0))
            fail = int(row.get("fails", 0))
            if succ <= 0 and fail > 2:
                continue
            score = succ * 2.0 - fail * 0.5
            scored.append((score, act))
        if not scored:
            # fall back to any successful action in phase
            for act, row in bucket.items():
                if int(row.get("successes", 0)) > 0:
                    scored.append((float(row["successes"]), act))
        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][1]

    # --- death causes → prevention ---

    def note_death_cause(self, prev: dict[str, Any], action: str) -> str:
        """Infer why we died from the last alive observation; install prevention."""
        hp = float(prev.get("vision_player_hp") or prev.get("player", {}).get("hp") or 0.5)
        had_tgt = bool(prev.get("has_target"))
        stagnant = int(prev.get("stagnant") or 0)
        no_dmg = int(prev.get("no_damage_casts") or 0)
        motion = float(prev.get("motion") or 0)
        a = (action or "").lower()

        if hp < 0.35 and had_tgt and (a.startswith("key:") or a == "attack"):
            cause = "fought_too_long_low_hp"
            prevention = "flee_below_40"
            detail = "Kept casting while HP < 35% with a target."
        elif hp < 0.45 and (stagnant >= 4 or no_dmg >= 3 or motion < 2.0):
            cause = "stood_still_while_dying"
            prevention = "force_travel_on_stagnation"
            detail = "Low HP + idle/false-target casting; no motion."
        elif hp < 0.25:
            cause = "ignored_critical_hp"
            prevention = "flee_below_40"
            detail = "HP crashed below 25% without fleeing."
        elif had_tgt and no_dmg >= 5:
            cause = "false_target_tunnel"
            prevention = "veto_no_damage_target"
            detail = "Spam-cast sticky false target until death."
        else:
            cause = "unknown_death"
            prevention = "flee_below_40"
            detail = f"Died after {action!r} at hp≈{hp:.2f}."

        row = self.death_causes.setdefault(
            cause,
            {"count": 0, "prevention": prevention, "detail": detail, "last_hp": hp},
        )
        row["count"] = int(row.get("count", 0)) + 1
        row["last_hp"] = hp
        row["last_action"] = action
        row["last"] = _now()
        row["prevention"] = prevention
        row["detail"] = detail
        if prevention not in self.preventions:
            self.preventions.append(prevention)
        self.save()
        return cause

    def prevention_action(self, obs: dict[str, Any]) -> tuple[str, str] | None:
        """While alive, apply learned preventions before farming to death again."""
        if obs.get("is_dead") or obs.get("is_ghost"):
            return None
        if obs.get("life_phase") and obs.get("life_phase") != "alive":
            return None
        hp = float(obs.get("vision_player_hp") or obs.get("player", {}).get("hp") or 1.0)
        stagnant = int(obs.get("stagnant") or 0)
        no_dmg = int(obs.get("no_damage_casts") or 0)

        if "flee_below_40" in self.preventions and hp < 0.40 and hp > 0.05:
            return "hold:s:1.2", "memory:prevention flee_below_40"
        if "force_travel_on_stagnation" in self.preventions and (
            stagnant >= 4 or no_dmg >= 3
        ):
            return "hold:w:1.4", "memory:prevention force_travel_on_stagnation"
        if "veto_no_damage_target" in self.preventions and no_dmg >= 3:
            return "key:tab", "memory:prevention veto_no_damage_target"
        return None

    def apply_travel_snapshot(self, travel: Any) -> None:
        self.travel = {
            "heading": getattr(travel, "heading", "east"),
            "successes": dict(getattr(travel, "successes", {}) or {}),
            "blocked": dict(getattr(travel, "blocked", {}) or {}),
            "still_farm": int(getattr(travel, "still_farm", 0) or 0),
        }

    def restore_travel(self, travel: Any) -> None:
        if not self.travel:
            return
        if self.travel.get("heading"):
            travel.heading = str(self.travel["heading"])
        if isinstance(self.travel.get("successes"), dict):
            travel.successes = dict(self.travel["successes"])
        if isinstance(self.travel.get("blocked"), dict):
            travel.blocked = {k: float(v) for k, v in self.travel["blocked"].items()}

    def summary(self) -> str:
        causes = ", ".join(
            f"{k}×{v.get('count', 0)}" for k, v in sorted(self.death_causes.items())
        ) or "none"
        pipes = {ph: len(acts) for ph, acts in self.death_pipeline.items()}
        return f"causes=[{causes}] pipeline={pipes} preventions={self.preventions}"
