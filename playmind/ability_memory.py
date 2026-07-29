"""Runtime ability invention: LLM can bind new named skills to keys live.

Persists to data/playmind/owned/ability_memory.json — no restart needed.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _norm(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    return re.sub(r"\s+", " ", s)


@dataclass
class AbilityMemory:
    path: Path
    abilities: dict[str, dict[str, Any]] = field(default_factory=dict)
    _mtime: float = 0.0

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            mtime = self.path.stat().st_mtime
            if mtime <= self._mtime and self.abilities:
                return
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.abilities = {str(k).lower(): v for k, v in raw.get("abilities", {}).items()}
            self._mtime = mtime
        except Exception:
            pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated_at": time.time(), "abilities": self.abilities}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._mtime = self.path.stat().st_mtime

    def bind(
        self,
        name: str,
        key: str,
        *,
        hold: float | None = None,
        source: str = "llm",
        success: bool | None = None,
    ) -> str:
        """Create or update a named ability → key mapping. Returns normalized name."""
        n = _norm(name)
        if not n or not key:
            return n
        key = key.strip().lower()
        row = self.abilities.get(n, {"key": key, "hits": 0, "successes": 0, "source": source})
        row["key"] = key
        if hold is not None:
            row["hold"] = float(hold)
        row["hits"] = int(row.get("hits", 0)) + 1
        row["source"] = source
        row["last_seen"] = time.time()
        if success is True:
            row["successes"] = int(row.get("successes", 0)) + 1
        self.abilities[n] = row
        self.save()
        return n

    def lookup(self, name: str) -> dict[str, Any] | None:
        self.load()
        n = _norm(name)
        if n in self.abilities:
            return self.abilities[n]
        for k, row in self.abilities.items():
            if n in k or k in n:
                return row
        return None

    def known(self) -> list[str]:
        self.load()
        return sorted(self.abilities.keys())

    def known_summary(self, limit: int = 24) -> str:
        self.load()
        parts = []
        for name in self.known()[:limit]:
            row = self.abilities[name]
            parts.append(f"{name}={row.get('key')}")
        return ", ".join(parts)


# Seed common WoW-like bar slots so invention starts from something sensible.
DEFAULT_ABILITY_SEEDS = {
    "attack": {"key": "1", "hits": 1, "successes": 0, "source": "seed"},
    "primary": {"key": "1", "hits": 1, "successes": 0, "source": "seed"},
    "secondary": {"key": "2", "hits": 1, "successes": 0, "source": "seed"},
    "ability 2": {"key": "2", "hits": 1, "successes": 0, "source": "seed"},
    "ability 3": {"key": "3", "hits": 1, "successes": 0, "source": "seed"},
    "ability 4": {"key": "4", "hits": 1, "successes": 0, "source": "seed"},
    "ability 5": {"key": "5", "hits": 1, "successes": 0, "source": "seed"},
    "loot": {"key": "f", "hits": 1, "successes": 0, "source": "seed"},
    "interact": {"key": "e", "hits": 1, "successes": 0, "source": "seed"},
    "target nearest": {"key": "tab", "hits": 1, "successes": 0, "source": "seed"},
}


def ensure_ability_seeded(memory: AbilityMemory) -> None:
    memory.load()
    changed = False
    for k, v in DEFAULT_ABILITY_SEEDS.items():
        if k not in memory.abilities:
            memory.abilities[k] = dict(v)
            changed = True
    if changed:
        memory.save()


def parse_dynamic_action(action: str) -> dict[str, Any] | None:
    """Parse invented actions: key / hold / ability / bind / click variants."""
    a = (action or "").strip()
    if not a:
        return None

    # bind:Fireball=2  OR  invent:Frostbolt=shift+2  OR  learn:Heal=key:3
    m = re.match(
        r"(?:bind|invent|learn)\s*[:=]?\s*([^=]+?)\s*=\s*(?:key\s*[:=]?\s*)?(.+)$",
        a,
        flags=re.I,
    )
    if m:
        return {
            "type": "bind",
            "name": m.group(1).strip().strip("\"'"),
            "key": m.group(2).strip().strip("\"'").lower(),
        }

    # ability:Fireball  OR  cast:Fireball
    m = re.match(r"(?:ability|cast|use)\s*[:=]?\s*(.+)$", a, flags=re.I)
    if m and not re.match(r"(?:ability|cast|use)\s*[:=]?\s*\d", a, flags=re.I):
        label = m.group(1).strip().strip("\"'")
        if label and "," not in label:
            return {"type": "ability", "name": label}

    # hold:w:1.5  OR  hold:shift+w:0.8
    m = re.match(
        r"hold\s*[:=]?\s*([a-z0-9+\-]+)\s*[:=,]?\s*([0-9]*\.?[0-9]+)\s*$",
        a,
        flags=re.I,
    )
    if m:
        return {"type": "hold", "key": m.group(1).lower(), "seconds": float(m.group(2))}

    # key:2  key:shift+2  press:f1
    m = re.match(r"(?:key|press|tap)\s*[:=]?\s*([a-z0-9+\-]+)\s*$", a, flags=re.I)
    if m:
        return {"type": "key", "key": m.group(1).lower()}

    # click:0.5,0.55
    low = a.lower()
    m = re.match(r"click\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*$", low)
    if m:
        return {"type": "click_frac", "fx": float(m.group(1)), "fy": float(m.group(2))}

    m = re.match(r"click_label\s*[:=]?\s*(.+)$", a, flags=re.I)
    if m:
        return {"type": "click_label", "label": m.group(1).strip().strip("\"'")}

    m = re.match(r"click\s+(.+)$", a, flags=re.I)
    if m and "," not in m.group(1):
        return {"type": "click_label", "label": m.group(1).strip().strip("\"'")}

    return None
