"""Every-N-ticks coach: diagnose, suggest, and apply safe learning upgrades.

Run: python scripts/tick_coach.py
Writes data/playmind/owned/suggestions.jsonl and may patch process_memory.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playmind.process_memory import ProcessMemory


def _load_truth_mod():
    path = ROOT / "scripts" / "truth_check.py"
    spec = importlib.util.spec_from_file_location("truth_check", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_truth = _load_truth_mod()
compare_claim = _truth.compare_claim
truth_from_frame = _truth.truth_from_frame

DATA = ROOT / "data" / "playmind" / "owned"
CLAIM = DATA / "last_claim.json"
FRAME = DATA / "latest.png"
SUGGESTIONS = DATA / "suggestions.jsonl"
STATUS_SNAP = DATA / "last_coach.json"


def _load_claim() -> dict[str, Any]:
    if not CLAIM.exists():
        return {}
    try:
        return json.loads(CLAIM.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def diagnose(claim: dict[str, Any], truth: dict[str, Any]) -> list[dict[str, Any]]:
    """Return actionable suggestions toward farm + learn goals."""
    out: list[dict[str, Any]] = []
    phase = str(claim.get("life_phase") or truth.get("life_raw") or "alive")
    tgt = bool(claim.get("has_target"))
    action = str(claim.get("action") or "")
    stagnant = int(claim.get("stagnant") or 0)
    no_dmg = int(claim.get("no_damage_casts") or 0)
    stage = str(claim.get("progress_stage") or "")
    decision = str(claim.get("decision") or "")

    lies = compare_claim(claim, truth) if claim and "error" not in truth else []
    for lie in lies:
        out.append(
            {
                "id": "sensor_lie",
                "priority": 1,
                "suggestion": lie,
                "implement": "fix_sensors",
            }
        )

    # Idle combat / same-spot freeze
    if phase == "alive" and (stagnant >= 5 or no_dmg >= 4):
        if truth.get("target") is False or "no target" in (truth.get("ocr") or "").lower():
            out.append(
                {
                    "id": "false_target_idle",
                    "priority": 1,
                    "suggestion": "Casting in place with no real target — escalate leave_bubble + Tab.",
                    "implement": "force_progress_break",
                }
            )
        elif stagnant >= 6:
            out.append(
                {
                    "id": "stagnant_farm",
                    "priority": 2,
                    "suggestion": "Stuck in same spot — reinforce travel heading and punish idle casts.",
                    "implement": "boost_travel_memory",
                }
            )

    if phase in {"dead_dialog", "confirm", "rez_picker"}:
        out.append(
            {
                "id": "death_pipeline",
                "priority": 1,
                "suggestion": f"In {phase}: prefer remembered successful clicks for this phase.",
                "implement": "seed_death_pipeline",
                "phase": phase,
            }
        )

    if truth.get("body_should_be") == "dead" and phase == "alive":
        out.append(
            {
                "id": "alive_lie",
                "priority": 0,
                "suggestion": "Claim ALIVE but frame is dead — death sensor broken.",
                "implement": "fix_sensors",
            }
        )

    if stage in {"push", "break_loop"}:
        out.append(
            {
                "id": "curriculum_push",
                "priority": 2,
                "suggestion": f"Progress stage={stage}: keep forced travel until motion recovers.",
                "implement": "force_progress_break",
            }
        )

    if "travel:engage" in decision and no_dmg >= 2:
        out.append(
            {
                "id": "engage_trap",
                "priority": 1,
                "suggestion": "travel:engage is mashing 1 with no damage — abandon target.",
                "implement": "force_progress_break",
            }
        )

    # Always reinforce persistence goals
    out.append(
        {
            "id": "persist_processes",
            "priority": 3,
            "suggestion": "Snapshot travel + death pipeline so next death/restart reuses what worked.",
            "implement": "persist_snapshot",
        }
    )
    return sorted(out, key=lambda x: x.get("priority", 9))


def apply_suggestions(suggestions: list[dict[str, Any]], mem: ProcessMemory) -> list[str]:
    """Apply safe, idempotent upgrades to process memory."""
    applied: list[str] = []
    ids = {s["id"] for s in suggestions}
    implements = {s.get("implement") for s in suggestions}

    if (
        "force_progress_break" in ids
        or "false_target_idle" in ids
        or "engage_trap" in ids
        or "force_progress_break" in implements
    ):
        for p in ("force_travel_on_stagnation", "veto_no_damage_target", "flee_below_40"):
            if p not in mem.preventions:
                mem.preventions.append(p)
                applied.append(f"add_prevention:{p}")

    if "seed_death_pipeline" in ids or "seed_death_pipeline" in implements:
        # Seed from ui_memory successes if pipeline thin
        ui_path = DATA / "ui_memory.json"
        if ui_path.exists():
            try:
                ui = json.loads(ui_path.read_text(encoding="utf-8-sig"))
                labels = ui.get("labels") or {}
            except (OSError, json.JSONDecodeError):
                labels = {}
            mapping = {
                "yes": "confirm",
                "return to graveyard": "dead_dialog",
                "closest town": "rez_picker",
            }
            for label, phase in mapping.items():
                row = labels.get(label) or {}
                if int(row.get("successes", 0)) > 0:
                    mem.note_pipeline(phase, f"click_label:{label}", success=True)
                    applied.append(f"seed:{phase}:{label}")

    if "boost_travel_memory" in ids or "boost_travel_memory" in implements:
        heading = (mem.travel or {}).get("heading") or "east"
        succ = dict((mem.travel or {}).get("successes") or {})
        succ[heading] = int(succ.get(heading, 0)) + 2
        # Prefer turning away from current if stuck
        mem.travel = {
            **(mem.travel or {}),
            "heading": {"east": "north", "north": "west", "west": "south", "south": "east"}.get(
                heading, "north"
            ),
            "successes": succ,
            "still_farm": 0,
        }
        applied.append(f"travel_rehead:{mem.travel['heading']}")

    if applied or "persist_processes" in ids or "persist_snapshot" in implements:
        mem.save()
        applied.append("saved_process_memory")

    return applied


def main() -> None:
    claim = _load_claim()
    truth = truth_from_frame(FRAME) if FRAME.exists() else {"error": "no frame"}
    mem = ProcessMemory(DATA / "process_memory.json")
    mem.load()

    suggestions = diagnose(claim, truth)
    applied = apply_suggestions(suggestions, mem)

    report = {
        "t": time.time(),
        "tick": claim.get("tick"),
        "phase": claim.get("life_phase"),
        "progress_stage": claim.get("progress_stage"),
        "stagnant": claim.get("stagnant"),
        "no_damage_casts": claim.get("no_damage_casts"),
        "truth_body": truth.get("body_should_be"),
        "lies": compare_claim(claim, truth) if claim and "error" not in truth else [],
        "suggestions": suggestions,
        "applied": applied,
        "memory": mem.summary(),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    with SUGGESTIONS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report) + "\n")
    STATUS_SNAP.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
