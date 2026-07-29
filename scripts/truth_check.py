"""Cross-check claimed brain status vs what's actually on the latest frame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from playmind.life_fsm import classify_life_raw
from playmind.screen_llm import enrich_obs_from_screen
from playmind.vision import detect_death_dialog, detect_target_bar


def truth_from_frame(path: Path) -> dict[str, Any]:
    obs = enrich_obs_from_screen(path, {}, do_ocr=False)
    death = enrich_obs_from_screen(path, dict(obs), do_ocr=True, ocr_mode="death")
    alive = enrich_obs_from_screen(path, dict(obs), do_ocr=True, ocr_mode="alive")
    ocr = f"{death.get('screen_ocr') or ''} | {alive.get('screen_ocr') or ''}"
    ht, thp = detect_target_bar(path)
    dead_px, ghost_px = detect_death_dialog(path)
    merged = {
        **death,
        "screen_ocr": ocr,
        "has_target": ht,
        "is_dead": dead_px or death.get("is_dead"),
        "is_ghost": ghost_px or death.get("is_ghost"),
        "vision_player_hp": death.get("vision_player_hp") or obs.get("vision_player_hp"),
    }
    phase = classify_life_raw(merged)
    return {
        "desaturated": bool(obs.get("desaturated")),
        "sat_frac": obs.get("sat_frac"),
        "ghost_buttons": bool(obs.get("ghost_buttons")),
        "target": bool(ht),
        "target_hp": thp,
        "death_px": bool(dead_px),
        "ghost_px": bool(ghost_px),
        "life_raw": phase,
        "ocr": (ocr or "")[:220],
        "body_should_be": (
            "dead"
            if phase in {"dead_dialog", "confirm", "rez_picker"}
            else ("ghost" if phase == "ghost" else "alive")
        ),
    }


def compare_claim(claim: dict[str, Any], truth: dict[str, Any]) -> list[str]:
    lies: list[str] = []
    claimed_alive = (claim.get("life_phase") or "alive") == "alive" and not claim.get("is_dead")
    truth_dead = truth["body_should_be"] in {"dead", "ghost"}
    if claimed_alive and truth_dead:
        lies.append(
            f"LIE: claim ALIVE but frame is {truth['body_should_be']} "
            f"(ocr={truth['ocr'][:80]!r} desat={truth['desaturated']})"
        )
    if claim.get("is_dead") and truth["body_should_be"] == "alive" and not truth["desaturated"]:
        lies.append("LIE: claim DEAD but frame looks colored/alive")
    if bool(claim.get("has_target")) and not truth["target"] and truth["body_should_be"] == "alive":
        lies.append("WARN: claim has_target but no red-ring/target frame found")
    if not bool(claim.get("has_target")) and truth["target"]:
        lies.append("WARN: frame has target ring but claim has_target=False")
    return lies


def main() -> None:
    path = Path("data/playmind/owned/latest.png")
    claim_path = Path("data/playmind/owned/last_claim.json")
    truth = truth_from_frame(path) if path.exists() else {"error": "no frame"}
    claim: dict[str, Any] = {}
    if claim_path.exists():
        claim = json.loads(claim_path.read_text(encoding="utf-8-sig"))
    lies = compare_claim(claim, truth) if claim and "error" not in truth else []
    print(json.dumps({"truth": truth, "lies": lies, "claim_keys": list(claim.keys())[:12]}, indent=2))


if __name__ == "__main__":
    main()
