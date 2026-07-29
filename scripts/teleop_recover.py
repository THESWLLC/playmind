"""Focused teleop: close Options → click Ascension death buttons → verify alive."""

from __future__ import annotations

import time
from pathlib import Path

from playmind.actuators import OwnedGameKeyboardActuator, load_keymap
from playmind.ability_memory import AbilityMemory, ensure_ability_seeded
from playmind.capture import capture_config_from_dict
from playmind.learning import OnlinePolicy, owned_state_key
from playmind.owned_loop import load_owned_config, vision_obs_from_frame
from playmind.screen_llm import enrich_obs_from_screen
from playmind.ui_memory import UIMemory, ensure_seeded

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "playmind" / "owned"


def obs_now(path: Path, rois: dict) -> dict:
    o = vision_obs_from_frame(path, rois)
    o = enrich_obs_from_screen(path, o)
    if o.get("is_dead"):
        o["has_target"] = False
    return o


def show(tag: str, o: dict) -> None:
    print(
        f"{tag}: dead={o.get('is_dead')} ghost={o.get('is_ghost')} "
        f"modal={o.get('modal_menu')} hp={o.get('vision_player_hp')} "
        f"ocr={(o.get('screen_ocr') or '')[:120]!r}"
    )


def main() -> None:
    owned = load_owned_config(ROOT / "config" / "owned_game.json")
    rois = owned.get("rois", {})
    keymap = load_keymap(Path(owned.get("keymap_path", "config/keymap.example.json")))
    ui = UIMemory(DATA / "ui_memory.json")
    ensure_seeded(ui)
    # Seed Ascension death buttons (top-center)
    for label, fx, fy in (
        ("Return to Graveyard", 0.42, 0.09),
        ("return to graveyard", 0.42, 0.09),
        ("Resurrect in a Safe Zone", 0.58, 0.09),
        ("resurrect in a safe zone", 0.58, 0.09),
        ("Close", 0.62, 0.18),
    ):
        ui.remember(label, fx, fy, source="teleop_seed", success=True)

    abil = AbilityMemory(DATA / "ability_memory.json")
    ensure_ability_seeded(abil)
    act = OwnedGameKeyboardActuator(
        keymap=keymap,
        enabled=True,
        i_own_this_game=True,
        window_title_substr="Ascension",
        ui_memory=ui,
        ability_memory=abil,
        log_path=ROOT / "data" / "playmind" / "teleop_recover.jsonl",
    )
    policy = OnlinePolicy(epsilon=0.0, key_fn=owned_state_key)
    pp = DATA / "policy.json"
    if pp.exists():
        policy.load(pp)

    frame = DATA / "teleop_latest.png"
    cap = owned.get("capture", {})

    def snap():
        capture_config_from_dict(cap, frame)

    snap()
    o = obs_now(frame, rois)
    show("START", o)

    steps = []
    blob = (o.get("screen_ocr") or "").lower()
    if o.get("modal_menu") or "options" in blob or "exit game" in blob or "key bindings" in blob:
        steps.append(("key:esc", "close Options"))
    # Click both death buttons (one should work)
    steps.extend(
        [
            ("click:0.42,0.09", "Return to Graveyard frac"),
            ("click_label:Return to Graveyard", "Return label"),
            ("click:0.58,0.09", "Resurrect Safe Zone frac"),
            ("click_label:Resurrect in a Safe Zone", "Resurrect label"),
            ("release_spirit", "multi-point fallback"),
        ]
    )

    for action, why in steps:
        print(f"\n>>> {action} ({why})")
        before = o
        act.send(action)
        time.sleep(1.0)
        snap()
        o = obs_now(frame, rois)
        show(" after", o)
        if before.get("is_dead") and not o.get("is_dead"):
            print("*** LEFT DEATH ***")
            policy.teach(before, action, boost=2.0)
            break
        if before.get("modal_menu") and not o.get("modal_menu"):
            print("*** CLOSED MODAL ***")
            policy.teach(before, action, boost=1.5)
        # Still dead but Options closed — keep clicking
        if not o.get("is_dead") and (o.get("vision_player_hp") or 0) > 0.1:
            print("*** ALIVE ***")
            policy.teach(before, action, boost=2.0)
            break

    # If alive/ghost without death dialog, try a short farm teach
    snap()
    o = obs_now(frame, rois)
    show("MID", o)
    if not o.get("is_dead"):
        for action, why in (
            ("hold:w:1.0", "walk"),
            ("key:tab", "tab"),
            ("key:1", "atk"),
            ("key:1", "atk"),
            ("key:1", "atk"),
        ):
            print(f"\n>>> {action} ({why})")
            before = o
            act.send(action)
            time.sleep(0.7)
            snap()
            o = obs_now(frame, rois)
            show(" after", o)
            if (not before.get("has_target")) and o.get("has_target"):
                policy.teach(before, action, boost=1.5)
                print("*** TARGET ***")
            if before.get("has_target") and action.startswith("key:1"):
                policy.teach(before, action, boost=1.2)
                print("*** ATTACK TEACH ***")

    policy.save(pp)
    show("FINAL", o)
    print("done")


if __name__ == "__main__":
    main()
