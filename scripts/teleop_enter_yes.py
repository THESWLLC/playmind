"""Press Enter / click Yes on Ascension graveyard confirm; verify leave death."""

from __future__ import annotations

import time
from pathlib import Path

from playmind.actuators import OwnedGameKeyboardActuator, load_keymap
from playmind.capture import capture_config_from_dict
from playmind.owned_loop import load_owned_config, vision_obs_from_frame
from playmind.screen_llm import enrich_obs_from_screen
from playmind.ui_memory import UIMemory

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    owned = load_owned_config(ROOT / "config" / "owned_game.json")
    ui = UIMemory(ROOT / "data" / "playmind" / "owned" / "ui_memory.json")
    act = OwnedGameKeyboardActuator(
        keymap=load_keymap(Path(owned.get("keymap_path", "config/keymap.example.json"))),
        enabled=True,
        i_own_this_game=True,
        window_title_substr="Ascension",
        ui_memory=ui,
        log_path=ROOT / "data" / "playmind" / "enter_yes.jsonl",
    )
    frame = ROOT / "data" / "playmind" / "owned" / "latest.png"
    cap = owned.get("capture", {})

    def snap():
        capture_config_from_dict(cap, frame)
        return enrich_obs_from_screen(
            frame, vision_obs_from_frame(frame, owned.get("rois", {}))
        )

    o = snap()
    print("START dead=", o.get("is_dead"), (o.get("screen_ocr") or "")[:140])
    blob = (o.get("screen_ocr") or "").lower()
    if "intellect" in blob or "pyromancer" in blob:
        act.send("key:c")
        time.sleep(0.35)
    if "are you sure" not in blob:
        act.send("click:0.46,0.10")
        time.sleep(0.8)
        o = snap()
        print("OPEN", (o.get("screen_ocr") or "")[:120])

    for action in ("key:enter", "key:enter", "click:0.47,0.205", "key:space"):
        print("TRY", action)
        act.send(action)
        time.sleep(1.0)
        o = snap()
        blob = (o.get("screen_ocr") or "").lower()
        print(
            "  dead=",
            o.get("is_dead"),
            "sure=",
            "are you sure" in blob,
            "ocr=",
            blob[:100],
        )
        if not o.get("is_dead") or "are you sure" not in blob and "you are dead" not in blob:
            # left confirm or alive
            if "are you sure" not in blob:
                print("CONFIRM GONE")
                break
    print("FINAL dead=", o.get("is_dead"), (o.get("screen_ocr") or "")[:160])


if __name__ == "__main__":
    main()
