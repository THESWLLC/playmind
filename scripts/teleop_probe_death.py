"""Probe click grid on Ascension death UI until 'You are dead' clears."""

from __future__ import annotations

import time
from pathlib import Path

from playmind.actuators import OwnedGameKeyboardActuator, load_keymap
from playmind.capture import capture_config_from_dict
from playmind.owned_loop import load_owned_config, vision_obs_from_frame
from playmind.screen_llm import enrich_obs_from_screen
from playmind.ui_memory import UIMemory, ensure_seeded

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "playmind" / "owned"


def main() -> None:
    owned = load_owned_config(ROOT / "config" / "owned_game.json")
    rois = owned.get("rois", {})
    keymap = load_keymap(Path(owned.get("keymap_path", "config/keymap.example.json")))
    ui = UIMemory(DATA / "ui_memory.json")
    ensure_seeded(ui)
    act = OwnedGameKeyboardActuator(
        keymap=keymap,
        enabled=True,
        i_own_this_game=True,
        window_title_substr="Ascension",
        ui_memory=ui,
        log_path=ROOT / "data" / "playmind" / "teleop_probe.jsonl",
    )
    frame = DATA / "teleop_latest.png"
    cap = owned.get("capture", {})

    def snap_ocr() -> str:
        capture_config_from_dict(cap, frame)
        o = enrich_obs_from_screen(frame, vision_obs_from_frame(frame, rois))
        return (o.get("screen_ocr") or "").lower()

    # Close options if any
    ocr = snap_ocr()
    print("start", ocr[:160])
    if "options" in ocr or "exit game" in ocr:
        act.send("key:esc")
        time.sleep(0.6)
        ocr = snap_ocr()
        print("after esc", ocr[:160])

    # Dense probe of top-center death bar
    candidates = []
    for fy in (0.055, 0.07, 0.085, 0.10, 0.115, 0.13, 0.145):
        for fx in (0.38, 0.42, 0.46, 0.50, 0.54, 0.58, 0.62):
            candidates.append((fx, fy))

    for fx, fy in candidates:
        action = f"click:{fx:.3f},{fy:.3f}"
        print(f"try {action}")
        act.send(action)
        time.sleep(0.7)
        ocr = snap_ocr()
        deadish = ("you are dead" in ocr) or ("return to" in ocr and "graveyard" in ocr) or (
            "resurrect" in ocr and "safe" in ocr
        )
        print(f"  deadish={deadish} ocr={ocr[:100]!r}")
        if not deadish:
            print(f"*** HIT at fx={fx} fy={fy} ***")
            ui.remember("Return to Graveyard", fx, fy, source="probe", success=True)
            ui.remember("Resurrect in a Safe Zone", fx, fy, source="probe", success=True)
            with (DATA / "death_click_hit.txt").open("w") as f:
                f.write(f"{fx},{fy}\n")
            return

    print("no hit found")


if __name__ == "__main__":
    main()
