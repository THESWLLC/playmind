"""Sticky life / death finite-state machine.

Stops tick-to-tick flip-flopping between dead/ghost/alive and commits to
finishing each UI step before farming again.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

from playmind.ui_memory import explore_click_candidates, random_ui_probe


Phase = str  # alive | dead_dialog | confirm | rez_picker | ghost


def classify_life_raw(obs: dict[str, Any]) -> Phase:
    """One-frame classification from OCR + sensors (noisy — use with sticky)."""
    ocr = (obs.get("screen_ocr") or "").lower()
    hits = " ".join(str(h) for h in (obs.get("ui_hits") or [])).lower()
    blob = f"{ocr} {hits}"
    # OCR chunks are joined with " | " — normalize so phrases still match.
    blob_flat = re.sub(r"\s*\|\s*", " ", blob)
    blob_flat = re.sub(r"\s+", " ", blob_flat)
    desat = bool(obs.get("desaturated"))
    hp = float(obs.get("vision_player_hp") or obs.get("player", {}).get("hp") or 0.5)

    if "are you sure" in blob_flat or "want to return" in blob_flat:
        return "confirm"
    if "choose where" in blob_flat or (
        "closest" in blob_flat and any(k in blob_flat for k in ("town", "city", "gity", "cancel"))
    ):
        return "rez_picker"
    if "you are dead" in blob_flat:
        return "dead_dialog"
    if "return to graveyard" in blob_flat or "release spirit" in blob_flat:
        return "dead_dialog"
    # Split OCR often yields "resurrect in | a safe zone"
    if "resurrect" in blob_flat and "safe zone" in blob_flat:
        return "dead_dialog"
    if "graveyard" in blob_flat and ("return" in blob_flat or desat or hp < 0.1):
        return "dead_dialog"
    if re.search(r"\b\d+\s*yds?\b", blob_flat) or "spirit healer" in blob_flat:
        return "ghost"
    # Grey world + empty HP is strong evidence even before OCR catches the title.
    if desat and hp < 0.08:
        return "dead_dialog"
    if desat and obs.get("ghost_buttons"):
        return "ghost"
    return "alive"


@dataclass
class LifeFSM:
    """Hysteresis so one bad OCR frame cannot yank us out of death recovery."""

    phase: Phase = "alive"
    hold: int = 0  # ticks spent in current phase
    alive_evidence: int = 0
    _commit: str | None = None  # repeat same action a few times
    _commit_left: int = 0
    _tried: set[str] = field(default_factory=set)  # failed probes this phase

    def update(self, obs: dict[str, Any]) -> Phase:
        raw = classify_life_raw(obs)
        hp = float(obs.get("vision_player_hp") or 0)
        desat = bool(obs.get("desaturated"))

        # Strong alive evidence: color world + real HP
        if raw == "alive" and not desat and hp >= 0.12:
            self.alive_evidence += 1
        else:
            self.alive_evidence = 0

        if self.phase == "alive":
            if raw != "alive":
                self.phase = raw
                self.hold = 0
            else:
                self.hold += 1
            return self.phase

        # In any death-side phase: only advance on matching raw, or exit to alive
        # after sustained alive evidence.
        if self.alive_evidence >= 2 and raw == "alive":
            self.phase = "alive"
            self.hold = 0
            self._commit = None
            self._commit_left = 0
            self._tried.clear()
            return self.phase

        # Allow forward progress through the death pipeline.
        order = ("dead_dialog", "confirm", "rez_picker", "ghost", "alive")
        try:
            cur_i = order.index(self.phase) if self.phase in order else 0
            raw_i = order.index(raw) if raw in order else cur_i
        except ValueError:
            cur_i, raw_i = 0, 0

        if raw in {"dead_dialog", "confirm", "rez_picker", "ghost"}:
            # Move forward or stay; don't jump backwards more than one without evidence
            if raw_i >= cur_i or raw == self.phase:
                if raw != self.phase:
                    self.phase = raw
                    self.hold = 0
                    self._commit = None
                    self._commit_left = 0
                    self._tried.clear()
                else:
                    self.hold += 1
            else:
                # Sticky phases can outlive the dialog. Step back only with strong
                # evidence — never thrash confirm↔dead_dialog on OCR flicker.
                ocr_l = (obs.get("screen_ocr") or "").lower()
                ocr_flat = re.sub(r"\s*\|\s*", " ", ocr_l)
                still_confirm = (
                    "are you sure" in ocr_flat
                    or "want to return" in ocr_flat
                    or ("yes" in ocr_flat and "cancel" in ocr_flat)
                )
                strong_confirm = raw == "confirm" and self.phase == "rez_picker"
                if still_confirm and self.phase in {"dead_dialog", "confirm", "rez_picker"}:
                    if self.phase != "confirm":
                        self.phase = "confirm"
                        self.hold = 0
                        self._commit = None
                        self._commit_left = 0
                        self._tried.clear()
                    else:
                        self.hold += 1
                elif strong_confirm:
                    self.phase = "confirm"
                    self.hold = 0
                    self._commit = None
                    self._commit_left = 0
                    self._tried.clear()
                elif self.hold >= 6 and raw_i < cur_i:
                    self.phase = raw
                    self.hold = 0
                    self._commit = None
                    self._commit_left = 0
                    self._tried.clear()
                else:
                    self.hold += 1
        else:
            # raw==alive while we think we're dead/ghost — wait for evidence
            self.hold += 1

        return self.phase

    def action(
        self,
        obs: dict[str, Any],
        frame_path: Any = None,
        ui_memory: Any = None,
        process_memory: Any = None,
    ) -> tuple[str, str]:
        """Return (action, reason) for the sticky phase — never farm while dead."""
        phase = self.update(obs)
        ocr = (obs.get("screen_ocr") or "").lower()
        hits = " ".join(str(h) for h in (obs.get("ui_hits") or [])).lower()
        blob = f"{ocr} {hits}"

        # Short commits only — then try something else so we can learn.
        if self._commit and self._commit_left > 0 and self.hold < 3:
            self._commit_left -= 1
            return self._commit, f"life_fsm:{phase} commit→{self._commit}"

        if phase == "alive":
            return "", "life_fsm:alive"

        if obs.get("modal_menu") or "options" in blob or "exit game" in blob:
            act = "click_label:Close" if "close" in blob else "key:esc"
            return act, f"life_fsm:{phase} close Options"

        if "pyromancer" in blob or "item level" in blob or "intellect" in blob:
            return "key:c", f"life_fsm:{phase} close character panel"

        if phase == "confirm":
            wants = ["yes", "accept"]
            extras = ["key:enter"]
            reason = "confirm Yes"
            semantic = "click_label:yes"
        elif phase == "rez_picker":
            # Ascension: Closest Town / Closest City — NOT "safe zone" text.
            wants = ["closest town", "closest city", "closest", "town"]
            extras = []
            reason = "Closest Town"
            semantic = "click_label:closest town"
        elif phase == "dead_dialog":
            wants = ["return to graveyard", "release spirit", "resurrect in a safe zone"]
            extras = []
            reason = "death dialog"
            semantic = "click_label:return to graveyard"
        else:  # ghost
            if "spirit healer" in blob:
                act = "interact"
            else:
                # Occasionally turn while running so we learn directions.
                act = random.choice(
                    ["hold:w:1.0", "hold:w:1.0", "hold:a:0.4", "hold:d:0.4", "hold:w:1.2"]
                )
            self._set_commit(act, 1)
            return act, f"life_fsm:ghost runback"

        # Prefer actions that previously cleared this phase (process memory).
        if process_memory is not None:
            remembered = process_memory.best_pipeline_action(
                phase, [f"click_label:{w}" for w in wants] + extras
            )
            if remembered and remembered not in self._tried:
                self._tried.add(remembered)
                self._set_commit(remembered, 1)
                return remembered, f"life_fsm:{phase} memory→{remembered}"

        candidates = explore_click_candidates(frame_path, ui_memory, wants, ban="cancel")
        for h in obs.get("ui_hits") or []:
            hl = str(h).lower()
            if "cancel" in hl:
                continue
            if any(w in hl for w in wants):
                a = f"click_label:{h}"
                if a not in candidates:
                    candidates.insert(0, a)
        preferred = [f"click_label:{w}" for w in wants]
        ordered = [a for a in preferred if a in candidates or a.startswith("click_label:")]
        ordered.extend(a for a in candidates if a not in ordered)
        ordered.extend(extras)
        if not ordered:
            ordered = preferred + extras

        # Try→measure→remember: after first miss, randomly probe nearby pixels.
        # Escalate randomness the longer we stay stuck in-phase.
        explore_p = 0.35 if self.hold <= 1 else min(0.95, 0.45 + 0.08 * self.hold)
        if self.hold >= 1 and random.random() < explore_p:
            probe = random_ui_probe(frame_path, wants, tried=self._tried, memory=ui_memory)
            if probe:
                self._tried.add(probe)
                self._set_commit(probe, 1)
                return probe, f"life_fsm:{phase} random_try→{probe}"
            spatial = [a for a in ordered if a.startswith("click:") and a not in self._tried]
            if spatial:
                act = random.choice(spatial)
                self._tried.add(act)
                self._set_commit(act, 1)
                return act, f"life_fsm:{phase} random_try→{act}"
            # OCR dry — keyboard / jitter so we never infinite-spam the same label.
            if phase == "confirm":
                fallbacks = [a for a in ("key:enter", "key:y") if a not in self._tried]
                if not fallbacks:
                    fallbacks = ["key:enter", "key:y"]
                act = random.choice(fallbacks)
                self._tried.add(act)
                self._set_commit(act, 1)
                return act, f"life_fsm:{phase} stuck_fallback→{act}"

        # First attempt / memory-guided: semantic label (OCR resolve + offset).
        act = semantic
        for a in ordered:
            if phase == "rez_picker" and "closest town" in a.lower():
                act = a
                break
            if phase == "confirm" and a.lower().startswith("click_label:yes"):
                act = a
                break
            if phase == "dead_dialog" and "return to graveyard" in a.lower():
                act = a
                break
        # After many identical semantic misses, prefer Enter over another Yes click.
        if phase == "confirm" and self.hold >= 4 and act.lower().startswith("click_label:yes"):
            act = "key:enter"
        self._tried.add(act)
        self._set_commit(act, 1)
        return act, f"life_fsm:{phase} {reason}→{act}"

    def _set_commit(self, act: str, n: int) -> None:
        self._commit = act
        self._commit_left = max(0, n - 1)

    def note_try_failed(self, action: str) -> None:
        """Record a probe that did not advance the phase."""
        if action:
            self._tried.add(action)

    def patch_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Force coherent flags from sticky phase."""
        out = dict(obs)
        out["life_phase"] = self.phase
        if self.phase in {"dead_dialog", "confirm", "rez_picker"}:
            out["is_dead"] = True
            out["is_ghost"] = False
            out["has_target"] = False
            out["in_combat"] = False
            out["vision_player_hp"] = 0.0
            out["confirm_pending"] = self.phase == "confirm"
            if isinstance(out.get("player"), dict):
                out["player"] = {**out["player"], "hp": 0.0}
        elif self.phase == "ghost":
            out["is_dead"] = False
            out["is_ghost"] = True
            out["has_target"] = False
            out["vision_player_hp"] = 0.0
        elif self.phase == "alive":
            out["is_dead"] = False
            out["is_ghost"] = False
            # Distrust near-zero HP while colored/alive sensors say alive
            hp = float(out.get("vision_player_hp") or 0)
            if hp < 0.05 and not out.get("desaturated"):
                out["vision_player_hp"] = 0.5  # unknown but not "dying"
                if isinstance(out.get("player"), dict):
                    out["player"] = {**out["player"], "hp": 0.5}
        return out
