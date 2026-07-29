"""Screen → LLM action brain for owned games.

Uses an Ollama vision model to look at the live frame and choose one action.
Falls back to text LLM + OCR/sensor summary if no vision model is available.
"""

from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from playmind.learning import OWNED_ACTIONS
from playmind.planner import HeuristicPlanner, OllamaPlanner, Planner, ollama_available


SCREEN_VISION_SYSTEM = """You are the SOUL of an owned MMO character (Ascension / WoW-like).
You inhabit this body. You SEE the screenshot. Feel it — do not thrash randomly.

YOUR INNER STATE:
{soul}

{goal_rules}

Reply with EXACTLY ONE action line:
1) Known names: {actions}
2) key:<key>              e.g. key:2   key:tab   key:esc
3) hold:<key>:<seconds>   e.g. hold:w:0.6
4) ability:<name>         remembered skill
5) bind:<Name>=<key>      name a spell you SEE on the action bar (read the icon / tooltip)
6) click_label:<text>     only real UI buttons (Close, Yes, Accept, Return to Graveyard…)
7) click:<fx>,<fy>        fractional coords

Remembered abilities: {abilities}

HOW A SOUL ACTS:
1) If DEAD / grey / "You are dead" / rez dialog → ONLY death UI (never Tab/attack/move to farm).
   - "Are you sure?" → click_label:Yes (or click the Yes button you SEE)
   - "Choose where to resurrect" → click_label:Closest Town (the red button — NOT Cancel)
   - "Return to Graveyard" / "Resurrect in a Safe Zone" → click that button
   - Prefer click:fx,fy on the button center if you can see it
2) If GHOST (wispy, "N yds" to corpse) → hold:w toward corpse / interact spirit healer
3) If ALIVE with a red target ring → cast key:1 / key:2
4) If ALIVE, no target, farming → Tab then cast; walk if nothing near
5) READ the action bar: bind:SpellName=1 when you can name a spell
6) Never click enemy nameplates. Never open Options while dead on purpose.

Stuck hint: {stuck_hint}

Reply with ONLY the action line. No explanation."""


_DEATH_LABELS = (
    "accept",
    "yes",
    "resurrect now",
    "return to graveyard",
    "release spirit",
    "resurrect in a safe zone",
)

_MODAL_MARKERS = (
    "options",
    "key bindings",
    "quick binding",
    "video",
    "sound",
    "interface",
    "addons",
    "add ons",
    "macros",
    "logout",
    "exit game",
    "join discord",
    "report bug",
)


def _death_recovery_action(obs: dict[str, Any]) -> str:
    """Pick a remembered / OCR'd death UI control instead of only hardcoded clicks."""
    ocr = (obs.get("screen_ocr") or "").lower()
    for label in _DEATH_LABELS:
        if label in ocr:
            return f"click_label:{label}"
    hits = obs.get("ui_hits") or []
    for label in _DEATH_LABELS:
        for hit in hits:
            if label in str(hit).lower():
                return f"click_label:{hit}"
    return "release_spirit"


def _modal_close_action(obs: dict[str, Any]) -> str | None:
    """If a blocking game menu is open, close it (Esc or Close button)."""
    if not obs.get("modal_menu"):
        return None
    ocr = (obs.get("screen_ocr") or "").lower()
    hits = " ".join(str(h) for h in (obs.get("ui_hits") or [])).lower()
    blob = f"{ocr} {hits}"
    if "close" in blob:
        return "click_label:Close"
    # Escape always dismisses WoW/Ascension Options / Game Menu
    return "key:esc"


def _resize_jpeg_b64(path: Path, max_width: int = 1024, quality: int = 70) -> str:
    from PIL import Image  # type: ignore

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w > max_width:
        nh = int(h * (max_width / w))
        img = img.resize((max_width, nh))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _ocr_regions(path: Path, *, mode: str = "full") -> str:
    """OCR via shared ROI boxes (one tesseract pass — cached per frame)."""
    from playmind.ui_memory import ocr_text_boxes

    hits = ocr_text_boxes(path, min_conf=30, mode=mode)
    if not hits:
        return ""
    # Prefer higher-confidence / longer UI phrases
    ranked = sorted(hits, key=lambda h: (-(len(h.label)), -h.conf))
    chunks: list[str] = []
    seen: set[str] = set()
    for h in ranked:
        if h.label in seen or len(h.label) < 2:
            continue
        seen.add(h.label)
        chunks.append(h.label)
        if len(chunks) >= 12:
            break
    return " | ".join(chunks)[:700]


def list_ollama_models(host: str = "http://127.0.0.1:11434") -> list[str]:
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [str(m.get("name", "")) for m in data.get("models", [])]
    except Exception:
        return []


def pick_vision_model(preferred: str | None = None, host: str = "http://127.0.0.1:11434") -> str | None:
    models = list_ollama_models(host)
    if preferred and any(preferred in m or m.startswith(preferred) for m in models):
        return next(m for m in models if preferred in m or m.startswith(preferred))
    # Prefer stronger UI/OCR vision models first; llava is legacy fallback.
    for needle in (
        "qwen2.5vl",
        "qwen3-vl",
        "qwen2-vl",
        "gemma3",
        "llama3.2-vision",
        "minicpm-v",
        "llava",
        "moondream",
        "bakllava",
    ):
        for m in models:
            if needle in m.lower():
                return m
    return None


@dataclass
class ScreenLLMPlanner:
    """Look at the live frame with a vision LLM, then choose one owned-game action."""

    vision_model: str = "qwen2.5vl:7b"
    text_model: str = "llama3.2"
    host: str = "http://127.0.0.1:11434"
    actions: Sequence[str] = field(default_factory=lambda: OWNED_ACTIONS)
    ability_summary: str = "(none yet — invent with bind:Name=key)"
    timeout_s: float = 90.0
    fallback: Planner | None = None
    last_raw: str = ""
    last_error: str = ""
    last_see: str = ""
    last_mode: str = ""
    last_prompt_hint: str = ""

    def __post_init__(self) -> None:
        if self.fallback is None:
            self.fallback = OllamaPlanner.for_owned(model=self.text_model)

    def plan(
        self,
        obs: dict[str, Any],
        directive: str | None = None,
        frame_path: Path | None = None,
    ) -> str:
        # No Ascension-specific scripts here — the VLM invents; stuck/reward teach what works.
        if frame_path and Path(frame_path).exists():
            action = self._plan_from_image(Path(frame_path), obs, directive)
            if action:
                return action

        # Text fallback with OCR + sensors
        ocr = ""
        if frame_path and Path(frame_path).exists():
            ocr = _ocr_regions(Path(frame_path))
            self.last_see = ocr
        obs = dict(obs)
        obs["screen_ocr"] = ocr
        self.last_mode = "text_llm"
        try:
            return self.fallback.plan(obs, directive)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            return "target_nearest"

    def _plan_from_image(
        self,
        frame_path: Path,
        obs: dict[str, Any],
        directive: str | None,
    ) -> str | None:
        model = pick_vision_model(self.vision_model, self.host)
        if not model:
            self.last_error = "no_vision_model"
            return None
        try:
            b64 = _resize_jpeg_b64(frame_path)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"image_encode:{exc}"
            return None

        abilities = obs.get("ability_summary") or self.ability_summary
        stuck_hint = obs.get("stuck_hint") or "none"
        goal_rules = obs.get("goal_rules") or (
            f"PLAYER DIRECTIVE: {directive or 'farm'}\n"
            "This is an MMO: tab-target, keys 1-5 attack, Esc closes menus."
        )
        sensors = {
            "directive": directive,
            "goal": obs.get("goal_summary"),
            "soul": obs.get("soul_summary"),
            "body": obs.get("soul_body"),
            "hp": obs.get("vision_player_hp"),
            "has_target": obs.get("has_target"),
            "is_dead": obs.get("is_dead"),
            "is_ghost": obs.get("is_ghost"),
            "modal_menu": obs.get("modal_menu"),
            "confirm_pending": obs.get("confirm_pending"),
            "desaturated": obs.get("desaturated"),
            "ghost_buttons": obs.get("ghost_buttons"),
            "bar_slots_filled": obs.get("bar_slots_filled"),
            "ui_hits": obs.get("ui_hits"),
            "ocr": (obs.get("screen_ocr") or "")[:200],
            "stuck_hint": stuck_hint,
        }
        prompt = (
            SCREEN_VISION_SYSTEM.format(
                actions=", ".join(self.actions),
                abilities=abilities,
                stuck_hint=stuck_hint,
                goal_rules=goal_rules,
                soul=obs.get("soul_summary")
                or "I am unsure of my body. Check if the world is grey (dead) or colored (alive).",
            )
            + "\nSensors (may help): "
            + json.dumps(sensors, sort_keys=True)
            + "\nAction:"
        )
        self.last_prompt_hint = f"goal={obs.get('goal_summary') or directive} stuck={stuck_hint[:40]}"
        try:
            text = self._generate_vision(model, prompt, b64)
            self.last_raw = text
            self.last_error = ""
            self.last_mode = f"vision:{model}"
            self.last_see = text[:400]
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            self.last_error = str(exc)
            self.last_mode = "vision_failed"
            return None

        return self._parse_action(text)

    def _parse_action(self, text: str) -> str | None:
        from playmind.ability_memory import parse_dynamic_action

        cleaned = text.strip()
        for line in cleaned.splitlines():
            line = line.strip().strip("`\"'")
            if not line:
                continue
            # Strip "Action:" prefixes models sometimes echo
            if ":" in line and line.lower().split(":", 1)[0].strip() in {
                "action",
                "output",
                "answer",
                "response",
            }:
                line = line.split(":", 1)[1].strip()
            dyn = parse_dynamic_action(line)
            if dyn:
                return _format_dynamic(dyn)
            token = line.lower().replace("-", "_").split()[0].strip("`\"'.,: ")
            for action in self.actions:
                if token == action:
                    return action
        low = cleaned.lower().replace("-", "_")
        dyn = parse_dynamic_action(cleaned)
        if dyn:
            return _format_dynamic(dyn)
        for action in self.actions:
            if action in low:
                return action
        if "graveyard" in low:
            return "click_label:Return to Graveyard"
        if "release spirit" in low:
            return "click_label:Release Spirit"
        if "resurrect" in low or "accept" in low:
            return "click_label:Accept"
        if "attack" in low or "cast" in low:
            return "attack"
        if "tab" in low or "target" in low:
            return "target_nearest"
        # Bare digit → invent key press
        m = re.search(r"\b([1-9]|f1[0-2]?|tab)\b", low)
        if m:
            return f"key:{m.group(1)}"
        return None

    def _generate_vision(self, model: str, prompt: str, image_b64: str) -> str:
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 64},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host.rstrip('/')}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return str(payload.get("response", ""))


def _format_dynamic(dyn: dict[str, Any]) -> str:
    t = dyn.get("type")
    if t == "click_frac":
        return f"click:{dyn['fx']:.3f},{dyn['fy']:.3f}"
    if t == "click_label":
        return f"click_label:{dyn['label']}"
    if t == "key":
        return f"key:{dyn['key']}"
    if t == "hold":
        return f"hold:{dyn['key']}:{dyn['seconds']}"
    if t == "ability":
        return f"ability:{dyn['name']}"
    if t == "bind":
        return f"bind:{dyn['name']}={dyn['key']}"
    return str(dyn)


def enrich_obs_from_screen(
    frame_path: Path,
    obs: dict[str, Any],
    *,
    do_ocr: bool = True,
    ocr_mode: str | None = None,
) -> dict[str, Any]:
    """Add desaturation / ghost-button heuristics for the screen brain.

    do_ocr=False → pixel sensors only (~fast). Use after actions while alive.
    """
    out = dict(obs)
    try:
        from PIL import Image  # type: ignore

        img = Image.open(frame_path).convert("RGB")
        w, h = img.size
        sample = img.resize((160, 90))
        px = list(sample.getdata())
        sat = 0
        for r, g, b in px:
            mx, mn = max(r, g, b), min(r, g, b)
            if mx > 0 and (mx - mn) / mx > 0.2:
                sat += 1
        sat_frac = sat / max(1, len(px))
        out["desaturated"] = sat_frac < 0.35

        # Top-center button band: dark panels with gold text common on ghost UI
        band = img.crop((int(w * 0.30), int(h * 0.03), int(w * 0.70), int(h * 0.16)))
        bp = list(band.getdata())
        dark = sum(1 for r, g, b in bp if r + g + b < 140)
        gold = sum(1 for r, g, b in bp if r > 150 and g > 110 and b < 100)
        out["ghost_buttons"] = (dark / max(1, len(bp)) > 0.25) and gold > 80 and out["desaturated"]
        if out["ghost_buttons"]:
            out["is_ghost"] = True
    except Exception:
        pass

    if not do_ocr:
        return out

    mode = ocr_mode or ("death" if out.get("desaturated") or out.get("is_dead") else "alive")
    ocr_text = _ocr_regions(frame_path, mode=mode)
    if ocr_text:
        out["screen_ocr"] = ocr_text
        low = ocr_text.lower()
        if "release spirit" in low or "return to graveyard" in low or "resurrect" in low:
            out["is_dead"] = True
            out["ghost_buttons"] = True
        if "you are dead" in low or ("safe zone" in low and "resurrect" in low):
            out["is_dead"] = True
            out["ghost_buttons"] = True
        if "are you sure" in low and "graveyard" in low:
            out["is_dead"] = True
            out["confirm_pending"] = True
            out["ghost_buttons"] = True
        # Already released: corpse distance / spirit healer ⇒ ghost, not death dialog.
        if re.search(r"\b\d+\s*yds?\b", low) or "spirit healer" in low:
            out["is_ghost"] = True
            out["is_dead"] = False
            out["confirm_pending"] = False
            out["ghost_buttons"] = False
        elif "return to graveyard" in low and out.get("desaturated"):
            out["is_ghost"] = True
        # Options / Game Menu style blocking overlay
        markers = sum(1 for m in _MODAL_MARKERS if m in low)
        if ("close" in low and markers >= 1) or markers >= 2 or "options" in low:
            out["modal_menu"] = True
    # Death nullifies targeting — keep sensors consistent for learning.
    if out.get("is_dead"):
        out["has_target"] = False
        out["in_combat"] = False
        out["hostiles_near"] = False
        out["vision_player_hp"] = 0.0
        if isinstance(out.get("player"), dict):
            out["player"] = {**out["player"], "hp": 0.0}
    return out
