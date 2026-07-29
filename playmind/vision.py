"""Vision helpers: load frames, estimate bars, OCR text (optional deps).

Works with:
- image files (png/jpg)
- numpy arrays if available
- demo ASCII render saved as a text "frame" fallback

Heavy deps (Pillow / pytesseract / opencv) are optional.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class VisionReading:
    player_hp: float | None = None
    quest_text: str | None = None
    raw_text: str = ""
    notes: list[str] | None = None

    def to_obs_patch(self) -> dict[str, Any]:
        patch: dict[str, Any] = {}
        if self.player_hp is not None:
            patch["vision_player_hp"] = self.player_hp
        if self.quest_text:
            patch["vision_quest_text"] = self.quest_text
        if self.raw_text:
            patch["vision_raw_text"] = self.raw_text
        return patch


def _try_pil_open(path: Path):
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return None
    return Image.open(path).convert("RGB")


def estimate_hp_from_image(path: Path, roi: tuple[int, int, int, int] | None = None) -> float | None:
    """Estimate fill ratio of a reddish/green health bar ROI via Pillow."""
    img = _try_pil_open(path)
    if img is None:
        return None
    if roi:
        img = img.crop(roi)
    pixels = list(img.getdata())
    if not pixels:
        return None
    # Count "bar-like" saturated pixels vs dark background.
    lit = 0
    for r, g, b in pixels:
        if r + g + b < 40:
            continue
        if r > 100 or g > 100:
            lit += 1
    return max(0.0, min(1.0, lit / max(1, len(pixels))))


def ocr_image(path: Path) -> str:
    """OCR with pytesseract if installed; else empty string."""
    img = _try_pil_open(path)
    if img is None:
        return ""
    try:
        import pytesseract  # type: ignore
    except ImportError:
        return ""
    try:
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""


_QUEST_LINE = re.compile(r"(kill|collect|talk|deliver|quest)[^\n]{0,120}", re.I)


def parse_quest_from_text(text: str) -> str | None:
    if not text.strip():
        return None
    m = _QUEST_LINE.search(text)
    if m:
        return m.group(0).strip()
    # Fallback: first nontrivial line
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 8:
            return line
    return None


def read_frame(path: str | Path, hp_roi: tuple[int, int, int, int] | None = None) -> VisionReading:
    path = Path(path)
    notes: list[str] = []
    if not path.exists():
        return VisionReading(notes=[f"missing frame: {path}"])

    if path.suffix.lower() in {".txt", ".ascii"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        notes.append("ascii_frame")
        return VisionReading(quest_text=parse_quest_from_text(text), raw_text=text, notes=notes)

    hp = estimate_hp_from_image(path, hp_roi)
    if hp is None:
        notes.append("pillow_unavailable_or_failed")
    text = ocr_image(path)
    if not text:
        notes.append("ocr_unavailable_or_empty")
    return VisionReading(
        player_hp=hp,
        quest_text=parse_quest_from_text(text),
        raw_text=text,
        notes=notes,
    )


def save_demo_ascii_frame(ascii_map: str, quest_text: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ascii_map + "\n\nQUEST: " + quest_text + "\n", encoding="utf-8")
    return out
