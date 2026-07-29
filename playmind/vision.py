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
    else:
        # Never scan full ultrawide — default to top-left player HP strip.
        w, h = img.size
        img = img.crop((int(w * 0.02), int(h * 0.03), int(w * 0.14), int(h * 0.09)))
    # Tiny resize keeps estimate stable and cheap.
    if img.size[0] > 120 or img.size[1] > 40:
        img = img.resize((min(120, img.size[0]), min(40, max(8, img.size[1]))))
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


def _configure_tesseract() -> bool:
    try:
        import pytesseract  # type: ignore
    except ImportError:
        return False
    from pathlib import Path

    configured = getattr(pytesseract.pytesseract, "tesseract_cmd", "") or ""
    if configured and Path(configured).exists():
        return True
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return True
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_image(path: Path) -> str:
    """OCR with pytesseract if installed; else empty string."""
    img = _try_pil_open(path)
    if img is None:
        return ""
    if not _configure_tesseract():
        return ""
    try:
        import pytesseract  # type: ignore

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


def detect_death_dialog(path: Path) -> tuple[bool, bool]:
    """Detect WoW-like death dialog / ghost world.

    Ascension shows greyscale world + top buttons
    ("Return to Graveyard" / "Resurrect in a Safe Zone"), not classic red Release.

    Returns (is_dead, is_ghost_world).
    """
    img = _try_pil_open(path)
    if img is None:
        return False, False
    w, h = img.size

    # Player HP strip empty?
    hp_strip = img.crop((int(w * 0.03), int(h * 0.04), int(w * 0.12), int(h * 0.09)))
    hp_strip = hp_strip.resize((64, 16))
    hp_px = list(hp_strip.getdata())
    greenish = sum(1 for r, g, b in hp_px if g > 120 and g > r and g > b)
    empty_hp = (greenish / max(1, len(hp_px))) < 0.02

    # Greyscale / spirit world: low saturation mid-frame (downsampled)
    sample = img.crop((int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)))
    sample = sample.resize((80, 45))
    sp = list(sample.getdata())
    sat = 0
    for r, g, b in sp:
        mx, mn = max(r, g, b), min(r, g, b)
        if mx > 0 and (mx - mn) / mx > 0.18:
            sat += 1
    low_sat = (sat / max(1, len(sp))) < 0.28

    # Classic gold/red release dialog (retail-ish)
    dialog = img.crop((int(w * 0.35), int(h * 0.05), int(w * 0.65), int(h * 0.22)))
    dialog = dialog.resize((120, 48))
    pixels = list(dialog.getdata())
    dark = sum(1 for r, g, b in pixels if r + g + b < 120)
    gold = sum(1 for r, g, b in pixels if r > 140 and g > 100 and b < 90 and r > b)
    red_btn = sum(1 for r, g, b in pixels if r > 140 and g < 90 and b < 90)
    classic_dialog = bool(pixels) and (dark / len(pixels) > 0.30) and (gold >= 8) and (red_btn >= 4)

    # Ascension: empty HP + desaturated world = dead/ghost
    spirit = bool(empty_hp and low_sat)
    is_dead = classic_dialog or spirit
    is_ghost = spirit and not classic_dialog
    return is_dead, is_ghost


_MOB_LABEL_WORDS = (
    "boar",
    "grell",
    "nightsaber",
    "wolf",
    "spider",
    "crab",
    "kobold",
    "gnoll",
    "murloc",
    "bandit",
    "scorpion",
    "thistle",
    "sabre",
    "saber",
    "sprite",
)


def is_world_mob_label(label: str) -> bool:
    """True if OCR/UI text is a world nameplate, not a clickable UI button."""
    low = (label or "").lower().strip()
    if not low:
        return False
    if any(w in low for w in _MOB_LABEL_WORDS):
        return True
    # "Young Thistle Boar", "Level 1 Beast", etc.
    if re.search(r"\byoung\b|\blevel\s*\d|\bbeast\b|\bhostile\b", low):
        return True
    return False


def detect_target_bar(
    path: Path,
    roi: tuple[int, int, int, int] | None = None,
) -> tuple[bool, float | None]:
    """Detect an active target (unit frame OR red ground selection circle).

    Ascension often shows only a red ring under the mob + floating nameplate,
    with no classic top-center target frame — especially on ultrawide.
    """
    img = _try_pil_open(path)
    if img is None:
        return False, None
    w, h = img.size

    def _red_ratio(im, box: tuple[int, int, int, int]) -> float:
        crop = im.crop(box)
        # Downsample crop before counting — getdata() on full tiles was ~1s+/frame.
        cw, ch = crop.size
        if cw > 64 or ch > 48:
            crop = crop.resize((min(64, cw), min(48, ch)))
        pixels = list(crop.getdata())
        if not pixels:
            return 0.0
        red = sum(
            1
            for r, g, b in pixels
            if r > 130 and r > g + 30 and r > b + 30
        )
        return red / max(1, len(pixels))

    if roi is not None:
        ratio = _red_ratio(img, roi)
        return ratio > 0.015, min(1.0, ratio * 8.0)

    # 1) Classic target-frame ROIs (may be empty on Ascension).
    frame_rois = (
        (int(w * 0.08), int(h * 0.01), int(w * 0.28), int(h * 0.14)),
        (int(w * 0.35), int(h * 0.02), int(w * 0.62), int(h * 0.12)),
    )
    best_frame = max(_red_ratio(img, box) for box in frame_rois)

    # 2) Tile-scan a downscaled world view for a red selection ring.
    #    Full ultrawide tile getdata() was the #1 tick bottleneck (~1.3s).
    wx0, wy0 = int(w * 0.22), int(h * 0.28)
    wx1, wy1 = int(w * 0.78), int(h * 0.72)
    world = img.crop((wx0, wy0, wx1, wy1))
    ww, wh = world.size
    scale = 480 / max(1, ww)
    if scale < 1.0:
        world = world.resize((480, max(1, int(wh * scale))))
        ww, wh = world.size
    cx0, cx1 = int(ww * 0.28), int(ww * 0.72)
    cy0, cy1 = int(wh * 0.45), int(wh * 1.01)
    tile_w, tile_h = max(24, ww // 14), max(18, wh // 10)
    best_tile = 0.0
    y = 0
    while y < wh:
        x = 0
        while x < ww:
            overlaps_player = not (x >= cx1 or x + tile_w <= cx0 or y >= cy1 or y + tile_h <= cy0)
            if not overlaps_player:
                ratio = _red_ratio(world, (x, y, min(ww, x + tile_w), min(wh, y + tile_h)))
                if ratio > best_tile:
                    best_tile = ratio
            x += tile_w
        y += tile_h

    has = best_frame > 0.012 or best_tile > 0.055
    conf = 0.0
    if has:
        conf = min(1.0, max(best_frame * 8.0, best_tile * 6.0))
    return has, conf


def detect_hostile_nameplate_ocr(ocr_text: str) -> bool:
    """True if OCR looks like hostile nameplates are on-screen (not necessarily selected)."""
    return is_world_mob_label(ocr_text or "")


def frame_motion(prev_path: Path | None, cur_path: Path, sample_box: tuple[int, int, int, int] | None = None) -> float:
    """Mean RGB abs-diff in a center crop; higher ⇒ camera/world moved."""
    if prev_path is None or not prev_path.exists() or not cur_path.exists():
        return 0.0
    try:
        from PIL import ImageChops, ImageStat  # type: ignore
    except ImportError:
        return 0.0
    a = _try_pil_open(prev_path)
    b = _try_pil_open(cur_path)
    if a is None or b is None:
        return 0.0
    if a.size != b.size:
        b = b.resize(a.size)
    if sample_box is None:
        w, h = a.size
        sample_box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    a = a.crop(sample_box)
    b = b.crop(sample_box)
    diff = ImageChops.difference(a, b)
    mean = ImageStat.Stat(diff).mean
    return float(sum(mean) / max(1, len(mean)))


def read_frame(
    path: str | Path,
    hp_roi: tuple[int, int, int, int] | None = None,
    *,
    do_ocr: bool = False,
) -> VisionReading:
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
    # Full-frame OCR here was ~seconds per tick on ultrawide — use ROI OCR in enrich instead.
    text = ""
    if do_ocr:
        text = ocr_image(path)
        if not text:
            notes.append("ocr_unavailable_or_empty")
    else:
        notes.append("ocr_skipped")
    return VisionReading(
        player_hp=hp,
        quest_text=parse_quest_from_text(text) if text else None,
        raw_text=text,
        notes=notes,
    )


def save_demo_ascii_frame(ascii_map: str, quest_text: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ascii_map + "\n\nQUEST: " + quest_text + "\n", encoding="utf-8")
    return out
