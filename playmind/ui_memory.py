"""Runtime UI memory: discover new on-screen controls without restarting.

When OCR/VLM sees a new button or dialog, we store a click target
(fractional client coords) under a label and reuse it next time.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class UIHit:
    label: str
    fx: float  # 0..1 of client width
    fy: float  # 0..1 of client height
    conf: float = 1.0
    source: str = "ocr"


@dataclass
class UIMemory:
    path: Path
    labels: dict[str, dict[str, Any]] = field(default_factory=dict)
    _mtime: float = 0.0

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            mtime = self.path.stat().st_mtime
            if mtime <= self._mtime and self.labels:
                return
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.labels = {str(k).lower(): v for k, v in raw.get("labels", {}).items()}
            self._mtime = mtime
        except Exception:
            pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": time.time(),
            "labels": self.labels,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._mtime = self.path.stat().st_mtime

    def remember(
        self,
        label: str,
        fx: float,
        fy: float,
        *,
        source: str = "discover",
        success: bool | None = None,
    ) -> None:
        key = _norm_label(label)
        if not key:
            return
        fx = float(min(0.98, max(0.02, fx)))
        fy = float(min(0.98, max(0.02, fy)))
        row = self.labels.get(key, {"fx": fx, "fy": fy, "hits": 0, "successes": 0, "source": source})
        # EMA toward new observation
        old_fx, old_fy = float(row.get("fx", fx)), float(row.get("fy", fy))
        row["fx"] = round(0.7 * old_fx + 0.3 * fx, 4)
        row["fy"] = round(0.7 * old_fy + 0.3 * fy, 4)
        row["hits"] = int(row.get("hits", 0)) + 1
        row["source"] = source
        row["last_seen"] = time.time()
        if success is True:
            row["successes"] = int(row.get("successes", 0)) + 1
        self.labels[key] = row
        self.save()

    def lookup(self, label: str) -> tuple[float, float] | None:
        self.load()
        key = _norm_label(label)
        if key in self.labels:
            row = self.labels[key]
            return float(row["fx"]), float(row["fy"])
        # fuzzy contains
        for k, row in self.labels.items():
            if key in k or k in key:
                return float(row["fx"]), float(row["fy"])
        return None

    def known_labels(self) -> list[str]:
        self.load()
        return sorted(self.labels.keys())


def _norm_label(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", (label or "").lower()).strip()
    return re.sub(r"\s+", " ", s)


def _ocr_pil_data(img: Any, min_conf: int) -> list[tuple[str, float, float, float, float, float]]:
    """Run tesseract image_to_data; return (label, conf01, left, top, right, bottom) in image px."""
    import pytesseract  # type: ignore

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n = len(data.get("text", []))
    lines: dict[tuple[int, int], list[int]] = {}
    short_words = {"yes", "no", "ok", "accept", "cancel", "close", "continue", "town", "city"}
    out: list[tuple[str, float, float, float, float, float]] = []
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf < min_conf:
            continue
        key = (int(data["block_num"][i]), int(data["line_num"][i]))
        lines.setdefault(key, []).append(i)
        word = _norm_label(text)
        if word in short_words or len(word) >= 4:
            left, top = int(data["left"][i]), int(data["top"][i])
            ww, hh = int(data["width"][i]), int(data["height"][i])
            out.append((word, conf / 100.0, left, top, left + ww, top + hh))
    for idxs in lines.values():
        words = [(data["text"][i] or "").strip() for i in idxs]
        label = _norm_label(" ".join(words))
        if len(label) < 2:
            continue
        xs, ys, x2s, y2s, confs = [], [], [], [], []
        for i in idxs:
            left, top = int(data["left"][i]), int(data["top"][i])
            ww, hh = int(data["width"][i]), int(data["height"][i])
            xs.append(left)
            ys.append(top)
            x2s.append(left + ww)
            y2s.append(top + hh)
            try:
                confs.append(float(data["conf"][i]))
            except Exception:
                pass
        out.append(
            (
                label,
                (sum(confs) / max(1, len(confs))) / 100.0,
                min(xs),
                min(ys),
                max(x2s),
                max(y2s),
            )
        )
    return out


# Tight ROIs — full ultrawide OCR is noisy (chat, skills panel, fonts).
# (left, top, right, bottom, scale)
_ROI_DIALOG = (0.36, 0.03, 0.64, 0.28, 2)  # scale 2 ≈176ms; scale 3 was ~266ms
_ROI_MODAL = (0.34, 0.18, 0.66, 0.55, 2)
_ROI_PLAYER = (0.01, 0.01, 0.18, 0.12, 2)

_UI_ROIS_BY_MODE: dict[str, list[tuple[float, float, float, float, int]]] = {
    "death": [_ROI_DIALOG],  # rez / confirm — fast + accurate enough
    "alive": [_ROI_PLAYER],  # cheap HP/name check
    "full": [_ROI_DIALOG, _ROI_MODAL, _ROI_PLAYER],
}

# Same-file cache so enrich + discover don't double-run tesseract in one tick.
_OCR_CACHE: dict[str, tuple[float, str, list[UIHit]]] = {}


def ocr_text_boxes(
    path: Path,
    min_conf: int = 40,
    *,
    mode: str = "full",
) -> list[UIHit]:
    """OCR important UI bands (upscaled), not the whole ultrawide frame.

    mode: death | alive | full — fewer ROIs = much faster ticks.
    """
    try:
        from PIL import Image, ImageOps, ImageEnhance  # type: ignore
    except ImportError:
        return []
    from playmind.vision import _configure_tesseract

    if not _configure_tesseract():
        return []
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    cache_key = str(path.resolve())
    cached = _OCR_CACHE.get(cache_key)
    if cached and cached[0] == mtime and cached[1] == mode:
        return list(cached[2])

    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return []
    w, h = img.size
    hits: list[UIHit] = []
    seen: set[tuple[str, int, int]] = set()
    rois = _UI_ROIS_BY_MODE.get(mode) or _UI_ROIS_BY_MODE["full"]

    for L, T, R, B, scale in rois:
        box = (int(w * L), int(h * T), int(w * R), int(h * B))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        crop = img.crop(box)
        big = crop.resize((crop.size[0] * scale, crop.size[1] * scale), Image.Resampling.LANCZOS)
        try:
            gray = ImageEnhance.Contrast(ImageOps.grayscale(big)).enhance(2.0)
            rows = _ocr_pil_data(gray, min_conf=max(25, min_conf - 10))
        except Exception:
            continue
        for label, conf, left, top, right, bottom in rows:
            cx = box[0] + ((left + right) / 2.0) / scale
            cy = box[1] + ((top + bottom) / 2.0) / scale
            fx, fy = cx / max(1, w), cy / max(1, h)
            key = (label, int(fx * 200), int(fy * 200))
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                UIHit(
                    label=label,
                    fx=fx,
                    fy=fy,
                    conf=conf,
                    source="ocr_roi",
                )
            )
    _OCR_CACHE[cache_key] = (mtime, mode, list(hits))
    if len(_OCR_CACHE) > 8:
        # Drop oldest entries
        for k in list(_OCR_CACHE.keys())[: max(0, len(_OCR_CACHE) - 4)]:
            _OCR_CACHE.pop(k, None)
    return hits


def find_label_on_frame(
    path: Path,
    label: str,
    min_conf: int = 25,
    *,
    mode: str | None = None,
) -> list[UIHit]:
    """Live OCR search for a label — agent discovery, not hardcoded coords."""
    want = _norm_label(label)
    if not want:
        return []
    if mode is None:
        death_keys = (
            "closest",
            "town",
            "graveyard",
            "resurrect",
            "yes",
            "accept",
            "release",
            "cancel",
            "sure",
        )
        mode = "death" if any(k in want for k in death_keys) else "alive"
    want_words = [w for w in want.split() if len(w) > 1]
    out: list[UIHit] = []
    for hit in ocr_text_boxes(path, min_conf=min_conf, mode=mode):
        lab = hit.label
        if want == lab or want in lab or lab in want:
            out.append(hit)
            continue
        if want_words and all(w in lab for w in want_words):
            out.append(hit)
            continue
        if want.startswith("closest") and "closest" in lab:
            out.append(hit)
            continue
        if want in {"town", "city"} and want[:3] in lab:
            out.append(hit)
            continue
        if want == "closest town" and "closest" in lab and "city" not in lab and "cancel" not in lab:
            out.append(hit)
            continue

    def _rank(h: UIHit) -> tuple:
        lab = h.label
        exact = 1 if want == lab else 0
        town = 1 if "town" in lab and "city" not in lab else 0
        return (-exact, -town, -h.conf, abs(0.5 - h.fx))

    out.sort(key=_rank)
    return out


def resolve_click_target(
    path: Path | None,
    memory: UIMemory | None,
    label: str,
) -> tuple[float, float, str] | None:
    """Memory first (successful clicks), else live OCR. Returns fx, fy, source."""
    want = _norm_label(label)
    if memory is not None:
        memory.load()
        best: tuple[float, float, int] | None = None
        for k, row in memory.labels.items():
            if want == k or want in k or k in want:
                succ = int(row.get("successes", 0))
                cand = (float(row["fx"]), float(row["fy"]), succ)
                if best is None or cand[2] > best[2]:
                    best = cand
        if best is not None and best[2] > 0:
            return best[0], best[1], "memory_success"
    if path is not None and path.exists():
        # Rez picker: click the actual "Closest Town" word when ROI OCR sees it.
        if want.startswith("closest") or want in {"town", "city"}:
            search = "closest city" if "city" in want and "town" not in want else "closest town"
            hits = find_label_on_frame(path, search)
            if not hits and want.startswith("closest"):
                hits = find_label_on_frame(path, label)
            for h in hits:
                if "city" in h.label or "cancel" in h.label:
                    continue
                if "town" in h.label or "closest" in h.label:
                    return h.fx, h.fy, "ocr_roi"
            if hits:
                return hits[0].fx, hits[0].fy, "ocr_roi"
            for hit in ocr_text_boxes(path, min_conf=25, mode="death"):
                if "choose where" in hit.label:
                    return max(0.35, hit.fx - 0.08), min(0.95, hit.fy + 0.03), "ocr_rez_button"
        if want in {"yes", "accept", "ok"}:
            hits = find_label_on_frame(path, "yes", mode="death")
            if hits:
                return hits[0].fx, hits[0].fy, "ocr_roi"
            for hit in ocr_text_boxes(path, min_conf=25, mode="death"):
                if "are you sure" in hit.label or "want to return" in hit.label:
                    return hit.fx - 0.03, min(0.95, hit.fy + 0.03), "ocr_below_prompt"
        hits = find_label_on_frame(path, label)
        if hits:
            return hits[0].fx, hits[0].fy, "ocr_roi"
    if memory is not None:
        hit = memory.lookup(label)
        if hit:
            return hit[0], hit[1], "memory"
    return None


def explore_click_candidates(
    path: Path | None,
    memory: UIMemory | None,
    wants: list[str],
    *,
    ban: str | None = None,
) -> list[str]:
    """Build discoverable actions for wanted labels (no hardcoded fracs)."""
    acts: list[str] = []
    seen: set[str] = set()
    ban_l = (ban or "").lower()

    def _add(a: str) -> None:
        if a.lower() == ban_l:
            return
        if "cancel" in a.lower() and "cancel" not in " ".join(wants).lower():
            return
        if a not in seen:
            seen.add(a)
            acts.append(a)

    for want in wants:
        _add(f"click_label:{want}")
        if path is not None and path.exists():
            for hit in find_label_on_frame(path, want)[:3]:
                _add(f"click:{hit.fx:.3f},{hit.fy:.3f}")
                for dy, dx in (
                    (0.0, 0.0),
                    (0.01, 0.0),
                    (-0.01, 0.0),
                    (0.0, -0.02),
                    (0.015, -0.02),
                ):
                    _add(
                        f"click:{max(0.05, min(0.95, hit.fx + dx)):.3f},"
                        f"{max(0.05, min(0.95, hit.fy + dy)):.3f}"
                    )
        if memory is not None:
            hit = memory.lookup(want)
            if hit:
                _add(f"click:{hit[0]:.3f},{hit[1]:.3f}")
    if path is not None and path.exists():
        for hit in ocr_text_boxes(path, min_conf=25, mode="death"):
            lab = hit.label
            if "choose where" in lab:
                # Left / mid / right button slots under the title
                for dx in (-0.08, -0.05, 0.0, 0.05):
                    _add(
                        f"click:{max(0.05, min(0.95, hit.fx + dx)):.3f},"
                        f"{max(0.05, min(0.95, hit.fy + 0.025)):.3f}"
                    )
            if "are you sure" in lab or "want to return" in lab:
                for dy, dx in ((0.03, -0.03), (0.04, 0.0), (0.05, -0.06), (0.02, 0.05)):
                    _add(
                        f"click:{max(0.05, min(0.95, hit.fx + dx)):.3f},"
                        f"{max(0.05, min(0.95, hit.fy + dy)):.3f}"
                    )
    return acts


def random_ui_probe(
    path: Path | None,
    wants: list[str],
    *,
    tried: set[str] | None = None,
) -> str | None:
    """Pick a fresh spatial click near death/confirm UI for try→measure→remember."""
    import random

    pool = explore_click_candidates(path, None, wants)
    spatial = [a for a in pool if a.startswith("click:")]
    if not spatial:
        return None
    tried = tried or set()
    fresh = [a for a in spatial if a not in tried]
    pick_from = fresh or spatial
    return random.choice(pick_from)


def discover_and_remember(
    path: Path,
    memory: UIMemory,
    *,
    mode: str = "full",
) -> list[UIHit]:
    """Scan frame, remember new/updated labels, return visible hits."""
    hits = ocr_text_boxes(path, mode=mode)
    interesting = []
    keywords = (
        "accept",
        "yes",
        "ok",
        "release",
        "spirit",
        "graveyard",
        "resurrect",
        "cancel",
        "close",
        "options",
        "logout",
        "macros",
        "addons",
        "continue",
        "quest",
        "loot",
        "vendor",
        "train",
        "buy",
        "sell",
        "abandon",
        "discord",
        "bindings",
        "closest",
        "town",
        "city",
        "sure",
    )
    for hit in hits:
        if any(k in hit.label for k in keywords) or len(hit.label.split()) >= 2:
            memory.remember(hit.label, hit.fx, hit.fy, source="ocr")
            interesting.append(hit)
    return interesting


def parse_dynamic_action(action: str) -> dict[str, Any] | None:
    """Backward-compatible re-export — full parser lives in ability_memory."""
    from playmind.ability_memory import parse_dynamic_action as _parse

    return _parse(action)


# No hardcoded death/button coords — agent discovers via OCR + remembers successes.
DEFAULT_SEEDS: dict[str, dict[str, Any]] = {}


def ensure_seeded(memory: UIMemory) -> None:
    memory.load()
    changed = False
    for k, v in DEFAULT_SEEDS.items():
        if k not in memory.labels:
            memory.labels[k] = dict(v)
            changed = True
    if changed:
        memory.save()
