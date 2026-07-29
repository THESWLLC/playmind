"""Draw ROI rectangles + confidence labels onto a frame (Pillow optional)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

RoiBox = Union[Sequence[float], Mapping[str, Any]]


def draw_rois(
    frame_path: Path | str,
    rois: Mapping[str, Any],
    output_path: Path | str,
    *,
    confidences: Mapping[str, Any] | None = None,
    labels: Mapping[str, Any] | None = None,
) -> Optional[Path]:
    """Draw named ROI rectangles onto ``frame_path`` and save to ``output_path``.

    ``rois`` values may be:
    - ``[L, T, R, B]`` / ``(L, T, R, B)``
    - ``{"box": [L, T, R, B], "confidence": 0.9, "label": "..."}``

    Returns the output path on success, or ``None`` when Pillow is unavailable
    (prints a short message and becomes a no-op).
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError:
        print("vision_overlay: Pillow not available; skipping ROI draw")
        return None

    src = Path(frame_path)
    dst = Path(output_path)
    if not src.exists():
        print(f"vision_overlay: frame not found: {src}")
        return None

    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    confidences = confidences or {}
    labels = labels or {}
    colors = _color_cycle()

    for idx, (name, raw) in enumerate(rois.items()):
        box = _normalize_box(raw)
        if box is None:
            continue
        color = colors[idx % len(colors)]
        draw.rectangle(box, outline=color, width=2)

        conf = None
        if isinstance(raw, Mapping) and raw.get("confidence") is not None:
            conf = raw.get("confidence")
        elif name in confidences:
            conf = confidences[name]

        label_text = None
        if isinstance(raw, Mapping) and raw.get("label") is not None:
            label_text = str(raw.get("label"))
        elif name in labels:
            label_text = str(labels[name])
        else:
            label_text = str(name)

        caption = label_text
        if conf is not None:
            try:
                caption = f"{label_text} {float(conf):.2f}"
            except (TypeError, ValueError):
                caption = f"{label_text} {conf}"

        text_xy = (int(box[0]) + 2, max(0, int(box[1]) - 12))
        if font is not None:
            draw.text(text_xy, caption, fill=color, font=font)
        else:
            draw.text(text_xy, caption, fill=color)

    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)
    return dst


def _normalize_box(raw: Any) -> Optional[tuple[int, int, int, int]]:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        if "box" in raw:
            raw = raw["box"]
        elif all(k in raw for k in ("left", "top", "right", "bottom")):
            raw = (raw["left"], raw["top"], raw["right"], raw["bottom"])
        elif all(k in raw for k in ("x1", "y1", "x2", "y2")):
            raw = (raw["x1"], raw["y1"], raw["x2"], raw["y2"])
        else:
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        l, t, r, b = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
    except (TypeError, ValueError):
        return None
    if r < l:
        l, r = r, l
    if b < t:
        t, b = b, t
    return (l, t, r, b)


def _color_cycle() -> list[tuple[int, int, int]]:
    return [
        (0, 220, 120),
        (255, 80, 80),
        (80, 160, 255),
        (255, 200, 40),
        (200, 80, 255),
        (255, 140, 40),
    ]
