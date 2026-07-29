#!/usr/bin/env python3
"""Review / label sensor frames for PlayMind Learning V2.

Modes:
  --list DIR              list frame paths under a directory
  --label                 read one JSON label line from stdin and append
  --serve / --ui          stdlib HTTP labeling UI on port 8788 (default)
  --append JSON           append a single label JSON object (CLI)

Labels append to data/playmind/labels/sensor_labels.jsonl by default.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from playmind.sensor_metrics import (
    DEFAULT_LABELS_PATH,
    LIFE_PHASES,
    append_label_jsonl,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def list_frame_paths(directory: Path | str, *, recursive: bool = True) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []
    paths: list[Path] = []
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in sorted(iterator):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(path)
    return paths


def append_label(
    record: dict[str, Any],
    path: Path | str | None = None,
) -> Path:
    if "frame" not in record and "path" in record:
        record = {**record, "frame": record["path"]}
    return append_label_jsonl(record, path=path)


def _parse_label_payload(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        raise ValueError("empty label JSON")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("label JSON must be an object")
    return data


def _bool_from_form(values: list[str] | None) -> bool:
    if not values:
        return False
    return values[0].lower() in {"1", "true", "on", "yes", "y"}


def _optional_float(values: list[str] | None) -> float | None:
    if not values:
        return None
    text = values[0].strip()
    if text == "":
        return None
    return float(text)


def label_from_form(form: dict[str, list[str]]) -> dict[str, Any]:
    frame = (form.get("frame") or [""])[0].strip()
    life_phase = (form.get("life_phase") or ["unknown"])[0].strip() or "unknown"
    record: dict[str, Any] = {
        "frame": frame,
        "is_dead": _bool_from_form(form.get("is_dead")),
        "is_ghost": _bool_from_form(form.get("is_ghost")),
        "has_target": _bool_from_form(form.get("has_target")),
        "in_combat": _bool_from_form(form.get("in_combat")),
        "modal": _bool_from_form(form.get("modal")),
        "life_phase": life_phase,
    }
    player_hp = _optional_float(form.get("player_hp"))
    target_hp = _optional_float(form.get("target_hp"))
    if player_hp is not None:
        record["player_hp"] = player_hp
    if target_hp is not None:
        record["target_hp"] = target_hp
    return record


def labeling_html(
    frames: list[Path],
    *,
    index: int = 0,
    message: str = "",
    serve_images: bool = False,
) -> str:
    n = len(frames)
    if n == 0:
        body = "<p>No frames found. Pass <code>--frames-dir DIR</code>.</p>"
        preview = ""
        frame_value = ""
        nav = ""
    else:
        index = max(0, min(index, n - 1))
        frame = frames[index]
        frame_value = str(frame)
        if serve_images:
            preview = (
                f'<p><img src="/frame?i={index}" alt="frame" '
                f'style="max-width:100%;max-height:420px;border:1px solid #444"/></p>'
            )
        else:
            preview = (
                f"<p><strong>Frame path</strong> (no image server): "
                f"<code>{_esc(frame_value)}</code></p>"
            )
        nav = (
            f'<p>Frame {index + 1} / {n} '
            f'<a href="/?i={max(0, index - 1)}">prev</a> · '
            f'<a href="/?i={min(n - 1, index + 1)}">next</a></p>'
        )
        body = ""

    phase_opts = "\n".join(
        f'<option value="{_esc(p)}">{_esc(p)}</option>' for p in LIFE_PHASES
    )
    msg = f"<p style='color:green'>{_esc(message)}</p>" if message else ""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>PlayMind sensor labeling</title>
<style>
body {{ font-family: Georgia, serif; margin: 1.5rem; background: #1a1c1e; color: #eee; }}
label {{ display: inline-block; margin-right: 1rem; }}
input[type=number] {{ width: 6rem; }}
button {{ margin-top: 1rem; padding: 0.4rem 1rem; }}
code {{ color: #9fd; }}
a {{ color: #8cf; }}
</style></head><body>
<h1>Sensor frame labeling</h1>
{msg}
{nav}
{preview}
{body}
<form method="POST" action="/label">
  <input type="hidden" name="frame" value="{_esc(frame_value)}"/>
  <input type="hidden" name="i" value="{index}"/>
  <p>
    <label><input type="checkbox" name="is_dead" value="1"/> is_dead</label>
    <label><input type="checkbox" name="is_ghost" value="1"/> is_ghost</label>
    <label><input type="checkbox" name="has_target" value="1"/> has_target</label>
    <label><input type="checkbox" name="in_combat" value="1"/> in_combat</label>
    <label><input type="checkbox" name="modal" value="1"/> modal</label>
  </p>
  <p>
    <label>player_hp <input type="number" name="player_hp" min="0" max="1" step="0.01"/></label>
    <label>target_hp <input type="number" name="target_hp" min="0" max="1" step="0.01"/></label>
    <label>life_phase
      <select name="life_phase">{phase_opts}</select>
    </label>
  </p>
  <button type="submit">Submit label</button>
</form>
</body></html>
"""


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def serve_labeling_ui(
    frames_dir: Path | str,
    *,
    host: str = "127.0.0.1",
    port: int = 8788,
    labels_path: Path | str | None = None,
    serve_images: bool = False,
) -> None:
    frames = list_frame_paths(frames_dir)
    labels_file = Path(labels_path) if labels_path else DEFAULT_LABELS_PATH

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            if parsed.path == "/frame" and serve_images:
                try:
                    i = int((qs.get("i") or ["0"])[0])
                except ValueError:
                    i = 0
                if not frames or i < 0 or i >= len(frames):
                    self.send_error(404)
                    return
                data = frames[i].read_bytes()
                mime = mimetypes.guess_type(frames[i].name)[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path not in {"/", "/index.html"}:
                self.send_error(404)
                return
            try:
                i = int((qs.get("i") or ["0"])[0])
            except ValueError:
                i = 0
            msg = (qs.get("msg") or [""])[0]
            html = labeling_html(
                frames, index=i, message=msg, serve_images=serve_images
            )
            payload = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/label":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length).decode("utf-8")
            form = parse_qs(body, keep_blank_values=True)
            record = label_from_form(form)
            append_label(record, path=labels_file)
            try:
                i = int((form.get("i") or ["0"])[0])
            except ValueError:
                i = 0
            nxt = min(i + 1, max(0, len(frames) - 1))
            self.send_response(303)
            self.send_header(
                "Location", f"/?i={nxt}&msg=saved+{len(frames) and 'ok' or 'empty'}"
            )
            self.end_headers()

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Sensor labeling UI at http://{host}:{port}/  ({len(frames)} frames)")
    print(f"Labels append to {labels_file}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--list",
        metavar="DIR",
        help="List image frame paths under DIR and exit",
    )
    p.add_argument(
        "--label",
        action="store_true",
        help="Read one JSON label object from stdin and append to labels file",
    )
    p.add_argument(
        "--append",
        metavar="JSON",
        help="Append a JSON label object passed on the command line",
    )
    p.add_argument(
        "--serve",
        "--ui",
        action="store_true",
        dest="serve",
        help="Serve labeling UI (default port 8788)",
    )
    p.add_argument(
        "--frames-dir",
        default="data/playmind/frames",
        help="Directory of frames for --serve / --list default",
    )
    p.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS_PATH),
        help="Path to sensor_labels.jsonl",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8788)
    p.add_argument(
        "--serve-images",
        action="store_true",
        help="Serve frame bytes at /frame (otherwise show path text only)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels_path = Path(args.labels)

    if args.list:
        for path in list_frame_paths(args.list):
            print(path)
        return 0

    if args.label:
        raw = sys.stdin.read()
        record = _parse_label_payload(raw)
        out = append_label(record, path=labels_path)
        print(f"appended -> {out}")
        return 0

    if args.append:
        record = _parse_label_payload(args.append)
        out = append_label(record, path=labels_path)
        print(f"appended -> {out}")
        return 0

    if args.serve:
        serve_labeling_ui(
            args.frames_dir,
            host=args.host,
            port=args.port,
            labels_path=labels_path,
            serve_images=args.serve_images,
        )
        return 0

    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
