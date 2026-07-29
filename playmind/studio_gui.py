"""Stdlib-only HTTP server for the offline PlayMind Studio.

This entrypoint is deliberately separate from :mod:`playmind.owned_gui`.
It has no live capture, process inspection, planner runtime, or actuator
imports. Open http://127.0.0.1:8787 after starting it.
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from playmind.gui.studio_dashboard import (
    INDEX_HTML,
    StudioGuiState,
    annotation_categories,
    studio_doctor,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
STATE = StudioGuiState()


class Handler(BaseHTTPRequestHandler):
    server_version = "PlayMindStudio/1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _json(self, status: int, value: Any) -> None:
        body = json.dumps(value, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self) -> None:
        body = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length > 2_000_000:
            raise ValueError("request body is too large")
        if length == 0:
            return {}
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON request body must be an object")
        return value

    def _error(self, exc: Exception) -> None:
        if isinstance(exc, PermissionError):
            status = 403
        elif isinstance(exc, RuntimeError) and "already running" in str(exc):
            status = 409
        elif isinstance(exc, (ValueError, KeyError, FileNotFoundError)):
            status = 400
        else:
            status = 500
        STATE.add_alert(f"{type(exc).__name__}: {exc}")
        self._json(
            status,
            {
                "ok": False,
                "error": str(exc),
                "type": type(exc).__name__,
                "offline_only": True,
            },
        )

    @staticmethod
    def _block_live_request(path: str, options: dict[str, Any]) -> None:
        blocked_paths = (
            "/api/live",
            "/api/capture",
            "/api/input",
            "/api/actuator",
            "/api/planner/live",
        )
        if path.startswith(blocked_paths) or any(
            bool(options.get(name))
            for name in ("live", "send_input", "capture_live", "generated_input")
        ):
            raise PermissionError(
                "retail_wow_offline_only prohibits all live controls"
            )

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/index.html"}:
                self._html()
                return
            if parsed.path == "/api/status":
                self._json(200, STATE.status())
                return
            if parsed.path == "/api/projects":
                self._json(
                    200,
                    {
                        "projects": STATE.app.list_projects(),
                        "current_project_id": STATE.app.current_project_id,
                    },
                )
                return
            if parsed.path == "/api/annotations":
                rows = (
                    [
                        item.to_dict()
                        for item in STATE.app.annotations().list()
                    ]
                    if STATE.app.current_project_id
                    else []
                )
                self._json(200, {"annotations": rows})
                return
            if parsed.path == "/api/annotation/categories":
                self._json(
                    200, {"categories": sorted(annotation_categories())}
                )
                return
            if parsed.path == "/api/analysis":
                rows = (
                    STATE.app.store.load_analysis(
                        STATE.app.current_project_id
                    )
                    if STATE.app.current_project_id
                    else []
                )
                self._json(200, {"analysis": rows})
                return
            if parsed.path == "/api/datasets":
                self._json(200, STATE.datasets())
                return
            if parsed.path == "/api/benchmarks":
                self._json(200, {"benchmarks": STATE.benchmarks()})
                return
            if parsed.path == "/api/readiness":
                self._json(200, STATE.app.readiness())
                return
            if parsed.path in {"/api/evaluations", "/api/evaluate"}:
                self._json(200, STATE.app.evaluations())
                return
            if parsed.path == "/api/training":
                self._json(200, STATE.training_status())
                return
            if parsed.path == "/api/learning_proof":
                self._json(200, STATE.learning_proof())
                return
            if parsed.path in {"/api/models", "/api/registry/models"}:
                self._json(200, {"models": STATE.models()})
                return
            if parsed.path == "/api/corrections":
                self._json(200, {"corrections": STATE.corrections()})
                return
            if parsed.path == "/api/doctor":
                self._json(200, studio_doctor(STATE))
                return
            if parsed.path == "/api/alerts":
                self._json(200, {"alerts": STATE.alerts[-50:]})
                return
            if parsed.path == "/api/jobs":
                self._json(
                    200,
                    {
                        "jobs": {
                            name: job.snapshot()
                            for name, job in STATE.jobs.items()
                        }
                    },
                )
                return
            if parsed.path == "/api/job/log":
                name = (parse_qs(parsed.query).get("name") or [""])[0]
                job = STATE.jobs.get(name)
                if job is None:
                    raise KeyError(f"unknown job: {name!r}")
                text = (
                    job.log_path.read_text(encoding="utf-8", errors="replace")
                    if job.log_path.exists()
                    else ""
                )
                self._json(200, {"name": name, "log": text[-100_000:]})
                return
            self._json(404, {"error": "not_found"})
        except Exception as exc:  # noqa: BLE001
            self._error(exc)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            options = self._read_json()
            self._block_live_request(parsed.path, options)
            if parsed.path in {"/api/wizard", "/api/wizard/complete"}:
                if parsed.path.endswith("/complete"):
                    options["completed"] = True
                self._json(200, STATE.update_wizard(options))
                return
            if parsed.path in {
                "/api/projects/select",
                "/api/project/select",
            }:
                project = STATE.app.select_project(
                    str(options.get("project_id") or "")
                )
                self._json(200, {"ok": True, "project": project})
                return
            if parsed.path in {
                "/api/import",
                "/api/video/import",
                "/api/projects/import",
            }:
                provenance = options.get("provenance")
                if not isinstance(provenance, dict):
                    raise ValueError("provenance is required")
                if not (
                    provenance.get("rights_confirmed")
                    or provenance.get("permission_confirmed")
                    or provenance.get("training_use_allowed")
                ):
                    raise ValueError(
                        "recording rights or permission must be confirmed"
                    )
                source = options.get("path") or options.get("source")
                if not source:
                    raise ValueError("local video path is required")
                project = STATE.app.import_video(
                    str(source),
                    provenance=provenance,
                    name=options.get("name"),
                    project_id=options.get("project_id"),
                    mode=str(options.get("mode") or "copy"),
                    profile=STATE.profile.name,
                )
                self._json(201, {"ok": True, "project": project})
                return
            if parsed.path in {"/api/extract", "/api/frames/extract"}:
                result = STATE.app.extract_frames(
                    strategy=str(options.get("strategy") or "overview")
                )
                self._json(200, {"ok": True, "result": result})
                return
            if parsed.path in {"/api/analyze", "/api/projects/analyze"}:
                if options.get("extract_frames"):
                    STATE.app.extract_frames(
                        strategy=str(options.get("strategy") or "overview")
                    )
                rows = STATE.app.analyze(
                    do_ocr=bool(options.get("do_ocr", False))
                )
                self._json(200, {"ok": True, "analysis": rows})
                return
            if parsed.path == "/api/annotations":
                item = STATE.app.add_annotation(options)
                self._json(201, {"ok": True, "annotation": item.to_dict()})
                return
            if parsed.path == "/api/annotations/update":
                segment_id = str(options.pop("segment_id", ""))
                item = STATE.app.annotations().update(segment_id, **options)
                self._json(200, {"ok": True, "annotation": item.to_dict()})
                return
            if parsed.path == "/api/annotations/review":
                item = STATE.app.annotations().review(
                    str(options.get("segment_id") or ""),
                    accepted=bool(options.get("accepted", True)),
                )
                self._json(200, {"ok": True, "annotation": item.to_dict()})
                return
            if parsed.path == "/api/annotations/delete":
                STATE.app.annotations().remove(
                    str(options.get("segment_id") or "")
                )
                self._json(200, {"ok": True})
                return
            if parsed.path == "/api/annotations/undo":
                rows = [
                    item.to_dict()
                    for item in STATE.app.annotations().undo()
                ]
                self._json(200, {"ok": True, "annotations": rows})
                return
            if parsed.path in {
                "/api/datasets/export",
                "/api/export/datasets",
            }:
                result = STATE.app.export_datasets(
                    project_ids=options.get("project_ids"),
                    seed=int(options.get("seed", 0)),
                    allow_unverified_private=bool(
                        options.get("allow_unverified_private", False)
                    ),
                )
                self._json(200, {"ok": True, "result": result})
                return
            if parsed.path in {
                "/api/benchmarks/freeze",
                "/api/benchmark/freeze",
            }:
                scenarios = options.get("scenarios")
                if not isinstance(scenarios, list):
                    raise ValueError("scenarios must be a JSON list")
                result = STATE.benchmark_root.mkdir(
                    parents=True, exist_ok=True
                )
                del result
                from playmind.gui.studio_dashboard import BenchmarkBuilder

                frozen = BenchmarkBuilder(STATE.benchmark_root).freeze(
                    scenarios,
                    benchmark_id=str(
                        options.get("benchmark_id")
                        or "studio_real_benchmark"
                    ),
                    tier=str(options.get("tier") or "frozen_real"),
                    version=options.get("version"),
                    required_categories=options.get(
                        "required_categories", ()
                    ),
                )
                self._json(201, {"ok": True, "benchmark": frozen})
                return
            if parsed.path == "/api/readiness":
                project_ids = options.pop("project_ids", None)
                self._json(
                    200,
                    STATE.app.readiness(
                        project_ids=project_ids, **options
                    ),
                )
                return
            if parsed.path in {
                "/api/training/smoke",
                "/api/training/start",
            }:
                self._json(202, STATE.start_smoke_training())
                return
            if parsed.path in {"/api/evaluate", "/api/evaluation/start"}:
                self._json(202, STATE.start_evaluation(options))
                return
            if parsed.path == "/api/corrections":
                if not STATE.app.current_project_id:
                    raise RuntimeError("select a Studio project first")
                from playmind.gui.studio_dashboard import (
                    CorrectionStore,
                    PlanCorrection,
                )

                item = CorrectionStore(
                    STATE.app.current_project_id, STATE.projects_root
                ).add(
                    PlanCorrection(
                        project_id=STATE.app.current_project_id,
                        planner_state=dict(options.get("planner_state") or {}),
                        candidate_plan=dict(
                            options.get("candidate_plan") or {}
                        ),
                        corrected_plan=dict(
                            options.get("corrected_plan") or {}
                        ),
                        timestamp=options.get("timestamp"),
                        notes=str(options.get("notes") or ""),
                    )
                )
                self._json(201, {"ok": True, "correction": item.to_dict()})
                return
            if parsed.path == "/api/corrections/review":
                if not STATE.app.current_project_id:
                    raise RuntimeError("select a Studio project first")
                from playmind.gui.studio_dashboard import CorrectionStore

                item = CorrectionStore(
                    STATE.app.current_project_id, STATE.projects_root
                ).review(
                    str(options.get("correction_id") or ""),
                    accepted=bool(options.get("accepted", True)),
                )
                self._json(200, {"ok": True, "correction": item.to_dict()})
                return
            if parsed.path == "/api/review/focus":
                STATE.review_focused = bool(options.get("focused", False))
                STATE.selected_annotation_id = (
                    str(options["segment_id"])
                    if options.get("segment_id")
                    else None
                )
                self._json(
                    200,
                    {
                        "ok": True,
                        "focused": STATE.review_focused,
                        "segment_id": STATE.selected_annotation_id,
                    },
                )
                return
            self._json(404, {"error": "not_found"})
        except Exception as exc:  # noqa: BLE001
            self._error(exc)

    def do_PUT(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        prefix = "/api/annotations/"
        if not parsed.path.startswith(prefix):
            self._json(404, {"error": "not_found"})
            return
        try:
            changes = self._read_json()
            segment_id = parsed.path[len(prefix) :]
            item = STATE.app.annotations().update(segment_id, **changes)
            self._json(200, {"ok": True, "annotation": item.to_dict()})
        except Exception as exc:  # noqa: BLE001
            self._error(exc)

    do_PATCH = do_PUT

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        prefix = "/api/annotations/"
        if not parsed.path.startswith(prefix):
            self._json(404, {"error": "not_found"})
            return
        try:
            STATE.app.annotations().remove(parsed.path[len(prefix) :])
            self._json(200, {"ok": True})
        except Exception as exc:  # noqa: BLE001
            self._error(exc)


def _start_annotation_hotkeys() -> None:
    """Start optional F7/F8 review keys, gated by explicit browser focus."""

    try:
        from pynput import keyboard  # type: ignore
    except Exception:  # noqa: BLE001
        STATE.hotkey_note = (
            "Annotation hotkeys unavailable; use Accept/Reject buttons. "
            "If pynput is installed, F7/F8 only work while review is focused."
        )
        return

    def on_press(key: Any) -> None:
        if not STATE.review_focused or not STATE.selected_annotation_id:
            return
        try:
            if key == keyboard.Key.f7:
                STATE.app.annotations().review(
                    STATE.selected_annotation_id, accepted=True
                )
            elif key == keyboard.Key.f8:
                STATE.app.annotations().review(
                    STATE.selected_annotation_id, accepted=False
                )
        except Exception as exc:  # noqa: BLE001
            STATE.add_alert(f"Annotation hotkey failed: {exc}")

    def listen() -> None:
        try:
            with keyboard.Listener(on_press=on_press) as listener:
                STATE.hotkey_note = (
                    "F7 accept / F8 reject. Hotkeys are active ONLY while "
                    "the Studio annotation review panel has focus."
                )
                listener.join()
        except Exception as exc:  # noqa: BLE001
            STATE.hotkey_note = (
                f"Annotation hotkeys unavailable ({exc}); use UI buttons."
            )

    threading.Thread(
        target=listen, name="playmind-studio-review-hotkeys", daemon=True
    ).start()


def main(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    *,
    config_path: str | None = None,
) -> None:
    global STATE
    if config_path:
        STATE = StudioGuiState(config_path)
    _start_annotation_hotkeys()
    server = ThreadingHTTPServer((host, int(port)), Handler)
    url = f"http://{host}:{int(port)}/"
    print(f"PlayMind Studio (offline-only) at {url}")
    print(f"  {STATE.hotkey_note}")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down PlayMind Studio.")
    finally:
        server.server_close()


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    main(
        args.host,
        args.port,
        not args.no_browser,
        config_path=args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "Handler",
    "INDEX_HTML",
    "STATE",
    "StudioGuiState",
    "main",
]
