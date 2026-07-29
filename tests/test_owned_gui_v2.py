"""HTTP smoke tests for Learning V2 owned GUI endpoints."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from threading import Thread

from http.server import ThreadingHTTPServer

from playmind import owned_gui
from playmind.owned_gui import STATE, Handler, DEFAULT_LEARNING_V2


def _reset_state(tmp: Path) -> None:
    with STATE.lock:
        STATE.events.clear()
        STATE.next_id = 1
        STATE.running = False
        STATE.stop_flag = False
        STATE.thread = None
        STATE.loop = None
        STATE.last_status = {}
        STATE.demo_recorder = None
        STATE.demo_meta = {}
        STATE.learning_v2 = dict(DEFAULT_LEARNING_V2)
        STATE.data_dir = tmp


def _server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


def _post(port: int, path: str, body: dict | None = None) -> dict:
    import urllib.request

    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(port: int, path: str) -> dict:
    import urllib.request

    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_v2_config_and_events_smoke() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _reset_state(tmp)
        server, port = _server()
        try:
            events = _get(port, "/api/events?after=0")
            assert "running" in events
            assert "demo" in events
            assert "learning_v2" in events
            assert events["demo"]["recording"] is False

            cfg = _post(
                port,
                "/api/v2/config",
                {
                    "policy_mode": "behavior_clone",
                    "bc_checkpoint": "data/models/bc.pt",
                    "enabled": True,
                },
            )
            assert cfg["ok"] is True
            assert cfg["learning_v2"]["policy_mode"] == "behavior_clone"
            assert cfg["learning_v2"]["bc_checkpoint"] == "data/models/bc.pt"

            got = _get(port, "/api/v2/config")
            assert got["learning_v2"]["policy_mode"] == "behavior_clone"
        finally:
            server.shutdown()


def test_v2_config_rejects_bad_mode() -> None:
    with tempfile.TemporaryDirectory() as td:
        _reset_state(Path(td))
        server, port = _server()
        try:
            import urllib.error
            import urllib.request

            data = json.dumps({"policy_mode": "nope"}).encode("utf-8")
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/v2/config",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urllib.request.urlopen(req, timeout=5)
                raised = False
            except urllib.error.HTTPError as exc:
                raised = True
                assert exc.code == 400
                body = json.loads(exc.read().decode("utf-8"))
                assert "error" in body
            assert raised
        finally:
            server.shutdown()


def test_demo_start_stop_mark() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _reset_state(tmp)
        # Point recorder root into tmp via monkeypatch after start uses default;
        # override by constructing recorder with custom root through API start name.
        server, port = _server()
        try:
            # Redirect DemonstrationRecorder default by setting STATE recorder root
            from playmind.demonstrations import DemonstrationRecorder

            STATE.demo_recorder = DemonstrationRecorder(root=tmp / "demos")

            start = _post(
                port,
                "/api/demo/start",
                {"name": "smoke-demo", "goal": "farm", "profile": "test", "notes": "n1"},
            )
            assert start["ok"] is True
            assert start["recording"] is True
            assert start["session_id"]

            # Append one sample via internal helper
            owned_gui._maybe_append_demo(
                {
                    "tick": 1,
                    "action": "key:1",
                    "reward": 0.0,
                    "active_skill": "engage",
                    "episode_id": start.get("episode_id"),
                }
            )
            snap = STATE.demo_snapshot()
            assert snap["sample_count"] >= 1

            mark = _post(port, "/api/demo/mark", {"outcome": "success", "notes": "good"})
            assert mark["ok"] is True
            assert mark["outcome"] == "success"

            stop = _post(port, "/api/demo/stop", {})
            assert stop["ok"] is True
            assert stop["recording"] is False
            assert Path(stop["session_dir"]).is_dir()
        finally:
            server.shutdown()


def test_episode_reset_legacy_q_diagnostics() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _reset_state(tmp)
        server, port = _server()
        try:
            # Create a fake policy.json to clear
            policy = tmp / "policy.json"
            policy.write_text('{"q": {}}', encoding="utf-8")

            reset = _post(port, "/api/episode/reset", {})
            assert reset["ok"] is True

            cleared = _post(port, "/api/legacy_q/clear", {"data_dir": str(tmp)})
            assert cleared["ok"] is True
            assert cleared["cleared"] is True
            assert not policy.exists()
            assert Path(cleared["backup"]).exists()

            STATE.last_status = {"tick": 3, "active_skill": "wait", "confidence": 0.2}
            STATE.push("status", status=STATE.last_status)
            diag = _post(port, "/api/diagnostics/export", {})
            assert diag["ok"] is True
            zpath = Path(diag["path"])
            assert zpath.is_file()
            assert zpath.suffix == ".zip"
        finally:
            server.shutdown()


def test_index_html_contains_v2_controls() -> None:
    with tempfile.TemporaryDirectory() as td:
        _reset_state(Path(td))
        server, port = _server()
        try:
            import urllib.request

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
                html = resp.read().decode("utf-8")
            assert "policyMode" in html
            assert "behavior_clone" in html
            assert "Advanced V2" in html
            assert "btnDemoStart" in html
            assert "F9" in html
            assert "/api/diagnostics/export" in html
        finally:
            server.shutdown()
