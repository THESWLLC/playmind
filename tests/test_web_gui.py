import json
from threading import Thread

from playmind.web_gui import Handler, STATE, _run_episode
from http.server import ThreadingHTTPServer


def test_episode_updates_state_and_logs() -> None:
    with STATE.lock:
        STATE.logs.clear()
        STATE.running = False
        STATE.stop_flag = False
    _run_episode({"delay_ms": 0, "vision": True, "dry_run": True, "directive": ""})
    snap = STATE.snapshot()
    assert snap["running"] is False
    assert snap["steps"] is not None and snap["steps"] > 0
    assert any("QUEST COMPLETE" in x["msg"] or "action=" in x["msg"] for x in snap["logs"])


def test_http_state_endpoint() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        assert "logs" in data
        assert "running" in data
    finally:
        server.shutdown()
