"""Browser GUI with live logging for PlayMind demo runs.

Stdlib only: python -m playmind.web_gui
Then open http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from playmind.actuators import DemoActuator, DryRunKeyboardActuator
from playmind.agent import AgentConfig, PlayMindAgent
from playmind.demo_world import ACTIONS, DemoWorld


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PlayMind Monitor</title>
  <style>
    :root {
      --bg: #12161c;
      --panel: #1b222c;
      --text: #e7eef7;
      --muted: #9aabbd;
      --accent: #3dd6c6;
      --warn: #f0b429;
      --danger: #ff6b6b;
      --ok: #6ddf8e;
      --border: #2c3644;
      --mono: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
      --sans: "Segoe UI", system-ui, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; background:
        radial-gradient(1200px 600px at 10% -10%, #1d3a3a 0%, transparent 50%),
        radial-gradient(900px 500px at 100% 0%, #243049 0%, transparent 45%),
        var(--bg);
      color: var(--text); font-family: var(--sans);
      min-height: 100vh;
    }
    header {
      display: flex; justify-content: space-between; align-items: center;
      padding: 1rem 1.25rem; border-bottom: 1px solid var(--border);
      backdrop-filter: blur(8px); position: sticky; top: 0; background: rgba(18,22,28,.85);
    }
    h1 { font-size: 1.15rem; margin: 0; letter-spacing: .04em; }
    h1 span { color: var(--accent); }
    .badge {
      font-size: .75rem; color: var(--muted); border: 1px solid var(--border);
      padding: .25rem .55rem; border-radius: 999px;
    }
    main {
      display: grid; grid-template-columns: 1.1fr 1fr; gap: 1rem;
      padding: 1rem; max-width: 1200px; margin: 0 auto;
    }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
    .card {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 14px; padding: 1rem; min-height: 280px;
    }
    .card h2 {
      margin: 0 0 .75rem; font-size: .85rem; color: var(--muted);
      text-transform: uppercase; letter-spacing: .08em;
    }
    .controls { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: .75rem; }
    button, select, input {
      background: #10151c; color: var(--text); border: 1px solid var(--border);
      border-radius: 10px; padding: .55rem .8rem; font: inherit;
    }
    button {
      cursor: pointer; background: linear-gradient(180deg, #243140, #1a222d);
    }
    button.primary { border-color: #2a6b63; background: linear-gradient(180deg, #1f4f4a, #173832); }
    button.danger { border-color: #7a3030; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    #map {
      font-family: var(--mono); white-space: pre; line-height: 1.25;
      background: #0d1117; border-radius: 10px; padding: .75rem;
      border: 1px solid var(--border); min-height: 180px;
    }
    #stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: .5rem; margin-top: .75rem; }
    .stat {
      background: #121821; border: 1px solid var(--border); border-radius: 10px; padding: .55rem .7rem;
    }
    .stat .k { color: var(--muted); font-size: .72rem; text-transform: uppercase; }
    .stat .v { font-family: var(--mono); margin-top: .15rem; }
    #log {
      font-family: var(--mono); font-size: .78rem; height: 420px; overflow: auto;
      background: #0d1117; border-radius: 10px; padding: .75rem; border: 1px solid var(--border);
    }
    .log-line { padding: .15rem 0; border-bottom: 1px solid rgba(44,54,68,.45); }
    .log-line .t { color: var(--muted); }
    .log-line .a { color: var(--accent); }
    .log-line .ok { color: var(--ok); }
    .log-line .warn { color: var(--warn); }
    .row { display: flex; gap: .5rem; flex-wrap: wrap; align-items: center; margin-top: .5rem; }
    label { color: var(--muted); font-size: .85rem; }
  </style>
</head>
<body>
  <header>
    <h1><span>PlayMind</span> Monitor</h1>
    <div class="badge" id="conn">connecting…</div>
  </header>
  <main>
    <section class="card">
      <h2>Controls</h2>
      <div class="controls">
        <button class="primary" id="btnStart">Start episode</button>
        <button class="danger" id="btnStop" disabled>Stop</button>
        <button id="btnClear">Clear log</button>
      </div>
      <div class="row">
        <label>Speed <input id="delay" type="number" min="0" step="50" value="120" style="width:6rem" /> ms</label>
        <label>Vision <input id="vision" type="checkbox" checked /></label>
        <label>Dry-run keys <input id="dryrun" type="checkbox" checked /></label>
      </div>
      <div class="row">
        <label>Directive
          <input id="directive" placeholder="farm / stop / turn in" style="width:14rem" />
        </label>
        <button id="btnDir">Set</button>
      </div>
      <h2 style="margin-top:1rem">World</h2>
      <div id="map">Waiting to start…</div>
      <div id="stats">
        <div class="stat"><div class="k">Action</div><div class="v" id="sAction">—</div></div>
        <div class="stat"><div class="k">Reward</div><div class="v" id="sReward">—</div></div>
        <div class="stat"><div class="k">Kills</div><div class="v" id="sKills">—</div></div>
        <div class="stat"><div class="k">HP</div><div class="v" id="sHp">—</div></div>
        <div class="stat"><div class="k">Steps</div><div class="v" id="sSteps">—</div></div>
        <div class="stat"><div class="k">Status</div><div class="v" id="sStatus">idle</div></div>
      </div>
    </section>
    <section class="card">
      <h2>Live log</h2>
      <div id="log"></div>
    </section>
  </main>
  <script>
    const logEl = document.getElementById('log');
    let lastLogLen = 0;

    function addLocal(msg, cls='') {
      const d = document.createElement('div');
      d.className = 'log-line';
      d.innerHTML = `<span class="t">${new Date().toLocaleTimeString()}</span> <span class="${cls}">${msg}</span>`;
      logEl.appendChild(d);
      logEl.scrollTop = logEl.scrollHeight;
    }

    async function api(path, opts) {
      const r = await fetch(path, opts);
      return r.json();
    }

    document.getElementById('btnStart').onclick = async () => {
      const body = {
        delay_ms: Number(document.getElementById('delay').value || 120),
        vision: document.getElementById('vision').checked,
        dry_run: document.getElementById('dryrun').checked,
        directive: document.getElementById('directive').value,
      };
      const res = await api('/api/start', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
      addLocal(res.message || 'started', res.ok ? 'ok' : 'warn');
    };
    document.getElementById('btnStop').onclick = async () => {
      const res = await api('/api/stop', {method:'POST'});
      addLocal(res.message || 'stop', 'warn');
    };
    document.getElementById('btnClear').onclick = async () => {
      await api('/api/clear', {method:'POST'});
      logEl.innerHTML = '';
      lastLogLen = 0;
    };
    document.getElementById('btnDir').onclick = async () => {
      const directive = document.getElementById('directive').value;
      const res = await api('/api/directive', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({directive})});
      addLocal('directive: ' + (res.directive || '(cleared)'), 'a');
    };

    async function poll() {
      try {
        const s = await api('/api/state');
        document.getElementById('conn').textContent = s.running ? 'running' : 'idle';
        document.getElementById('btnStart').disabled = !!s.running;
        document.getElementById('btnStop').disabled = !s.running;
        document.getElementById('map').textContent = s.map || '—';
        document.getElementById('sAction').textContent = s.action || '—';
        document.getElementById('sReward').textContent = s.reward != null ? Number(s.reward).toFixed(2) : '—';
        document.getElementById('sKills').textContent = s.kills != null ? s.kills : '—';
        document.getElementById('sHp').textContent = s.hp != null ? s.hp : '—';
        document.getElementById('sSteps').textContent = s.steps != null ? s.steps : '—';
        document.getElementById('sStatus').textContent = s.status || 'idle';
        if (s.logs && s.logs.length > lastLogLen) {
          for (let i = lastLogLen; i < s.logs.length; i++) {
            const line = s.logs[i];
            const d = document.createElement('div');
            d.className = 'log-line';
            d.innerHTML = `<span class="t">${line.t}</span> <span class="a">${line.msg}</span>`;
            logEl.appendChild(d);
          }
          lastLogLen = s.logs.length;
          logEl.scrollTop = logEl.scrollHeight;
        }
      } catch (e) {
        document.getElementById('conn').textContent = 'disconnected';
      }
      setTimeout(poll, 250);
    }
    poll();
  </script>
</body>
</html>
"""


@dataclass
class GuiState:
    running: bool = False
    stop_flag: bool = False
    status: str = "idle"
    map: str = ""
    action: str | None = None
    reward: float | None = None
    kills: str | None = None
    hp: float | None = None
    steps: int | None = None
    logs: deque[dict[str, str]] = field(default_factory=lambda: deque(maxlen=500))
    lock: threading.Lock = field(default_factory=threading.Lock)
    worker: threading.Thread | None = None
    agent: PlayMindAgent | None = None

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        with self.lock:
            self.logs.append({"t": ts, "msg": msg})

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "status": self.status,
                "map": self.map,
                "action": self.action,
                "reward": self.reward,
                "kills": self.kills,
                "hp": self.hp,
                "steps": self.steps,
                "logs": list(self.logs),
            }


STATE = GuiState()


def _run_episode(opts: dict[str, Any]) -> None:
    delay = max(0, int(opts.get("delay_ms", 120))) / 1000.0
    vision = bool(opts.get("vision", True))
    dry_run = bool(opts.get("dry_run", True))
    directive = (opts.get("directive") or "").strip()

    cfg = AgentConfig(
        learn=True,
        use_vision=vision,
        data_dir=Path("data/playmind/gui"),
        vision_frame_path=Path("data/playmind/gui/frames/latest.txt"),
    )
    actuator = DryRunKeyboardActuator(
        log_path=Path("data/playmind/gui/actuator_dryrun.jsonl")
    ) if dry_run else DemoActuator()
    agent = PlayMindAgent(world=DemoWorld(), config=cfg, actuator=actuator)
    if directive:
        agent.set_directive(directive)

    with STATE.lock:
        STATE.agent = agent
        STATE.running = True
        STATE.stop_flag = False
        STATE.status = "running"

    STATE.log(f"Episode start vision={vision} dry_run={dry_run} directive={directive or '-'}")
    try:
        for i in range(120):
            if STATE.stop_flag:
                STATE.log("Stopped by user")
                break
            result = agent.tick()
            obs = result["obs"]
            with STATE.lock:
                STATE.map = agent.world.render_ascii()
                STATE.action = result["action"]
                STATE.reward = float(result["reward"])
                STATE.kills = f"{obs.get('quest_kills', 0)}/{obs.get('quest_kills_needed', 0)}"
                STATE.hp = obs.get("player", {}).get("hp")
                STATE.steps = obs.get("steps")
                STATE.status = "quest_complete" if result["done"] else "running"
            vision_q = ""
            if agent.last_vision and agent.last_vision.quest_text:
                vision_q = f" vision_quest='{agent.last_vision.quest_text}'"
            STATE.log(
                f"step={obs.get('steps')} action={result['action']} "
                f"reward={result['reward']:.2f} kills={STATE.kills} hp={STATE.hp}{vision_q}"
            )
            if result["done"]:
                STATE.log("QUEST COMPLETE")
                break
            time.sleep(delay)
        else:
            STATE.log("Episode timeout")
        agent.save()
        STATE.log(f"Saved learning artifacts under {cfg.data_dir}")
    except Exception as exc:  # noqa: BLE001
        STATE.log(f"ERROR: {exc}")
        with STATE.lock:
            STATE.status = "error"
    finally:
        with STATE.lock:
            STATE.running = False
            if STATE.status == "running":
                STATE.status = "idle"


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _html(self, body: str) -> None:
        raw = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._html(INDEX_HTML)
            return
        if path == "/api/state":
            self._json(200, STATE.snapshot())
            return
        self._json(404, {"ok": False, "message": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            body = {}

        if path == "/api/start":
            if STATE.running:
                self._json(200, {"ok": False, "message": "already running"})
                return
            t = threading.Thread(target=_run_episode, args=(body,), daemon=True)
            STATE.worker = t
            t.start()
            self._json(200, {"ok": True, "message": "episode started"})
            return
        if path == "/api/stop":
            STATE.stop_flag = True
            self._json(200, {"ok": True, "message": "stop requested"})
            return
        if path == "/api/clear":
            with STATE.lock:
                STATE.logs.clear()
            self._json(200, {"ok": True, "message": "log cleared"})
            return
        if path == "/api/directive":
            directive = (body.get("directive") or "").strip()
            with STATE.lock:
                if STATE.agent is not None:
                    STATE.agent.set_directive(directive)
            STATE.log(f"directive set to '{directive or '-'}'")
            self._json(200, {"ok": True, "directive": directive})
            return
        self._json(404, {"ok": False, "message": "not found"})


def main() -> None:
    host = "127.0.0.1"
    port = 8765
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"PlayMind GUI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
