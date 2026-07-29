"""Live brain monitor for the owned-game loop.

Shows realtime LLM output, OCR, invented abilities, and actions.
Stdlib only: python -m playmind.owned_gui
Then open http://127.0.0.1:8777
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

from playmind.owned_loop import OwnedGameLoop, OwnedLoopConfig


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PlayMind Brain</title>
  <style>
    :root {
      --bg: #0e1218;
      --panel: #171d27;
      --text: #e8eef6;
      --muted: #8fa0b3;
      --accent: #5ce1c5;
      --warn: #f0b429;
      --danger: #ff6b7a;
      --ok: #7dffb3;
      --border: #2a3444;
      --mono: "Cascadia Mono", "IBM Plex Mono", Consolas, monospace;
      --sans: "Segoe UI", system-ui, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; color: var(--text); font-family: var(--sans);
      background:
        radial-gradient(900px 500px at 0% 0%, #1a3a36 0%, transparent 55%),
        radial-gradient(800px 400px at 100% 0%, #24304a 0%, transparent 50%),
        var(--bg);
      min-height: 100vh;
    }
    header {
      display: flex; justify-content: space-between; align-items: center; gap: 1rem;
      padding: .9rem 1.2rem; border-bottom: 1px solid var(--border);
      background: rgba(14,18,24,.9); position: sticky; top: 0; z-index: 5;
    }
    h1 { margin: 0; font-size: 1.1rem; letter-spacing: .03em; }
    h1 span { color: var(--accent); }
    .badge {
      font-size: .72rem; color: var(--muted); border: 1px solid var(--border);
      padding: .2rem .55rem; border-radius: 999px;
    }
    .badge.live { color: var(--ok); border-color: #2d6b4f; }
    main {
      display: grid; grid-template-columns: 1.15fr .85fr; gap: 1rem;
      padding: 1rem; max-width: 1400px; margin: 0 auto;
    }
    @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
    .card {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; padding: 1rem; min-height: 200px;
    }
    .card h2 {
      margin: 0 0 .7rem; font-size: .78rem; color: var(--muted);
      text-transform: uppercase; letter-spacing: .1em;
    }
    .controls { display: flex; flex-wrap: wrap; gap: .45rem; margin-bottom: .75rem; }
    button, input, select {
      background: #10151c; color: var(--text); border: 1px solid var(--border);
      border-radius: 9px; padding: .5rem .75rem; font: inherit;
    }
    button { cursor: pointer; background: linear-gradient(180deg, #243140, #1a222d); }
    button.primary { border-color: #2a6b63; background: linear-gradient(180deg, #1f4f4a, #173832); }
    button.danger { border-color: #7a3030; }
    button:disabled { opacity: .4; cursor: not-allowed; }
    #think {
      font-family: var(--mono); font-size: .85rem; line-height: 1.45;
      background: #0b0f14; border: 1px solid var(--border); border-radius: 10px;
      padding: 1rem; min-height: 220px; white-space: pre-wrap; word-break: break-word;
      color: var(--accent); max-height: 360px; overflow: auto;
    }
    #think.empty { color: var(--muted); }
    #log {
      font-family: var(--mono); font-size: .74rem; height: 520px; overflow: auto;
      background: #0b0f14; border-radius: 10px; padding: .7rem; border: 1px solid var(--border);
    }
    .log-line { padding: .35rem 0; border-bottom: 1px solid rgba(42,52,68,.55); }
    .log-line .t { color: var(--muted); }
    .log-line .a { color: var(--accent); font-weight: 600; }
    .log-line .think { color: #c9d7ea; margin-top: .2rem; }
    .log-line .err { color: var(--danger); }
    .log-line .meta { color: var(--muted); font-size: .7rem; }
    #stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: .45rem; }
    .stat {
      background: #10161f; border: 1px solid var(--border); border-radius: 9px; padding: .45rem .55rem;
    }
    .stat .k { color: var(--muted); font-size: .65rem; text-transform: uppercase; }
    .stat .v { font-family: var(--mono); font-size: .85rem; margin-top: .1rem; }
    #abilities, #ocr {
      font-family: var(--mono); font-size: .75rem; color: var(--muted);
      background: #10161f; border: 1px solid var(--border); border-radius: 9px;
      padding: .55rem; min-height: 48px; white-space: pre-wrap;
    }
    .row { display: flex; gap: .5rem; flex-wrap: wrap; align-items: center; margin-top: .5rem; }
    label { color: var(--muted); font-size: .82rem; }
  </style>
</head>
<body>
  <header>
    <h1><span>PlayMind</span> Brain</h1>
    <div class="badge" id="conn">connecting…</div>
  </header>
  <main>
    <section class="card">
      <h2>Controls</h2>
      <div class="controls">
        <button class="primary" id="btnStart">Start live</button>
        <button class="danger" id="btnStop" disabled>Stop</button>
        <button id="btnClear">Clear log</button>
      </div>
      <div class="row">
        <label>ticks <input id="maxTicks" type="number" value="0" min="0" style="width:5rem" title="0 = run until Stop" /></label>
        <label style="color:var(--muted);font-size:.75rem">0 = infinite</label>
        <label>tick delay <input id="tickSec" type="number" value="0.05" min="0" step="0.05" style="width:4.5rem" /></label>
        <label>vlm every <input id="visionEvery" type="number" value="24" min="1" style="width:3.5rem" title="Call vision LLM every N ticks — higher = much faster ticks" /></label>
        <label>directive <input id="directive" value="farm to level 2" style="width:14rem" title="e.g. farm to level 2 | kill grell | go north | quest" /></label>
        <label><input id="live" type="checkbox" checked /> live keys</label>
        <label><input id="ollama" type="checkbox" checked /> LLM</label>
      </div>
      <h2 style="margin-top:1rem">Latest LLM thought</h2>
      <div id="think" class="empty">(waiting for first tick…)</div>
      <h2 style="margin-top:1rem">Soul</h2>
      <div id="soul" style="font-family:var(--mono);font-size:.82rem;color:var(--accent);line-height:1.4;min-height:2.5rem">(waking…)</div>
      <h2 style="margin-top:1rem">Live stats</h2>
      <div id="stats"></div>
      <h2 style="margin-top:1rem">OCR / UI</h2>
      <div id="ocr"></div>
      <h2 style="margin-top:1rem">Invented abilities</h2>
      <div id="abilities"></div>
    </section>
    <section class="card">
      <h2>Realtime thinking log</h2>
      <div id="log"></div>
    </section>
  </main>
<script>
const logEl = document.getElementById('log');
const thinkEl = document.getElementById('think');
const statsEl = document.getElementById('stats');
const ocrEl = document.getElementById('ocr');
const abilEl = document.getElementById('abilities');
const soulEl = document.getElementById('soul');
const conn = document.getElementById('conn');
let lastId = 0;

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function renderStats(s) {
  const cells = [
    ['tick', s.tick],
    ['body', s.soul_body || (s.is_dead ? 'dead' : 'alive')],
    ['action', s.action],
    ['reward', s.reward],
    ['hp', s.vision_hp],
    ['goal', s.goal],
    ['brain', s.decision || s.brain_mode || s.brain],
    ['teacher', s.teacher_teaches],
    ['taught', s.teacher_last],
    ['stuck', s.stuck],
    ['modal', s.modal_menu],
    ['target', s.has_target],
    ['bar', s.bar_slots],
  ];
  statsEl.innerHTML = cells.map(([k,v]) =>
    `<div class="stat"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`
  ).join('');
  ocrEl.textContent = s.screen_ocr || '(none)';
  const abs = s.ability_summary || (s.abilities || []).join(', ') || '(none yet)';
  abilEl.textContent = abs;
  if (soulEl) soulEl.textContent = s.soul || '(waking…)';
  const thought = s.thinking || s.llm_raw || s.screen_see || '';
  if (thought) {
    thinkEl.classList.remove('empty');
    thinkEl.textContent = thought;
  }
}

function appendLog(ev) {
  const s = ev.status || {};
  const div = document.createElement('div');
  div.className = 'log-line';
  const thought = s.thinking || s.llm_raw || '';
  const err = s.llm_error ? `<div class="err">error: ${esc(s.llm_error)}</div>` : '';
  div.innerHTML = `
    <div><span class="t">#${esc(s.tick)}</span>
      <span class="a">${esc(s.action)}</span>
      <span class="meta">${esc(s.decision || s.brain_mode || '')} · r=${esc(s.reward)} · hp=${esc(s.vision_hp)}</span>
    </div>
    ${thought ? `<div class="think">${esc(thought)}</div>` : ''}
    ${err}
  `;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
  renderStats(s);
}

async function poll() {
  try {
    const r = await fetch('/api/events?after=' + lastId);
    const data = await r.json();
    conn.textContent = data.running ? 'LIVE' : 'idle';
    conn.className = 'badge' + (data.running ? ' live' : '');
    document.getElementById('btnStart').disabled = !!data.running;
    document.getElementById('btnStop').disabled = !data.running;
    for (const ev of data.events || []) {
      lastId = Math.max(lastId, ev.id);
      if (ev.type === 'status') appendLog(ev);
      else if (ev.type === 'info' || ev.type === 'error') {
        const div = document.createElement('div');
        div.className = 'log-line';
        div.innerHTML = `<div class="${ev.type === 'error' ? 'err' : 'meta'}">${esc(ev.message)}</div>`;
        logEl.appendChild(div);
        logEl.scrollTop = logEl.scrollHeight;
      }
    }
  } catch (e) {
    conn.textContent = 'offline';
  }
}

document.getElementById('btnStart').onclick = async () => {
  const body = {
    live: document.getElementById('live').checked,
    ollama: document.getElementById('ollama').checked,
    max_ticks: Number(document.getElementById('maxTicks').value || 0),
    tick_seconds: Number(document.getElementById('tickSec').value || 0.05),
    vision_every: Number(document.getElementById('visionEvery').value || 24),
    llm_mix: 0.02,
    epsilon: 0.3,
    replay_n: 4,
    directive: document.getElementById('directive').value || 'farm',
  };
  await fetch('/api/start', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
};
document.getElementById('btnStop').onclick = async () => {
  await fetch('/api/stop', { method: 'POST' });
};
document.getElementById('btnClear').onclick = () => { logEl.innerHTML = ''; };
setInterval(poll, 400);
poll();
</script>
</body>
</html>
"""


@dataclass
class GuiState:
    running: bool = False
    stop_flag: bool = False
    events: deque = field(default_factory=lambda: deque(maxlen=500))
    next_id: int = 1
    thread: threading.Thread | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, typ: str, **payload: Any) -> None:
        with self.lock:
            ev = {"id": self.next_id, "type": typ, "t": time.time(), **payload}
            self.next_id += 1
            self.events.append(ev)

    def since(self, after: int) -> list[dict[str, Any]]:
        with self.lock:
            return [e for e in self.events if e["id"] > after]


STATE = GuiState()


def _run_owned(opts: dict[str, Any]) -> None:
    STATE.running = True
    STATE.stop_flag = False
    STATE.push("info", message="Owned loop starting…")
    try:
            max_ticks_raw = opts.get("max_ticks", 0)
            try:
                max_ticks = int(max_ticks_raw)
            except (TypeError, ValueError):
                max_ticks = 0
            cfg = OwnedLoopConfig(
                config_path=Path(opts.get("config") or "config/owned_game.json"),
                dry_run=not bool(opts.get("live", True)),
                use_ollama=bool(opts.get("ollama", True)),
                ollama_model=str(opts.get("ollama_model") or "llama3.2"),
                vision_model=str(opts.get("vision_model") or "qwen2.5vl:7b"),
                use_screen_llm=True,
                max_ticks=max_ticks,  # 0 = run until Stop
                learn=True,
                use_learned_policy=True,
                use_teacher=True,
                teacher_model=str(opts.get("teacher_model") or opts.get("ollama_model") or "llama3.2"),
                tick_seconds=float(opts.get("tick_seconds") if opts.get("tick_seconds") is not None else 0.05),
                vision_every=int(opts.get("vision_every") or 24),
                llm_mix=float(opts.get("llm_mix") if opts.get("llm_mix") is not None else 0.02),
                epsilon=float(opts.get("epsilon") if opts.get("epsilon") is not None else 0.3),
                replay_n=int(opts.get("replay_n") if opts.get("replay_n") is not None else 4),
            )

            def on_status(status: dict[str, Any]) -> None:
                STATE.push("status", status=status)
                print(
                    f"[brain] tick={status.get('tick')} action={status.get('action')} "
                    f"think={(status.get('thinking') or '')[:120]!r}"
                )

            loop = OwnedGameLoop(
                cfg=cfg,
                directive=str(opts.get("directive") or "farm") or None,
                on_status=on_status,
                should_stop=lambda: STATE.stop_flag,
            )
            loop.run()
            if STATE.stop_flag:
                STATE.push("info", message="Stopped by you.")
            else:
                STATE.push("info", message="Owned loop finished.")
    except SystemExit as exc:
        STATE.push("info", message=f"Stopped: {exc}")
    except Exception as exc:  # noqa: BLE001
        STATE.push("error", message=str(exc))
    finally:
        STATE.running = False
        STATE.stop_flag = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self) -> None:
        body = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._html()
            return
        if parsed.path == "/api/events":
            qs = parse_qs(parsed.query)
            after = int((qs.get("after") or ["0"])[0])
            self._json(
                200,
                {
                    "running": STATE.running,
                    "events": STATE.since(after),
                },
            )
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            opts = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            opts = {}

        if parsed.path == "/api/start":
            if STATE.running:
                self._json(409, {"error": "already_running"})
                return
            t = threading.Thread(target=_run_owned, args=(opts,), daemon=True)
            STATE.thread = t
            t.start()
            self._json(200, {"ok": True})
            return
        if parsed.path == "/api/stop":
            STATE.stop_flag = True
            self._json(200, {"ok": True, "stopping": True})
            return
        self._json(404, {"error": "not_found"})


def main(host: str = "127.0.0.1", port: int = 8777, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"PlayMind Brain GUI at {url}")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down GUI.")
    finally:
        STATE.stop_flag = True
        server.server_close()


if __name__ == "__main__":
    main()
