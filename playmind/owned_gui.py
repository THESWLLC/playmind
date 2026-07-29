"""Live brain monitor for the owned-game loop.

Shows realtime LLM output, OCR, invented abilities, and Learning V2 controls
(policy mode, demo recording, episode/diagnostics). Demo recording is owned by
GUI state (DemonstrationRecorder); live-loop policy mode is applied from
in-memory ``learning_v2`` config (or the owned_game.json ``learning_v2``
section) on the next Start — OwnedGameLoop does not yet expose mid-run hooks
for every V2 control.

Stdlib only: python -m playmind.owned_gui
Then open http://127.0.0.1:8777

Optional: F9 toggles demo recording when ``pynput`` is installed.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
import zipfile
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from playmind.demonstrations import DemonstrationRecorder
from playmind.learning_v2_controller import LearningV2Config
from playmind.owned_loop import OwnedGameLoop, OwnedLoopConfig


DEFAULT_LEARNING_V2: dict[str, Any] = {
    "enabled": True,
    "policy_mode": "hybrid",
    "legacy_q_fallback": False,
    "history_length": 16,
    "confidence_threshold": 0.45,
    "bc_checkpoint": None,
    "use_rewards_v2": True,
    "track_episodes": True,
}

POLICY_MODES = ("scripted", "hybrid", "legacy_q", "behavior_clone")


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
    button, input, select, textarea {
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
    #stats, #v2stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: .45rem; }
    .stat {
      background: #10161f; border: 1px solid var(--border); border-radius: 9px; padding: .45rem .55rem;
    }
    .stat .k { color: var(--muted); font-size: .65rem; text-transform: uppercase; }
    .stat .v { font-family: var(--mono); font-size: .85rem; margin-top: .1rem; word-break: break-word; }
    #abilities, #ocr, #sensors {
      font-family: var(--mono); font-size: .75rem; color: var(--muted);
      background: #10161f; border: 1px solid var(--border); border-radius: 9px;
      padding: .55rem; min-height: 48px; white-space: pre-wrap;
    }
    .row { display: flex; gap: .5rem; flex-wrap: wrap; align-items: center; margin-top: .5rem; }
    label { color: var(--muted); font-size: .82rem; }
    details.v2 {
      margin-top: 1rem; border: 1px solid var(--border); border-radius: 10px;
      padding: .55rem .75rem; background: #121820;
    }
    details.v2 > summary {
      cursor: pointer; color: var(--accent); font-size: .82rem;
      text-transform: uppercase; letter-spacing: .08em; list-style: none;
    }
    details.v2 > summary::-webkit-details-marker { display: none; }
    .hint { color: var(--muted); font-size: .72rem; margin-top: .35rem; }
    .demo-rec { color: var(--warn); }
  </style>
</head>
<body>
  <header>
    <h1><span>PlayMind</span> Brain</h1>
    <div style="display:flex;gap:.5rem;align-items:center">
      <div class="badge" id="demoBadge">demo idle</div>
      <div class="badge" id="conn">connecting…</div>
    </div>
  </header>
  <main>
    <section class="card">
      <h2>Controls</h2>
      <div class="controls">
        <button class="primary" id="btnStart">Start live</button>
        <button class="danger" id="btnStop" disabled>Stop</button>
        <button id="btnClear">Clear log</button>
        <button id="btnResetEp">Reset episode</button>
        <button id="btnClearQ">Clear legacy Q</button>
        <button id="btnDiag">Export diagnostics</button>
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

      <h2 style="margin-top:1rem">Learning V2</h2>
      <div class="row">
        <label>policy
          <select id="policyMode">
            <option value="scripted">scripted</option>
            <option value="hybrid" selected>hybrid</option>
            <option value="legacy_q">legacy_q</option>
            <option value="behavior_clone">behavior_clone</option>
          </select>
        </label>
        <label>model path <input id="modelPath" type="text" placeholder="data/playmind/models/bc.pt" style="width:16rem" /></label>
        <button id="btnSaveV2">Apply V2 config</button>
      </div>
      <p class="hint">V2 config is applied in memory for the <em>next</em> Start (also written when owned_game.json has a learning_v2 section). Hotkey: F9 start/stop demo (if pynput available).</p>
      <div id="v2stats" style="margin-top:.55rem"></div>
      <h2 style="margin-top:.85rem">Sensor confidence</h2>
      <div id="sensors">(waiting…)</div>

      <details class="v2" id="advV2">
        <summary>Advanced V2 · Demonstration recording</summary>
        <div class="row" style="margin-top:.55rem">
          <label>name <input id="demoName" type="text" placeholder="session name" style="width:10rem" /></label>
          <label>goal <input id="demoGoal" type="text" placeholder="farm / quest" style="width:10rem" /></label>
          <label>profile <input id="demoProfile" type="text" placeholder="character" style="width:8rem" /></label>
        </div>
        <div class="row">
          <label>notes <input id="demoNotes" type="text" placeholder="optional notes" style="width:22rem" /></label>
        </div>
        <div class="controls" style="margin-top:.55rem">
          <button class="primary" id="btnDemoStart">Start recording</button>
          <button id="btnDemoStop" disabled>Stop recording</button>
          <button id="btnMarkOk">Mark Success</button>
          <button id="btnMarkFail">Mark Failure</button>
          <button class="danger" id="btnMarkBad">Mark Bad</button>
        </div>
        <p class="hint" id="demoHint">Not recording. F9 toggles start/stop when a global hotkey listener is available.</p>
      </details>

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
const v2El = document.getElementById('v2stats');
const ocrEl = document.getElementById('ocr');
const abilEl = document.getElementById('abilities');
const sensorsEl = document.getElementById('sensors');
const soulEl = document.getElementById('soul');
const conn = document.getElementById('conn');
const demoBadge = document.getElementById('demoBadge');
let lastId = 0;

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function renderV2(s) {
  const snap = s.skill_snapshot || {};
  const cells = [
    ['mode', s.policy_mode],
    ['model', s.model_version],
    ['path', s.model_path],
    ['confidence', s.confidence],
    ['skill', s.active_skill],
    ['skill status', s.skill_status || snap.last_status],
    ['skill elapsed', s.skill_elapsed],
    ['episode', s.episode_id],
    ['ep reward', s.episode_reward],
    ['allowed', (s.allowed_skills || []).slice(0, 8).join(', ')],
    ['masked', (s.masked_skills || []).slice(0, 8).join(', ')],
  ];
  v2El.innerHTML = cells.map(([k,v]) =>
    `<div class="stat"><div class="k">${esc(k)}</div><div class="v">${esc(v == null || v === '' ? '—' : v)}</div></div>`
  ).join('');
  const sc = s.sensor_confidence || s.sensor_confidence_summary || {};
  if (sc && typeof sc === 'object' && Object.keys(sc).length) {
    sensorsEl.textContent = Object.entries(sc).map(([k,v]) =>
      Array.isArray(v) ? `${k}: ${v.join('; ')}` : `${k}=${v}`
    ).join('  |  ');
  } else {
    sensorsEl.textContent = '(no sensor confidence yet)';
  }
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
  renderV2(s);
}

function appendLog(ev) {
  const s = ev.status || {};
  const div = document.createElement('div');
  div.className = 'log-line';
  const thought = s.thinking || s.llm_raw || '';
  const err = s.llm_error ? `<div class="err">error: ${esc(s.llm_error)}</div>` : '';
  const v2meta = s.active_skill
    ? ` · skill=${esc(s.active_skill)} · conf=${esc(s.confidence)}`
    : '';
  div.innerHTML = `
    <div><span class="t">#${esc(s.tick)}</span>
      <span class="a">${esc(s.action)}</span>
      <span class="meta">${esc(s.decision || s.brain_mode || '')} · r=${esc(s.reward)} · hp=${esc(s.vision_hp)}${v2meta}</span>
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
    const demo = data.demo || {};
    const recording = !!demo.recording;
    demoBadge.textContent = recording ? ('REC ' + (demo.sample_count || 0)) : 'demo idle';
    demoBadge.className = 'badge' + (recording ? ' live' : '');
    document.getElementById('btnDemoStart').disabled = recording;
    document.getElementById('btnDemoStop').disabled = !recording;
    document.getElementById('demoHint').textContent = recording
      ? ('Recording session ' + (demo.session_id || '') + ' · samples=' + (demo.sample_count || 0) + ' · F9 to stop')
      : 'Not recording. F9 toggles start/stop when a global hotkey listener is available.';
    if (data.learning_v2) {
      const lv = data.learning_v2;
      if (lv.policy_mode) document.getElementById('policyMode').value = lv.policy_mode;
      if (lv.bc_checkpoint != null) document.getElementById('modelPath').value = lv.bc_checkpoint || '';
    }
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

async function post(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(body || {}),
  });
  return r.json();
}

document.getElementById('btnStart').onclick = async () => {
  await post('/api/v2/config', {
    policy_mode: document.getElementById('policyMode').value,
    bc_checkpoint: document.getElementById('modelPath').value || null,
    enabled: true,
  });
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
  await post('/api/start', body);
};
document.getElementById('btnStop').onclick = async () => { await post('/api/stop'); };
document.getElementById('btnClear').onclick = () => { logEl.innerHTML = ''; };
document.getElementById('btnSaveV2').onclick = async () => {
  const res = await post('/api/v2/config', {
    policy_mode: document.getElementById('policyMode').value,
    bc_checkpoint: document.getElementById('modelPath').value || null,
    enabled: true,
  });
  const div = document.createElement('div');
  div.className = 'log-line';
  div.innerHTML = `<div class="meta">V2 config saved for next start: ${esc(JSON.stringify(res.learning_v2 || res))}</div>`;
  logEl.appendChild(div);
};
document.getElementById('btnResetEp').onclick = async () => {
  const res = await post('/api/episode/reset');
  const div = document.createElement('div');
  div.className = 'log-line';
  div.innerHTML = `<div class="meta">Episode reset: ${esc(JSON.stringify(res))}</div>`;
  logEl.appendChild(div);
};
document.getElementById('btnClearQ').onclick = async () => {
  if (!confirm('Rename policy.json → policy.json.legacy.bak?')) return;
  const res = await post('/api/legacy_q/clear');
  const div = document.createElement('div');
  div.className = 'log-line';
  div.innerHTML = `<div class="meta">Legacy Q clear: ${esc(JSON.stringify(res))}</div>`;
  logEl.appendChild(div);
};
document.getElementById('btnDiag').onclick = async () => {
  const res = await post('/api/diagnostics/export');
  const div = document.createElement('div');
  div.className = 'log-line';
  div.innerHTML = `<div class="meta">Diagnostics: ${esc(res.path || JSON.stringify(res))}</div>`;
  logEl.appendChild(div);
};

function demoPayload() {
  return {
    name: document.getElementById('demoName').value || undefined,
    goal: document.getElementById('demoGoal').value || undefined,
    profile: document.getElementById('demoProfile').value || undefined,
    notes: document.getElementById('demoNotes').value || undefined,
  };
}
document.getElementById('btnDemoStart').onclick = async () => { await post('/api/demo/start', demoPayload()); };
document.getElementById('btnDemoStop').onclick = async () => { await post('/api/demo/stop', demoPayload()); };
document.getElementById('btnMarkOk').onclick = async () => {
  await post('/api/demo/mark', {...demoPayload(), outcome: 'success'});
};
document.getElementById('btnMarkFail').onclick = async () => {
  await post('/api/demo/mark', {...demoPayload(), outcome: 'failure'});
};
document.getElementById('btnMarkBad').onclick = async () => {
  await post('/api/demo/mark', {...demoPayload(), outcome: 'bad'});
};

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
    learning_v2: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LEARNING_V2))
    demo_recorder: DemonstrationRecorder | None = None
    demo_meta: dict[str, Any] = field(default_factory=dict)
    last_status: dict[str, Any] = field(default_factory=dict)
    loop: Any = None  # OwnedGameLoop while running, if accessible
    data_dir: Path = field(default_factory=lambda: Path("data/playmind/owned"))
    hotkey_note: str = "F9 start/stop demo (pynput optional)"

    def push(self, typ: str, **payload: Any) -> None:
        with self.lock:
            ev = {"id": self.next_id, "type": typ, "t": time.time(), **payload}
            self.next_id += 1
            self.events.append(ev)

    def since(self, after: int) -> list[dict[str, Any]]:
        with self.lock:
            return [e for e in self.events if e["id"] > after]

    def demo_snapshot(self) -> dict[str, Any]:
        rec = self.demo_recorder
        if rec is None:
            return {"recording": False, "sample_count": 0, "session_id": None}
        return {
            "recording": bool(rec.recording),
            "sample_count": int(rec.sample_count),
            "session_id": rec.session_id,
            "episode_id": rec.episode_id,
            "outcome": rec.outcome,
            "session_dir": str(rec.session_dir) if rec.session_dir else None,
        }


STATE = GuiState()


def _ensure_demo_recorder() -> DemonstrationRecorder:
    if STATE.demo_recorder is None:
        STATE.demo_recorder = DemonstrationRecorder()
    return STATE.demo_recorder


def _maybe_append_demo(status: dict[str, Any]) -> None:
    rec = STATE.demo_recorder
    if rec is None or not rec.recording:
        return
    try:
        frame = status.get("frame")
        notes = STATE.demo_meta.get("notes")
        rec.append(
            frame_path=frame,
            observation={
                k: status.get(k)
                for k in (
                    "tick",
                    "action",
                    "reward",
                    "vision_hp",
                    "has_target",
                    "is_dead",
                    "is_ghost",
                    "life_phase",
                    "active_skill",
                    "policy_mode",
                    "confidence",
                    "episode_id",
                    "goal",
                    "stuck",
                )
                if k in status or status.get(k) is not None
            },
            key_events=[{"action": status.get("action"), "tick": status.get("tick")}],
            goal=STATE.demo_meta.get("goal") or status.get("goal"),
            profile=STATE.demo_meta.get("profile"),
            notes=notes,
            episode_id=status.get("episode_id") or rec.episode_id,
            skill=status.get("active_skill"),
            label=None,
        )
    except Exception as exc:  # noqa: BLE001
        STATE.push("error", message=f"demo append failed: {exc}")


def _run_owned(opts: dict[str, Any]) -> None:
    # Hot-reload learning/sensor modules so Stop→Start picks up code edits
    # without requiring a full GUI process restart.
    import importlib

    import playmind.life_fsm as _life_fsm
    import playmind.owned_loop as _owned_loop
    import playmind.process_memory as _process_memory
    import playmind.progress as _progress
    import playmind.screen_llm as _screen_llm
    import playmind.travel as _travel
    import playmind.ui_memory as _ui_memory
    import playmind.vision as _vision
    import playmind.learning as _learning
    import playmind.learning_v2_controller as _lv2

    for mod in (
        _life_fsm,
        _ui_memory,
        _screen_llm,
        _vision,
        _progress,
        _process_memory,
        _travel,
        _learning,
        _lv2,
        _owned_loop,
    ):
        importlib.reload(mod)
    global OwnedGameLoop, OwnedLoopConfig, LearningV2Config
    OwnedGameLoop = _owned_loop.OwnedGameLoop
    OwnedLoopConfig = _owned_loop.OwnedLoopConfig
    LearningV2Config = _lv2.LearningV2Config

    STATE.running = True
    STATE.stop_flag = False
    STATE.loop = None
    STATE.push("info", message="Owned loop starting…")
    try:
        max_ticks_raw = opts.get("max_ticks", 0)
        try:
            max_ticks = int(max_ticks_raw)
        except (TypeError, ValueError):
            max_ticks = 0

        v2_raw = dict(STATE.learning_v2)
        if isinstance(opts.get("learning_v2"), dict):
            v2_raw.update(opts["learning_v2"])
        v2_raw.setdefault("enabled", True)
        v2_cfg = LearningV2Config.from_owned_dict({"learning_v2": v2_raw})

        data_dir = Path(opts.get("data_dir") or STATE.data_dir)
        STATE.data_dir = data_dir

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
            data_dir=data_dir,
            learning_v2=v2_cfg,
        )

        def on_status(status: dict[str, Any]) -> None:
            STATE.last_status = dict(status)
            STATE.push("status", status=status)
            _maybe_append_demo(status)
            print(
                f"[brain] tick={status.get('tick')} action={status.get('action')} "
                f"skill={status.get('active_skill')} "
                f"think={(status.get('thinking') or '')[:120]!r}"
            )

        loop = OwnedGameLoop(
            cfg=cfg,
            directive=str(opts.get("directive") or "farm") or None,
            on_status=on_status,
            should_stop=lambda: STATE.stop_flag,
        )
        STATE.loop = loop
        STATE.push(
            "info",
            message=(
                f"Learning V2 mode={v2_cfg.policy_mode} enabled={v2_cfg.enabled} "
                f"model={v2_cfg.bc_checkpoint or '(none)'} "
                "(live loop uses this in-memory config on start; "
                "config file learning_v2 also applies when learning_v2 is unset on OwnedLoopConfig)"
            ),
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
        STATE.loop = None


def _update_v2_config(opts: dict[str, Any]) -> dict[str, Any]:
    cur = dict(STATE.learning_v2)
    for key in (
        "enabled",
        "policy_mode",
        "legacy_q_fallback",
        "history_length",
        "confidence_threshold",
        "bc_checkpoint",
        "use_rewards_v2",
        "track_episodes",
        "model_path",
    ):
        if key not in opts:
            continue
        if key == "model_path":
            cur["bc_checkpoint"] = opts[key] or None
        elif key == "policy_mode":
            mode = str(opts[key] or "hybrid")
            if mode not in POLICY_MODES:
                raise ValueError(f"policy_mode must be one of {POLICY_MODES}, got {mode!r}")
            cur["policy_mode"] = mode
        else:
            cur[key] = opts[key]
    STATE.learning_v2 = cur
    return cur


def _clear_legacy_q(data_dir: Path | None = None) -> dict[str, Any]:
    root = Path(data_dir or STATE.data_dir)
    policy = root / "policy.json"
    bak = root / "policy.json.legacy.bak"
    if not policy.exists():
        return {"ok": True, "cleared": False, "reason": "policy.json not found", "path": str(policy)}
    if bak.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        bak = root / f"policy.json.legacy.bak.{stamp}"
    policy.rename(bak)
    return {"ok": True, "cleared": True, "backup": str(bak)}


def _reset_episode() -> dict[str, Any]:
    """Reset episode on live loop if accessible; always refresh demo episode id."""
    result: dict[str, Any] = {"ok": True, "loop_reset": False, "demo_reset": False}
    loop = STATE.loop
    if loop is not None:
        v2 = getattr(loop, "_v2", None)
        mgr = getattr(v2, "episode_mgr", None) if v2 is not None else None
        if mgr is not None:
            try:
                if mgr.current is not None and not mgr.current.done:
                    mgr.truncate("manual_reset", note="gui_reset")
                ep = mgr.start(reason="manual_reset")
                result["loop_reset"] = True
                result["episode_id"] = ep.episode_id
            except Exception as exc:  # noqa: BLE001
                result["loop_error"] = str(exc)
    rec = STATE.demo_recorder
    if rec is not None:
        import uuid

        rec.episode_id = str(uuid.uuid4())
        result["demo_reset"] = True
        result["demo_episode_id"] = rec.episode_id
    if not result["loop_reset"]:
        result["note"] = (
            "OwnedGameLoop episode manager not reachable; "
            "demo episode id refreshed if recorder exists. "
            "Full V2 episode tracking applies when the loop is running with learning_v2 enabled."
        )
    STATE.push("info", message=f"Episode reset: {json.dumps(result)}")
    return result


def _export_diagnostics() -> dict[str, Any]:
    out_dir = Path("data/playmind/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    zip_path = out_dir / f"diagnostics-{stamp}.zip"
    payload = {
        "schema_version": 1,
        "exported_at": time.time(),
        "learning_v2": STATE.learning_v2,
        "demo": STATE.demo_snapshot(),
        "demo_meta": STATE.demo_meta,
        "last_status": STATE.last_status,
        "running": STATE.running,
        "recent_events": list(STATE.since(max(0, STATE.next_id - 80))),
        "note": (
            "Live loop integration uses in-memory learning_v2 on Start; "
            "config file learning_v2 applies when OwnedLoopConfig.learning_v2 is unset."
        ),
    }
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("diagnostics.json", json.dumps(payload, indent=2, default=str) + "\n")
        # Optional: include latest frame path reference if present
        frame = STATE.last_status.get("frame")
        if frame and Path(frame).is_file():
            try:
                zf.write(frame, arcname="latest_frame" + Path(frame).suffix)
            except OSError:
                pass
    STATE.push("info", message=f"Diagnostics exported to {zip_path}")
    return {"ok": True, "path": str(zip_path)}


def _demo_start(opts: dict[str, Any]) -> dict[str, Any]:
    rec = _ensure_demo_recorder()
    if rec.recording:
        return {"ok": False, "error": "already_recording", **STATE.demo_snapshot()}
    STATE.demo_meta = {
        "name": opts.get("name"),
        "goal": opts.get("goal"),
        "profile": opts.get("profile"),
        "notes": opts.get("notes"),
    }
    session_id = opts.get("name") or opts.get("session_id")
    sid = rec.start(
        session_id=str(session_id) if session_id else None,
        goal=opts.get("goal"),
        profile=opts.get("profile"),
    )
    STATE.push("info", message=f"Demo recording started: {sid}")
    return {"ok": True, **STATE.demo_snapshot()}


def _demo_stop(opts: dict[str, Any] | None = None) -> dict[str, Any]:
    rec = STATE.demo_recorder
    if rec is None or not rec.recording:
        return {"ok": False, "error": "not_recording", **STATE.demo_snapshot()}
    if opts:
        if opts.get("notes") is not None:
            STATE.demo_meta["notes"] = opts.get("notes")
    path = rec.stop()
    STATE.push("info", message=f"Demo recording stopped: {path}")
    return {"ok": True, "session_dir": str(path), **STATE.demo_snapshot()}


def _demo_mark(opts: dict[str, Any]) -> dict[str, Any]:
    rec = _ensure_demo_recorder()
    outcome = str(opts.get("outcome") or "").lower()
    if outcome not in {"success", "failure", "bad"}:
        return {"ok": False, "error": "outcome must be success|failure|bad"}
    if rec.session_dir is None:
        # Allow marking only after at least one start created a session
        return {"ok": False, "error": "no_session", "hint": "Start recording first"}
    notes = opts.get("notes")
    if notes is None:
        notes = STATE.demo_meta.get("notes")
    rec.mark(outcome, notes=notes, sample_id=opts.get("sample_id"))  # type: ignore[arg-type]
    STATE.push("info", message=f"Demo marked {outcome}")
    return {"ok": True, "outcome": outcome, **STATE.demo_snapshot()}


def _toggle_demo_hotkey() -> None:
    rec = STATE.demo_recorder
    if rec is not None and rec.recording:
        _demo_stop()
    else:
        _demo_start(dict(STATE.demo_meta))


def _start_f9_hotkey_thread() -> None:
    """Optional F9 start/stop demo. Skips gracefully if pynput is unavailable."""
    try:
        from pynput import keyboard  # type: ignore
    except Exception:  # noqa: BLE001
        STATE.hotkey_note = "F9 hotkey unavailable (install pynput for global toggle)"
        return

    def on_press(key: Any) -> None:
        try:
            if key == keyboard.Key.f9:
                _toggle_demo_hotkey()
        except Exception:  # noqa: BLE001
            pass

    def run() -> None:
        try:
            with keyboard.Listener(on_press=on_press) as listener:
                STATE.hotkey_note = "F9 start/stop demo (pynput listener active)"
                listener.join()
        except Exception as exc:  # noqa: BLE001
            STATE.hotkey_note = f"F9 hotkey failed: {exc}"

    t = threading.Thread(target=run, name="playmind-f9-demo", daemon=True)
    t.start()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
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

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            opts = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            opts = {}
        return opts if isinstance(opts, dict) else {}

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
                    "demo": STATE.demo_snapshot(),
                    "learning_v2": STATE.learning_v2,
                    "hotkey": STATE.hotkey_note,
                },
            )
            return
        if parsed.path == "/api/v2/config":
            self._json(200, {"learning_v2": STATE.learning_v2, "hotkey": STATE.hotkey_note})
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        opts = self._read_json()

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
        if parsed.path == "/api/v2/config":
            try:
                cur = _update_v2_config(opts)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, {"ok": True, "learning_v2": cur})
            return
        if parsed.path == "/api/demo/start":
            self._json(200, _demo_start(opts))
            return
        if parsed.path == "/api/demo/stop":
            self._json(200, _demo_stop(opts))
            return
        if parsed.path == "/api/demo/mark":
            self._json(200, _demo_mark(opts))
            return
        if parsed.path == "/api/episode/reset":
            self._json(200, _reset_episode())
            return
        if parsed.path == "/api/legacy_q/clear":
            self._json(200, _clear_legacy_q(Path(opts["data_dir"]) if opts.get("data_dir") else None))
            return
        if parsed.path == "/api/diagnostics/export":
            self._json(200, _export_diagnostics())
            return
        self._json(404, {"error": "not_found"})


def main(host: str = "127.0.0.1", port: int = 8777, open_browser: bool = True) -> None:
    _start_f9_hotkey_thread()
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"PlayMind Brain GUI at {url}")
    print(f"  {STATE.hotkey_note}")
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
