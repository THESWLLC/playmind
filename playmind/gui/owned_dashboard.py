"""Data adapters and browser UI for the owned-game dashboard."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from playmind.demonstrations import list_sessions, load_session_samples
from playmind.planner_v2.model_registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLANNER_V2: dict[str, Any] = {
    "enabled": True,
    "provider": "ollama",
    "host": "http://127.0.0.1:11434",
    "production_model": "llama3.2",
    "candidate_model": None,
    "maximum_plan_skills": 5,
    "minimum_confidence": 0.55,
    "periodic_replan_seconds": 10,
    "timeout_seconds": 12,
    "fallback": "heuristic",
}
PLANNER_MODES = ("observe", "shadow", "assist", "hybrid", "autonomous", "replay")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def owned_config() -> dict[str, Any]:
    for path in (ROOT / "config/owned_game.json", ROOT / "config/owned_game.example.json"):
        if path.exists():
            return _read_json(path)
    return {}


def get_registry(state: Any) -> ModelRegistry:
    registry = getattr(state, "registry", None)
    if registry is None:
        registry = ModelRegistry(ROOT / "data/playmind/planner/registry.sqlite")
        state.registry = registry
    return registry


def latest_eval_report() -> dict[str, Any]:
    candidates: list[Path] = []
    for base in (
        ROOT / "data/playmind/eval",
        ROOT / "data/playmind/evaluation",
        ROOT / "data/playmind/planner/eval",
    ):
        if base.exists():
            candidates.extend(base.rglob("report.json"))
    if not candidates:
        return {
            "available": False,
            "path": None,
            "report": {},
            "comparisons": {},
            "warning": "No evaluation report found.",
        }
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    report = _read_json(path)
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, Mapping):
        comparisons = report.get("models")
    return {
        "available": bool(report),
        "path": str(path),
        "modified_at": path.stat().st_mtime,
        "report": report,
        "comparisons": dict(comparisons) if isinstance(comparisons, Mapping) else {},
    }


def _score(model: Mapping[str, Any] | None) -> float | None:
    if not model:
        return None
    for name in (
        "benchmark_score",
        "mean_agreement",
        "agreement_rate",
        "score",
        "accuracy",
    ):
        try:
            value = model.get(name)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
    for source_name in ("eval_metrics", "metrics", "summary"):
        source = model.get(source_name)
        if not isinstance(source, Mapping):
            continue
        for name in (
            "benchmark_score",
            "mean_agreement",
            "agreement_rate",
            "score",
            "accuracy",
        ):
            try:
                value = source.get(name)
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                pass
    return None


def learning_proof(state: Any) -> dict[str, Any]:
    try:
        models = get_registry(state).list_models()
    except Exception as exc:  # noqa: BLE001
        return {
            "verdict": "INSUFFICIENT",
            "reason": f"Model registry unavailable: {exc}",
            "production": None,
            "candidate": None,
        }
    production = next((row for row in models if row.get("status") == "production"), None)
    candidates = [row for row in models if row.get("status") == "candidate"]
    candidate = candidates[0] if candidates else None
    production_score = _score(production)
    candidate_score = _score(candidate)
    evidence_source = "registry"
    if production_score is None or candidate_score is None:
        report = latest_eval_report()
        comparisons = report.get("comparisons") or {}
        if isinstance(comparisons, Mapping):
            config = owned_config().get("planner_v2") or {}
            production_name = str(config.get("production_model") or "production")
            candidate_name = config.get("candidate_model")
            production_row = comparisons.get(production_name) or comparisons.get("production")
            candidate_row = (
                comparisons.get(str(candidate_name)) if candidate_name else None
            ) or comparisons.get("candidate")
            rows = [
                (str(name), row)
                for name, row in comparisons.items()
                if isinstance(row, Mapping)
            ]
            if production_row is None and rows:
                production_name, production_row = rows[0]
            if candidate_row is None and len(rows) > 1:
                candidate_name, candidate_row = rows[1]
            production_score = production_score if production_score is not None else _score(production_row)
            candidate_score = candidate_score if candidate_score is not None else _score(candidate_row)
            production = production or (
                {"model_id": production_name, "eval_metrics": dict(production_row)}
                if isinstance(production_row, Mapping)
                else None
            )
            candidate = candidate or (
                {"model_id": str(candidate_name), "eval_metrics": dict(candidate_row)}
                if isinstance(candidate_row, Mapping)
                else None
            )
            if production_score is not None or candidate_score is not None:
                evidence_source = "eval_report"
    if production_score is None or candidate_score is None:
        verdict = "INSUFFICIENT"
        reason = "Both production and candidate need comparable evaluation scores."
    elif candidate_score > production_score:
        verdict = "YES"
        reason = "Candidate score exceeds the production score on recorded evaluation."
    else:
        verdict = "NO"
        reason = "Candidate did not exceed the production score."
    return {
        "verdict": verdict,
        "reason": reason,
        "production": production,
        "candidate": candidate,
        "production_score": production_score,
        "candidate_score": candidate_score,
        "evidence_source": evidence_source,
        "evidence_only": True,
        "warning": "Offline scores do not prove live gameplay improvement.",
    }


def dataset_summary(state: Any) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []
    roots = (ROOT / "data/playmind/planner", ROOT / "data/playmind/datasets")
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*.manifest.json"):
            payload = _read_json(path)
            if payload:
                manifests.append({"path": str(path), **payload})
    split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    total = 0
    leakage: list[str] = []
    for manifest in manifests:
        counts = manifest.get("counts") if isinstance(manifest.get("counts"), Mapping) else {}
        total += int(counts.get("total") or 0)
        by_split = counts.get("by_split")
        if isinstance(by_split, Mapping):
            split_counts.update({str(k): int(v or 0) for k, v in by_split.items()})
        coverage = manifest.get("coverage")
        if isinstance(coverage, Mapping):
            for category in coverage.get("categories") or []:
                category_counts[str(category)] += 1
        warnings = manifest.get("leakage_warnings") or manifest.get("warnings") or []
        leakage.extend(str(item) for item in warnings)
    demo_root = getattr(getattr(state, "demo_recorder", None), "root", None)
    demos = list_sessions(demo_root)
    demo_samples = sum(len(load_session_samples(path)) for path in demos)
    denominator = max(1, max(category_counts.values(), default=1))
    return {
        "counts": {
            "manifest_records": total,
            "demonstration_sessions": len(demos),
            "demonstration_samples": demo_samples,
        },
        "split_counts": dict(split_counts),
        "coverage": {
            name: {"count": count, "percent": round(count * 100 / denominator, 1)}
            for name, count in sorted(category_counts.items())
        },
        "leakage_warnings": leakage,
        "manifests": manifests,
    }


def replay_summary(state: Any, *, limit: int = 60) -> dict[str, Any]:
    recorder = getattr(state, "demo_recorder", None)
    root = getattr(recorder, "root", None)
    sessions = list_sessions(root)
    if not sessions:
        return {"sessions": [], "selected": None, "steps": []}
    selected = recorder.session_dir if recorder and recorder.session_dir else sessions[-1]
    if selected not in sessions and Path(selected).exists():
        sessions.append(Path(selected))
    rows = load_session_samples(selected)
    return {
        "sessions": [str(item) for item in sessions[-30:]],
        "selected": str(selected),
        "step_count": len(rows),
        "steps": rows[-limit:],
    }


def demonstration_summary(state: Any) -> dict[str, Any]:
    snapshot = state.demo_snapshot()
    snapshot["goal"] = getattr(state, "demo_meta", {}).get("goal")
    snapshot["physical_input_count"] = int(getattr(state, "physical_input_count", 0))
    snapshot["segmented_skills"] = list(getattr(state, "segmented_skills", []))[-30:]
    capture = getattr(state, "physical_capture", None)
    snapshot["physical_capture"] = {
        "running": bool(getattr(capture, "running", False)),
        "available": bool(getattr(capture, "available", False)),
    }
    warnings: list[str] = []
    if snapshot["recording"] and not snapshot["physical_capture"]["available"]:
        warnings.append("Physical input capture is unavailable; install/configure pynput.")
    if snapshot["recording"] and snapshot["sample_count"] and not snapshot["physical_input_count"]:
        warnings.append("No physical input has been captured in this recording.")
    snapshot["quality_warnings"] = warnings
    return snapshot


def planner_summary(state: Any) -> dict[str, Any]:
    runtime = getattr(state, "planner_runtime", None)
    if runtime is None:
        config = dict(DEFAULT_PLANNER_V2)
        config.update(getattr(state, "planner_v2", {}) or {})
        return {
            "enabled": bool(config.get("enabled", True)),
            "mode": getattr(state, "mode", "shadow"),
            "objective": getattr(state, "last_status", {}).get("goal"),
            "plan": None,
            "summary": "Planner has not started.",
            "skills_queue": [],
            "confidence": None,
            "reason_code": None,
            "validation": None,
            "fallback": config.get("fallback", "heuristic"),
            "latency_ms": None,
            "replan_trigger": None,
            "awaiting_approval": False,
        }
    snap = runtime.snapshot()
    executor = snap.get("executor") or {}
    plan = executor.get("plan")
    skills = plan.get("skills") if isinstance(plan, Mapping) else []
    validation = getattr(runtime, "last_validation", None)
    if hasattr(validation, "__dict__"):
        validation = dict(validation.__dict__)
    return {
        "enabled": True,
        "mode": snap.get("mode"),
        "objective": (plan or {}).get("goal") if isinstance(plan, Mapping) else None,
        "plan": plan,
        "summary": (plan or {}).get("summary") if isinstance(plan, Mapping) else None,
        "skills_queue": skills or [],
        "confidence": (plan or {}).get("confidence") if isinstance(plan, Mapping) else None,
        "reason_code": (plan or {}).get("reason_code") if isinstance(plan, Mapping) else None,
        "validation": validation,
        "fallback": snap.get("plan_source"),
        "latency_ms": round(float(snap.get("latency_seconds") or 0) * 1000, 1),
        "replan_trigger": snap.get("trigger"),
        **snap,
    }


def training_summary(state: Any) -> dict[str, Any]:
    process = getattr(state, "training_process", None)
    running = bool(process is not None and process.poll() is None)
    status_path = ROOT / "data/playmind/training/status.json"
    payload = _read_json(status_path)
    payload.setdefault("preset", "rtx_4070_ti_3b_qlora")
    payload["running"] = running
    payload["pid"] = process.pid if running else None
    payload.setdefault("epoch", None)
    payload.setdefault("step", None)
    payload.setdefault("losses", {})
    payload.setdefault("vram", {"used_gb": None, "total_gb": None})
    payload.setdefault("overfitting_warning", None)
    payload["status_path"] = str(status_path)
    payload["log_path"] = str(ROOT / "data/playmind/training/smoke.log")
    return payload


def start_smoke_training(state: Any, opts: Mapping[str, Any]) -> dict[str, Any]:
    current = getattr(state, "training_process", None)
    if current is not None and current.poll() is None:
        return {"ok": False, "error": "training_already_running", **training_summary(state)}
    output = ROOT / "data/playmind/training"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts/train_behavior_clone.py"),
        "--data-dir",
        str(opts.get("data_dir") or ROOT / "data/playmind/demonstrations"),
        "--dry-validate-only",
    ]
    log = (output / "smoke.log").open("a", encoding="utf-8")
    process = subprocess.Popen(  # noqa: S603 - fixed local script
        command,
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    state.training_process = process
    state.training_log = log
    status = {
        "preset": str(opts.get("preset") or "smoke_validate"),
        "started_at": time.time(),
        "command": command,
        "running": True,
        "epoch": 0,
        "step": 0,
        "losses": {},
    }
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, **training_summary(state)}


def stop_training(state: Any) -> dict[str, Any]:
    process = getattr(state, "training_process", None)
    if process is None or process.poll() is not None:
        return {"ok": False, "error": "training_not_running", **training_summary(state)}
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
    log = getattr(state, "training_log", None)
    if log is not None:
        log.close()
    return {"ok": True, "stopped": True, **training_summary(state)}


def alert_summary(state: Any) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for event in list(getattr(state, "events", []))[-100:]:
        if event.get("type") == "error" or "warning" in str(event.get("message", "")).lower():
            alerts.append(
                {
                    "severity": "error" if event.get("type") == "error" else "warning",
                    "message": event.get("message"),
                    "timestamp": event.get("t"),
                }
            )
    status = getattr(state, "last_status", {})
    if status.get("modal_menu"):
        alerts.append({"severity": "warning", "message": "Blocking modal detected."})
    if status.get("stuck"):
        alerts.append({"severity": "warning", "message": f"Agent appears stuck ({status['stuck']})."})
    if getattr(state, "emergency_stop", False):
        alerts.append({"severity": "error", "message": "Emergency stop is active."})
    return alerts[-50:]


def rich_status(state: Any) -> dict[str, Any]:
    cfg = owned_config()
    status = dict(getattr(state, "last_status", {}))
    planner = planner_summary(state)
    proof = learning_proof(state)
    mode = str(getattr(state, "mode", None) or cfg.get("mode") or "shadow")
    auth = {
        "i_own_this_game": bool(cfg.get("i_own_this_game", False)),
        "enable_keyboard": bool(cfg.get("enable_keyboard", False)),
        "can_send_input": bool(planner.get("can_send_input", False)),
    }
    frame = status.get("frame")
    perception = {
        "latest_frame_path": frame,
        "latest_frame_url": "/api/frame/latest" if frame else None,
        "sensors": status.get("sensor_confidence") or status.get("sensor_confidence_summary") or {},
        "known": status.get("ui_known"),
        "unknown": status.get("unknown_sensors") or [],
        "confidence": status.get("confidence"),
        "life_phase": status.get("life_phase"),
        "ocr_snippet": (status.get("screen_ocr") or "")[:500],
        "stuck": status.get("stuck") or status.get("stuck_hint"),
        "modal": bool(status.get("modal_menu")),
        "focused": not bool(getattr(state, "soft_estop", False)),
    }
    models: list[dict[str, Any]]
    try:
        models = get_registry(state).list_models()
    except Exception as exc:  # noqa: BLE001
        models = [{"error": str(exc)}]
    return {
        "running": bool(getattr(state, "running", False)),
        "mode": mode,
        "focus": perception["focused"],
        "auth": auth,
        "active_model": (
            (proof.get("production") or {}).get("model_id")
            or (cfg.get("planner_v2") or {}).get("production_model")
            or "llama3.2"
        ),
        "recording": bool(getattr(getattr(state, "demo_recorder", None), "recording", False)),
        "training": training_summary(state),
        "emergency_stop": bool(getattr(state, "emergency_stop", False)),
        "warnings": [item["message"] for item in alert_summary(state)],
        "learning_proof": proof,
        "perception": perception,
        "planner": planner,
        "demonstration": demonstration_summary(state),
        "dataset": dataset_summary(state),
        "eval": latest_eval_report(),
        "models": models,
        "alerts": alert_summary(state),
        "replay": replay_summary(state),
        "learning_v2": getattr(state, "learning_v2", {}),
        "last_status": status,
        "updated_at": time.time(),
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PlayMind Control Center</title>
<style>
:root{--bg:#091018;--panel:#111c28;--line:#27384b;--text:#e8f0f8;--muted:#93a8bc;--a:#59e1c2;--warn:#ffc857;--bad:#ff6577}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px "Segoe UI",system-ui,sans-serif}
header{display:flex;justify-content:space-between;align-items:center;padding:14px 20px;border-bottom:1px solid var(--line);position:sticky;top:0;background:#091018ee;z-index:3}
h1{font-size:18px;margin:0}h1 span{color:var(--a)}.bad{color:var(--bad)}.ok{color:var(--a)}.warn{color:var(--warn)}
nav{display:flex;gap:6px;overflow:auto;padding:10px 16px;border-bottom:1px solid var(--line);position:sticky;top:49px;background:#091018;z-index:2}
button,input,select{color:var(--text);background:#172535;border:1px solid var(--line);border-radius:7px;padding:7px 10px}
button{cursor:pointer}button:hover,button.active{border-color:var(--a)}button.danger{border-color:var(--bad)}
main{max-width:1400px;margin:auto;padding:16px}.tab{display:none}.tab.active{display:block}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;min-height:120px}.card h2{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin:0 0 10px}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0}.metric{font-size:22px;font-weight:650}.label{font-size:11px;color:var(--muted)}
pre{white-space:pre-wrap;word-break:break-word;background:#080d13;border-radius:7px;padding:10px;max-height:550px;overflow:auto;font:12px Consolas,monospace}
table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid var(--line);padding:8px}.bar{height:8px;background:#253345;border-radius:6px}.bar i{display:block;height:100%;background:var(--a)}
@media(max-width:700px){nav{top:47px}main{padding:10px}}
</style></head><body>
<header><h1><span>PlayMind</span> Control Center</h1><div><span id="conn">connecting</span> · <span id="safe">shadow / keyboard off</span></div></header>
<nav id="tabs"></nav><main>
<section class="tab" data-tab="Dashboard"><div class="row">
<button id="btnStart">Start</button><button id="btnStop">Stop</button><button class="danger" id="estop">Emergency stop</button>
<select id="mode"><option>observe</option><option selected>shadow</option><option>assist</option><option>hybrid</option><option>autonomous</option><option>replay</option></select>
<input id="directive" value="farm to level 2"><label><input id="live" type="checkbox"> live keyboard</label>
</div><div class="grid" id="dashboard"></div></section>
<section class="tab" data-tab="Live Perception"><div class="grid"><div class="card"><h2>Latest frame</h2><img id="frame" style="max-width:100%;display:none"></div><div class="card"><h2>Perception</h2><pre id="perception"></pre></div></div></section>
<section class="tab" data-tab="Planner"><div class="row"><button id="approve">Approve plan</button><button id="reject">Reject plan</button></div><div class="card"><pre id="planner"></pre></div></section>
<section class="tab" data-tab="Demonstrations"><div class="row"><input id="demoName" placeholder="session"><input id="demoGoal" placeholder="goal"><button id="btnDemoStart">Start recording</button><button id="btnDemoStop">Stop</button><button data-mark="success">Success</button><button data-mark="failure">Failure</button><button data-mark="bad">Bad</button></div><div class="card"><pre id="demonstrations"></pre></div></section>
<section class="tab" data-tab="Dataset"><div class="card"><div id="coverage"></div><pre id="dataset"></pre></div></section>
<section class="tab" data-tab="Training"><div class="row"><select id="preset"><option>smoke_validate</option><option>rtx_4070_ti_3b_qlora</option></select><button id="trainStart">Start smoke train</button><button id="trainStop">Stop train</button></div><div class="card"><pre id="training"></pre></div></section>
<section class="tab" data-tab="Learning Proof"><div class="card"><div class="metric" id="proofVerdict">INSUFFICIENT</div><pre id="proof"></pre></div></section>
<section class="tab" data-tab="Model Comparison"><div class="card"><div id="comparison"></div></div></section>
<section class="tab" data-tab="Models"><div class="card"><div id="models"></div></div></section>
<section class="tab" data-tab="Alerts"><div class="card"><pre id="alerts"></pre></div></section>
<section class="tab" data-tab="Replay"><div class="card"><pre id="replay"></pre></div></section>
<section class="tab" data-tab="System Doctor"><div class="row"><button id="doctorRun">Run doctor</button></div><div class="card"><pre id="doctor"></pre></div></section>
<!-- Compatibility controls retained for existing clients/tests: policyMode behavior_clone Advanced V2 btnClear btnResetEp btnClearQ btnDiag F9 /api/diagnostics/export -->
<select id="policyMode" hidden><option>scripted</option><option>hybrid</option><option>legacy_q</option><option>behavior_clone</option></select>
</main><script>
const names=[...document.querySelectorAll('.tab')].map(x=>x.dataset.tab), nav=document.querySelector('#tabs');
names.forEach((name,i)=>{const b=document.createElement('button');b.textContent=name;b.onclick=()=>show(name);nav.appendChild(b);if(!i)show(name)});
function show(name){document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x.dataset.tab===name));[...nav.children].forEach(x=>x.classList.toggle('active',x.textContent===name))}
const esc=x=>String(x??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pretty=x=>JSON.stringify(x,null,2); async function post(url,body={}){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return r.json()}
function cards(s){const vals=[['Mode',s.mode],['Focus',s.focus],['Authorization',s.auth.can_send_input?'input allowed':'input blocked'],['Active model',s.active_model],['Recording',s.recording],['Training',s.training.running],['E-stop',s.emergency_stop],['Is the model learning?',s.learning_proof.verdict]];document.querySelector('#dashboard').innerHTML=vals.map(([k,v])=>`<div class=card><h2>${esc(k)}</h2><div class=metric>${esc(v)}</div></div>`).join('')+`<div class=card><h2>Warnings</h2>${(s.warnings||[]).map(x=>`<div class=warn>${esc(x)}</div>`).join('')||'None'}</div>`}
function table(rows){if(!rows.length)return '<p>No data available.</p>';const keys=[...new Set(rows.flatMap(x=>Object.keys(x)))].slice(0,8);return `<table><tr>${keys.map(k=>`<th>${esc(k)}</th>`).join('')}</tr>${rows.map(r=>`<tr>${keys.map(k=>`<td>${esc(typeof r[k]==='object'?JSON.stringify(r[k]):r[k])}</td>`).join('')}</tr>`).join('')}</table>`}
async function refresh(){try{const s=await (await fetch('/api/status')).json();window.statusBlob=s;document.querySelector('#conn').textContent=s.running?'LIVE':'idle';document.querySelector('#safe').textContent=`${s.mode} / ${s.auth.can_send_input?'keyboard authorized':'keyboard blocked'}`;document.querySelector('#mode').value=s.mode;document.querySelector('#estop').textContent=s.emergency_stop?'Clear emergency stop':'Emergency stop';cards(s);for(const [id,key] of [['perception','perception'],['planner','planner'],['demonstrations','demonstration'],['dataset','dataset'],['training','training'],['proof','learning_proof'],['alerts','alerts'],['replay','replay']])document.querySelector('#'+id).textContent=pretty(s[key]);document.querySelector('#proofVerdict').textContent=s.learning_proof.verdict;document.querySelector('#comparison').innerHTML=table(Object.entries(s.eval.comparisons||{}).map(([model,v])=>({model,...v})));document.querySelector('#models').innerHTML=table(s.models||[])+(s.models||[]).filter(x=>x.model_id).map(x=>`<div class=row><b>${esc(x.model_id)}</b><button onclick="modelAction('promote','${esc(x.model_id)}')">Promote</button><button onclick="modelAction('reject','${esc(x.model_id)}')">Reject</button><button onclick="modelAction('archive','${esc(x.model_id)}')">Archive</button><button onclick="modelAction('rollback','${esc(x.model_id)}')">Rollback</button><button onclick="modelAction('export_ollama','${esc(x.model_id)}')">Export Ollama</button></div>`).join('');document.querySelector('#coverage').innerHTML=Object.entries(s.dataset.coverage||{}).map(([k,v])=>`<label>${esc(k)} ${v.percent}%</label><div class=bar><i style="width:${v.percent}%"></i></div>`).join('');const img=document.querySelector('#frame');if(s.perception.latest_frame_url){img.src=s.perception.latest_frame_url+'?t='+Date.now();img.style.display='block'}}catch(e){document.querySelector('#conn').textContent='offline'}}
async function modelAction(action,id){if(!confirm(`${action} ${id}?`))return;await post('/api/registry/'+action,{model_id:id,reason:'GUI operator action'});refresh()}window.modelAction=modelAction;
document.querySelector('#mode').onchange=e=>post('/api/mode',{mode:e.target.value}).then(refresh);
document.querySelector('#btnStart').onclick=()=>post('/api/start',{directive:document.querySelector('#directive').value,live:document.querySelector('#live').checked}).then(refresh);
document.querySelector('#btnStop').onclick=()=>post('/api/stop').then(refresh);document.querySelector('#estop').onclick=()=>post('/api/emergency_stop',{active:!(window.statusBlob||{}).emergency_stop}).then(refresh);
document.querySelector('#approve').onclick=()=>post('/api/planner/approve').then(refresh);document.querySelector('#reject').onclick=()=>post('/api/planner/reject').then(refresh);
document.querySelector('#btnDemoStart').onclick=()=>post('/api/demo/start',{name:document.querySelector('#demoName').value,goal:document.querySelector('#demoGoal').value}).then(refresh);
document.querySelector('#btnDemoStop').onclick=()=>post('/api/demo/stop').then(refresh);document.querySelectorAll('[data-mark]').forEach(b=>b.onclick=()=>post('/api/demo/mark',{outcome:b.dataset.mark}).then(refresh));
document.querySelector('#trainStart').onclick=()=>post('/api/training/start',{preset:document.querySelector('#preset').value}).then(refresh);document.querySelector('#trainStop').onclick=()=>post('/api/training/stop').then(refresh);
document.querySelector('#doctorRun').onclick=async()=>document.querySelector('#doctor').textContent=pretty(await(await fetch('/api/doctor')).json());
setInterval(refresh,1000);refresh();
</script></body></html>"""


__all__ = [
    "DEFAULT_PLANNER_V2",
    "INDEX_HTML",
    "PLANNER_MODES",
    "alert_summary",
    "dataset_summary",
    "get_registry",
    "latest_eval_report",
    "learning_proof",
    "replay_summary",
    "rich_status",
    "start_smoke_training",
    "stop_training",
]
