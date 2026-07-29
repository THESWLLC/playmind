"""Offline Studio dashboard data helpers and browser application."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from playmind.planner_v2.model_registry import ModelRegistry
from playmind.studio.annotations import TimelineSegment, annotation_categories
from playmind.studio.app import StudioApp
from playmind.studio.benchmark_builder import BenchmarkBuilder
from playmind.studio.corrections import CorrectionStore, PlanCorrection
from playmind.studio.eval_index import load_latest
from playmind.studio.profiles import (
    PROFILE_RETAIL_WOW_OFFLINE_ONLY,
    get_profile,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config/studio.example.json"


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return default


def load_studio_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load Studio config without falling back to owned-game settings."""

    candidates = (
        [Path(path)]
        if path is not None
        else [ROOT / "config/studio.json", DEFAULT_CONFIG]
    )
    value: dict[str, Any] = {}
    for candidate in candidates:
        if candidate.is_file():
            loaded = _read_json(candidate, {})
            if isinstance(loaded, dict):
                value = dict(loaded)
                value["_config_path"] = str(candidate)
                break
    value.setdefault("profile", PROFILE_RETAIL_WOW_OFFLINE_ONLY)
    value.setdefault("storage_root", "data/playmind/studio")
    value.setdefault("projects_root", f"{value['storage_root']}/projects")
    value.setdefault("data_root", "data/playmind")
    value.setdefault(
        "registry_path", "data/playmind/planner/registry.sqlite"
    )
    return value


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _score(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    sources = [row]
    for name in ("eval_metrics", "metrics", "summary"):
        nested = row.get(name)
        if isinstance(nested, dict):
            sources.append(nested)
    for source in sources:
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
                continue
    return None


@dataclass
class LocalJob:
    name: str
    command: list[str]
    process: subprocess.Popen[Any]
    log_path: Path
    log_handle: Any
    started_at: float

    def snapshot(self) -> dict[str, Any]:
        code = self.process.poll()
        if code is not None and not self.log_handle.closed:
            self.log_handle.close()
        return {
            "name": self.name,
            "command": self.command,
            "pid": self.process.pid,
            "running": code is None,
            "returncode": code,
            "started_at": self.started_at,
            "log_path": str(self.log_path),
        }


class StudioGuiState:
    """Mutable state for a single local, offline-only Studio server."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        projects_root: str | Path | None = None,
        data_root: str | Path | None = None,
        storage_root: str | Path | None = None,
        registry_path: str | Path | None = None,
    ) -> None:
        self.config = load_studio_config(config_path)
        self.profile = get_profile(str(self.config["profile"]))
        self.storage_root = _repo_path(
            storage_root or self.config["storage_root"]
        )
        self.projects_root = _repo_path(
            projects_root or self.config["projects_root"]
        )
        self.data_root = _repo_path(data_root or self.config["data_root"])
        self.registry_path = _repo_path(
            registry_path or self.config["registry_path"]
        )
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.app = StudioApp(self.projects_root, data_root=self.data_root)
        self.lock = Lock()
        self.jobs: dict[str, LocalJob] = {}
        self.alerts: list[dict[str, Any]] = []
        self.review_focused = False
        self.selected_annotation_id: str | None = None
        self.hotkey_note = (
            "Annotation hotkeys unavailable; use review buttons. "
            "Hotkeys never work unless the Studio review panel is focused."
        )
        self._wizard_path = self.storage_root / "first_run.json"

    @property
    def registry(self) -> ModelRegistry:
        return ModelRegistry(self.registry_path)

    @property
    def evaluation_root(self) -> Path:
        return self.data_root / "planner/evaluation"

    @property
    def benchmark_root(self) -> Path:
        return self.evaluation_root

    def add_alert(self, message: str, severity: str = "error") -> None:
        self.alerts.append(
            {
                "severity": severity,
                "message": str(message),
                "timestamp": time.time(),
            }
        )
        del self.alerts[:-100]

    def current_project(self) -> dict[str, Any] | None:
        if not self.app.current_project_id:
            return None
        try:
            return self.app.store.load_project(self.app.current_project_id)
        except (KeyError, ValueError):
            return None

    def wizard_state(self) -> dict[str, Any]:
        saved = _read_json(self._wizard_path, {})
        if not isinstance(saved, dict):
            saved = {}
        projects = self.app.list_projects()
        project = self.current_project()
        reviewed = 0
        if project:
            reviewed = sum(
                item.review_status == "reviewed"
                for item in self.app.annotations().list()
            )
        doctor = studio_doctor(self)
        steps = [
            {
                "id": "welcome",
                "title": "Offline safety boundary",
                "complete": bool(saved.get("offline_acknowledged")),
                "detail": "Studio never captures a live client or sends input.",
            },
            {
                "id": "profile",
                "title": "Protected profile",
                "complete": self.profile.offline_only,
                "detail": self.profile.name,
            },
            {
                "id": "storage",
                "title": "Local storage",
                "complete": self.storage_root.is_dir(),
                "detail": str(self.storage_root),
            },
            {
                "id": "ffmpeg",
                "title": "FFmpeg tools",
                "complete": bool(
                    doctor["ffmpeg"]["available"]
                    and doctor["ffprobe"]["available"]
                ),
                "detail": "Required for recording import and frame extraction.",
            },
            {
                "id": "import",
                "title": "Import an authorized recording",
                "complete": bool(projects),
                "detail": "Confirm source rights before import.",
            },
            {
                "id": "review",
                "title": "Analyze and review annotations",
                "complete": bool(
                    project
                    and project.get("last_analysis")
                    and reviewed
                ),
                "detail": f"{reviewed} reviewed timeline segments",
            },
            {
                "id": "evidence",
                "title": "Export, freeze, train, and evaluate",
                "complete": bool(load_latest(self.evaluation_root)),
                "detail": "Learning Proof needs comparable held-out evaluation.",
            },
        ]
        return {
            "first_run": not bool(saved.get("completed")),
            "completed": bool(saved.get("completed")),
            "offline_acknowledged": bool(saved.get("offline_acknowledged")),
            "steps": steps,
            "next_step": next(
                (step["id"] for step in steps if not step["complete"]), None
            ),
        }

    def update_wizard(self, values: dict[str, Any]) -> dict[str, Any]:
        current = _read_json(self._wizard_path, {})
        if not isinstance(current, dict):
            current = {}
        if "offline_acknowledged" in values:
            current["offline_acknowledged"] = bool(
                values["offline_acknowledged"]
            )
        if values.get("completed"):
            if not current.get("offline_acknowledged"):
                raise ValueError("acknowledge the offline safety boundary first")
            current["completed"] = True
            current["completed_at"] = time.time()
        self._wizard_path.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self.wizard_state()

    def models(self) -> list[dict[str, Any]]:
        try:
            rows = self.registry.list_models()
        except Exception as exc:  # noqa: BLE001
            self.add_alert(f"Model registry unavailable: {exc}")
            return []
        result = []
        for row in rows:
            item = dict(row)
            item["display_label"] = (
                "SMOKE / NO REAL WEIGHTS TRAINED"
                if item.get("smoke")
                else str(item.get("display_name") or item.get("model_id"))
            )
            item["live_use_label"] = (
                "LIVE USE PROHIBITED"
                if item.get("live_use_prohibited")
                else ""
            )
            result.append(item)
        return result

    def learning_proof(self) -> dict[str, Any]:
        models = self.models()
        production = next(
            (
                item
                for item in models
                if item.get("status") == "production"
                and not item.get("smoke")
            ),
            None,
        )
        candidates = [
            item
            for item in models
            if item.get("status") == "candidate"
            and not item.get("smoke")
        ]
        candidate = candidates[0] if candidates else None
        smoke_models = [item for item in models if item.get("smoke")]
        latest = load_latest(self.evaluation_root)
        comparisons = (latest or {}).get("comparisons") or {}
        production_score = _score(production)
        candidate_score = _score(candidate)
        evidence_source = "registry"
        if isinstance(comparisons, dict):
            production_row = comparisons.get("production")
            candidate_row = comparisons.get("candidate")
            production_score = (
                production_score
                if production_score is not None
                else _score(production_row)
            )
            candidate_score = (
                candidate_score
                if candidate_score is not None
                else _score(candidate_row)
            )
            if production_score is not None or candidate_score is not None:
                evidence_source = "evaluation"
        if candidate is None and smoke_models:
            verdict = "INSUFFICIENT"
            card_state = "smoke_only"
            headline = "NO REAL TRAINING YET"
            reason = (
                "Smoke checks passed pipeline wiring only. "
                "SMOKE / NO REAL WEIGHTS TRAINED."
            )
        elif production_score is None or candidate_score is None:
            verdict = "INSUFFICIENT"
            card_state = "needs_evidence"
            headline = "NOT ENOUGH EVIDENCE"
            reason = (
                "A real production model and candidate need comparable "
                "held-out benchmark scores."
            )
        elif candidate_score > production_score:
            verdict = "YES"
            card_state = "proved"
            headline = "CANDIDATE IMPROVED OFFLINE"
            reason = "The real candidate exceeds production on recorded evaluation."
        else:
            verdict = "NO"
            card_state = "not_improved"
            headline = "NO MEASURED IMPROVEMENT"
            reason = "The candidate did not exceed production on recorded evaluation."
        return {
            "verdict": verdict,
            "card_state": card_state,
            "headline": headline,
            "reason": reason,
            "production": production,
            "candidate": candidate,
            "production_score": production_score,
            "candidate_score": candidate_score,
            "smoke_models": smoke_models,
            "evidence_source": evidence_source,
            "evaluation": latest,
            "evidence_only": True,
            "live_use_prohibited": self.profile.live_use_prohibited,
            "warning": (
                "Offline benchmark evidence does not prove live gameplay "
                "improvement or authorize live use."
            ),
        }

    def datasets(self) -> dict[str, Any]:
        manifests = []
        root = self.data_root / "planner/manifests/studio"
        if root.exists():
            for path in sorted(root.glob("*.json")):
                value = _read_json(path, {})
                if isinstance(value, dict):
                    manifests.append({"path": str(path), **value})
        vision = self.data_root / "vision/studio_visual_states.jsonl"
        return {
            "manifests": manifests,
            "visual_states": str(vision) if vision.exists() else None,
        }

    def benchmarks(self) -> list[dict[str, Any]]:
        rows = []
        if self.benchmark_root.exists():
            for path in sorted(self.benchmark_root.glob("*_v*.json")):
                value = _read_json(path, {})
                if isinstance(value, dict) and value.get("benchmark_id"):
                    rows.append({"path": str(path), **value})
        return rows

    def corrections(self) -> list[dict[str, Any]]:
        if not self.app.current_project_id:
            return []
        return [
            item.to_dict()
            for item in CorrectionStore(
                self.app.current_project_id, self.projects_root
            ).list()
        ]

    def training_status(self) -> dict[str, Any]:
        job = self.jobs.get("training")
        snapshot = job.snapshot() if job else {
            "name": "training",
            "running": False,
            "returncode": None,
        }
        manifests = sorted(
            (self.data_root / "planner/training/runs").glob(
                "*/training_manifest.json"
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        latest = _read_json(manifests[0], {}) if manifests else None
        return {
            **snapshot,
            "mode": "smoke",
            "label": "SMOKE / NO REAL WEIGHTS TRAINED",
            "real_weights_trained": False,
            "latest_manifest": latest,
        }

    def start_job(self, name: str, command: list[str]) -> dict[str, Any]:
        with self.lock:
            current = self.jobs.get(name)
            if current and current.process.poll() is None:
                raise RuntimeError(f"{name} is already running")
            logs = self.storage_root / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            log_path = logs / f"{name}.log"
            handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(  # noqa: S603 - fixed local scripts
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            job = LocalJob(name, command, process, log_path, handle, time.time())
            self.jobs[name] = job
            return job.snapshot()

    def start_smoke_training(self) -> dict[str, Any]:
        command = [
            sys.executable,
            str(ROOT / "scripts/train_planner_sft.py"),
            "--smoke",
            "--preset",
            "cpu_tiny_smoke",
            "--runs-root",
            str(self.data_root / "planner/training/runs"),
            "--registry-path",
            str(self.registry_path),
        ]
        return {
            **self.start_job("training", command),
            "label": "SMOKE / NO REAL WEIGHTS TRAINED",
            "real_weights_trained": False,
        }

    def start_evaluation(self, options: dict[str, Any]) -> dict[str, Any]:
        command = [
            sys.executable,
            str(ROOT / "scripts/evaluate_planner.py"),
            "--output-dir",
            str(self.evaluation_root),
            "--registry-path",
            str(self.registry_path),
            "--timeout",
            str(float(options.get("timeout", 5.0))),
        ]
        if options.get("suite"):
            command.extend(["--suite", str(_repo_path(options["suite"]))])
        if options.get("candidate_id"):
            command.extend(["--candidate-id", str(options["candidate_id"])])
        return self.start_job("evaluation", command)

    def status(self) -> dict[str, Any]:
        project = self.current_project()
        annotations = []
        analysis = []
        if project:
            annotations = [
                item.to_dict() for item in self.app.annotations().list()
            ]
            analysis = self.app.store.load_analysis(
                str(project["project_id"])
            )
        jobs = {name: job.snapshot() for name, job in self.jobs.items()}
        return {
            "state": self.app.state.value,
            "last_error": self.app.last_error,
            "profile": self.profile.to_dict(),
            "offline_only": True,
            "live_controls_blocked": True,
            "banner": (
                "OFFLINE-ONLY — retail_wow_offline_only blocks live capture, "
                "planning, process access, and generated input."
            ),
            "wizard": self.wizard_state(),
            "projects": self.app.list_projects(),
            "current_project": project,
            "annotation_count": len(annotations),
            "reviewed_annotation_count": sum(
                item.get("review_status") == "reviewed"
                for item in annotations
            ),
            "analysis_count": len(analysis),
            "datasets": self.datasets(),
            "benchmarks": self.benchmarks(),
            "evaluations": self.app.evaluations(),
            "training": self.training_status(),
            "jobs": jobs,
            "learning_proof": self.learning_proof(),
            "models": self.models(),
            "corrections": self.corrections(),
            "review_focused": self.review_focused,
            "selected_annotation_id": self.selected_annotation_id,
            "hotkey": self.hotkey_note,
            "alerts": self.alerts[-50:],
            "updated_at": time.time(),
        }


def studio_doctor(state: StudioGuiState) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    writable = os.access(state.storage_root, os.W_OK)
    return {
        "ok": bool(
            sys.version_info >= (3, 10)
            and ffmpeg
            and ffprobe
            and writable
        ),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "supported": sys.version_info >= (3, 10),
        },
        "ffmpeg": {
            "available": bool(ffmpeg),
            "path": ffmpeg,
            "note": "Install FFmpeg and ensure ffmpeg.exe is on PATH.",
        },
        "ffprobe": {"available": bool(ffprobe), "path": ffprobe},
        "storage": {
            "path": str(state.storage_root),
            "writable": writable,
            "free_gb": round(
                shutil.disk_usage(state.storage_root).free / 1024**3, 2
            ),
        },
        "optional": {
            "pynput": importlib.util.find_spec("pynput") is not None,
            "pillow": importlib.util.find_spec("PIL") is not None,
            "torch": importlib.util.find_spec("torch") is not None,
        },
        "profile": state.profile.to_dict(),
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PlayMind Studio</title>
<style>
:root{--bg:#081018;--panel:#111d29;--line:#294057;--text:#e9f2fa;--muted:#93aabd;--accent:#54dfbf;--warn:#ffd166;--bad:#ff6978}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px "Segoe UI",system-ui,sans-serif}header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;position:sticky;top:0;background:#081018f2;z-index:3}h1{font-size:19px;margin:0}h1 span{color:var(--accent)}
#offline{background:#502f00;color:#ffe8a6;padding:10px 20px;font-weight:700;text-align:center;border-bottom:1px solid #9a6200}.tabs{display:flex;gap:5px;padding:9px 12px;overflow:auto;position:sticky;top:48px;background:#081018;z-index:2;border-bottom:1px solid var(--line)}button,input,select,textarea{color:var(--text);background:#172838;border:1px solid var(--line);border-radius:7px;padding:8px}button{cursor:pointer}button:hover,button.active{border-color:var(--accent)}button.danger{border-color:var(--bad)}button:disabled{opacity:.45;cursor:not-allowed}
main{max-width:1450px;margin:auto;padding:16px}.tab{display:none}.tab.active{display:block}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:14px;margin-bottom:12px}.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 10px}.row{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:8px 0}.metric{font-size:24px;font-weight:700}.ok{color:var(--accent)}.warn{color:var(--warn)}.bad{color:var(--bad)}.muted{color:var(--muted)}.smoke{border:2px solid var(--warn);padding:12px;color:var(--warn);font-weight:800}.proof{border-left:5px solid var(--warn)}pre{background:#070c11;padding:10px;border-radius:7px;white-space:pre-wrap;word-break:break-word;max-height:500px;overflow:auto;font:12px Consolas,monospace}textarea{width:100%;min-height:90px}table{width:100%;border-collapse:collapse}td,th{padding:7px;text-align:left;border-bottom:1px solid var(--line)}.step{border-left:4px solid var(--line);padding:8px 12px;margin:7px 0}.step.done{border-color:var(--accent)}label{color:var(--muted)}input[type=text],input[type=number]{min-width:180px}
@media(max-width:700px){main{padding:10px}.tabs{top:47px}}
</style></head><body>
<header><h1><span>PlayMind</span> Studio</h1><div id="connection">connecting…</div></header>
<div id="offline">OFFLINE-ONLY · NO LIVE CAPTURE · NO GAME INPUT · retail_wow_offline_only</div>
<nav class="tabs" id="tabs"></nav><main>
<section class="tab" data-tab="Home/Wizard"><div class="grid"><div class="card"><h2>First-run wizard</h2><div id="wizard"></div><div class=row><button id="ack">Acknowledge offline boundary</button><button id="wizardDone">Finish wizard</button></div></div><div class="card proof"><h2>Learning Proof</h2><div class=metric id="homeProof">INSUFFICIENT</div><div id="homeProofReason"></div></div></div></section>
<section class="tab" data-tab="Projects/Import"><div class=card><h2>Import local authorized recording</h2><p>Path form only. Studio does not scrape or download media.</p><div class=row><input id="videoPath" type=text placeholder="C:\path\recording.mp4"><input id="projectName" type=text placeholder="Project name"><select id="importMode"><option>copy</option><option>reference</option></select><button id="importVideo">Import</button></div><label><input type=checkbox id="rights"> I own this recording or have permission to use it</label><label><input type=checkbox id="license"> License/consent permits local ML use</label><pre id="projects"></pre></div></section>
<section class="tab" data-tab="Analysis"><div class=card><div class=row><select id="frameStrategy"><option>overview</option><option>uniform</option><option>scene_change</option></select><button id="extract">Extract frames</button><label><input type=checkbox id="ocr"> OCR</label><button id="analyze">Analyze offline frames</button></div><pre id="analysis"></pre></div></section>
<section class="tab" data-tab="Annotation timeline"><div class=card id="reviewPanel" tabindex="0"><h2>Focus-gated review</h2><p id="hotkeyNote"></p><div class=row><input id="annStart" type=number min=0 step=.1 value=0><input id="annEnd" type=number min=0 step=.1 value=1><select id="annCategory"></select><select id="annType"><option>skill</option><option>goal</option><option>outcome</option></select><button id="annAdd">Add</button><button id="annAccept">Accept selected (F7)</button><button id="annReject">Reject selected (F8)</button><button id="annDelete" class=danger>Delete selected</button></div><div id="annotations"></div></div></section>
<section class="tab" data-tab="Datasets"><div class=card><div class=row><button id="exportData">Export reviewed project</button></div><pre id="datasets"></pre></div></section>
<section class="tab" data-tab="Benchmark Builder"><div class=card><p>Freeze reviewed, provenance-eligible scenarios into an immutable version.</p><input id="benchmarkId" type=text value="studio_real_benchmark"><textarea id="scenarios" placeholder='[{"scenario_id":"review-1","category":"loading","planner_state":{},"expected_plan":{"skills":["wait"]},"reviewed":true,"provenance_eligible":true}]'></textarea><button id="freeze">Freeze benchmark</button><pre id="benchmarks"></pre></div></section>
<section class="tab" data-tab="Training Readiness"><div class=card><button id="readinessRun">Refresh readiness</button><pre id="readiness"></pre></div></section>
<section class="tab" data-tab="Training"><div class=card><div class=smoke>SMOKE / NO REAL WEIGHTS TRAINED</div><p>This validates pipeline wiring only. It does not train usable weights.</p><button id="trainSmoke">Start smoke train</button><pre id="training"></pre></div></section>
<section class="tab" data-tab="Learning Proof"><div class="card proof"><div class=metric id="proofVerdict">INSUFFICIENT</div><h3 id="proofHeadline"></h3><pre id="proof"></pre><button id="evaluate">Evaluate offline</button></div></section>
<section class="tab" data-tab="Model Review/Corrections"><div class=card><h2>Add human plan correction</h2><textarea id="plannerState" placeholder='Planner state JSON: {}'>{}</textarea><textarea id="candidatePlan" placeholder='Candidate plan JSON: {"skills":["wait"]}'>{"skills":["wait"]}</textarea><textarea id="correctedPlan" placeholder='Corrected plan JSON: {"skills":["clear_modal","wait"]}'>{"skills":["clear_modal","wait"]}</textarea><button id="addCorrection">Add correction</button><div id="corrections"></div></div></section>
<section class="tab" data-tab="Models"><div class=card><div id="models"></div></div></section>
<section class="tab" data-tab="Doctor"><div class=card><button id="doctorRun">Run doctor</button><pre id="doctor"></pre></div></section>
<section class="tab" data-tab="Alerts"><div class=card><pre id="alerts"></pre></div></section>
</main><script>
const q=s=>document.querySelector(s), esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])), pretty=x=>JSON.stringify(x,null,2);let state={},selectedAnnotation=null;
document.querySelectorAll('.tab').forEach((x,i)=>{const b=document.createElement('button');b.textContent=x.dataset.tab;b.onclick=()=>show(x.dataset.tab);q('#tabs').appendChild(b);if(!i)show(x.dataset.tab)});function show(n){document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x.dataset.tab===n));[...q('#tabs').children].forEach(x=>x.classList.toggle('active',x.textContent===n))}
async function request(url,body,method='POST'){const r=await fetch(url,{method,headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data}async function act(fn){try{const x=await fn();if(x)alertResult(x);await refresh()}catch(e){alert(e.message)}}function alertResult(x){if(x.error)alert(x.error)}
function renderTable(rows,buttons){if(!rows||!rows.length)return '<p class=muted>No data yet.</p>';const keys=[...new Set(rows.flatMap(x=>Object.keys(x)))].slice(0,7);return `<table><tr>${keys.map(k=>`<th>${esc(k)}</th>`).join('')}<th></th></tr>${rows.map(r=>`<tr>${keys.map(k=>`<td>${esc(typeof r[k]==='object'?JSON.stringify(r[k]):r[k])}</td>`).join('')}<td>${buttons?buttons(r):''}</td></tr>`).join('')}</table>`}
function render(s){state=s;q('#connection').textContent='local / offline';q('#offline').textContent=s.banner;const w=s.wizard;q('#wizard').innerHTML=w.steps.map(x=>`<div class="step ${x.complete?'done':''}"><b>${x.complete?'✓':'○'} ${esc(x.title)}</b><div class=muted>${esc(x.detail)}</div></div>`).join('');q('#homeProof').textContent=s.learning_proof.verdict;q('#homeProofReason').textContent=s.learning_proof.reason;q('#projects').textContent=pretty({current:s.current_project,projects:s.projects});q('#analysis').textContent=pretty({count:s.analysis_count,last_analysis:(s.current_project||{}).last_analysis});q('#datasets').textContent=pretty(s.datasets);q('#benchmarks').textContent=pretty(s.benchmarks);q('#training').textContent=pretty(s.training);q('#proofVerdict').textContent=s.learning_proof.verdict;q('#proofHeadline').textContent=s.learning_proof.headline;q('#proof').textContent=pretty(s.learning_proof);q('#alerts').textContent=pretty(s.alerts);q('#hotkeyNote').textContent=s.hotkey;
q('#annotations').innerHTML=renderTable(window.annotations||[],r=>`<button onclick="selectAnn('${esc(r.segment_id)}')">${selectedAnnotation===r.segment_id?'Selected':'Select'}</button>`);q('#corrections').innerHTML=renderTable(s.corrections||[],r=>`<button onclick="reviewCorrection('${esc(r.correction_id)}',true)">Accept</button><button onclick="reviewCorrection('${esc(r.correction_id)}',false)">Reject</button>`);q('#models').innerHTML=(s.models||[]).map(x=>`<div class="card ${x.smoke?'smoke':''}"><b>${esc(x.model_id)}</b><div>${esc(x.display_label)}</div>${x.live_use_prohibited?'<div class=warn>LIVE USE PROHIBITED</div>':''}<pre>${esc(pretty(x))}</pre></div>`).join('')||'<p>No registered models.</p>'}
async function refresh(){try{const [s,a]=await Promise.all([fetch('/api/status').then(r=>r.json()),fetch('/api/annotations').then(r=>r.json())]);window.annotations=a.annotations||[];render(s)}catch(e){q('#connection').textContent='disconnected'}}
window.selectAnn=id=>{selectedAnnotation=id;request('/api/review/focus',{focused:true,segment_id:id}).then(refresh)};window.reviewCorrection=(id,accepted)=>act(()=>request('/api/corrections/review',{correction_id:id,accepted}));
q('#ack').onclick=()=>act(()=>request('/api/wizard',{offline_acknowledged:true}));q('#wizardDone').onclick=()=>act(()=>request('/api/wizard',{completed:true}));
q('#importVideo').onclick=()=>act(()=>request('/api/import',{path:q('#videoPath').value,name:q('#projectName').value||undefined,mode:q('#importMode').value,provenance:{source_type:'user_owned_recording',source_id:q('#videoPath').value,rights_confirmed:q('#rights').checked,license_confirmed:q('#license').checked}}));
q('#extract').onclick=()=>act(()=>request('/api/extract',{strategy:q('#frameStrategy').value}));q('#analyze').onclick=()=>act(()=>request('/api/analyze',{do_ocr:q('#ocr').checked}));
q('#annAdd').onclick=()=>act(()=>request('/api/annotations',{start:Number(q('#annStart').value),end:Number(q('#annEnd').value),category:q('#annCategory').value,segment_type:q('#annType').value}));
q('#annAccept').onclick=()=>selectedAnnotation&&act(()=>request('/api/annotations/review',{segment_id:selectedAnnotation,accepted:true}));q('#annReject').onclick=()=>selectedAnnotation&&act(()=>request('/api/annotations/review',{segment_id:selectedAnnotation,accepted:false}));q('#annDelete').onclick=()=>selectedAnnotation&&act(()=>request('/api/annotations/delete',{segment_id:selectedAnnotation}));
const panel=q('#reviewPanel');panel.onfocusin=()=>request('/api/review/focus',{focused:true,segment_id:selectedAnnotation});panel.onfocusout=e=>{if(!panel.contains(e.relatedTarget))request('/api/review/focus',{focused:false})};document.addEventListener('visibilitychange',()=>{if(document.hidden)request('/api/review/focus',{focused:false})});
q('#exportData').onclick=()=>act(()=>request('/api/datasets/export',{}));q('#freeze').onclick=()=>act(()=>request('/api/benchmarks/freeze',{benchmark_id:q('#benchmarkId').value,scenarios:JSON.parse(q('#scenarios').value)}));q('#readinessRun').onclick=async()=>q('#readiness').textContent=pretty(await request('/api/readiness',{},'POST'));q('#trainSmoke').onclick=()=>act(()=>request('/api/training/smoke',{}));q('#evaluate').onclick=()=>act(()=>request('/api/evaluate',{}));
q('#addCorrection').onclick=()=>act(()=>request('/api/corrections',{planner_state:JSON.parse(q('#plannerState').value),candidate_plan:JSON.parse(q('#candidatePlan').value),corrected_plan:JSON.parse(q('#correctedPlan').value)}));q('#doctorRun').onclick=async()=>q('#doctor').textContent=pretty(await fetch('/api/doctor').then(r=>r.json()));
fetch('/api/annotation/categories').then(r=>r.json()).then(x=>q('#annCategory').innerHTML=x.categories.map(k=>`<option>${esc(k)}</option>`).join(''));setInterval(refresh,1500);refresh();
</script></body></html>"""


__all__ = [
    "DEFAULT_CONFIG",
    "INDEX_HTML",
    "ROOT",
    "StudioGuiState",
    "annotation_categories",
    "load_studio_config",
    "studio_doctor",
]
