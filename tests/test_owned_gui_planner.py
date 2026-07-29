"""Planner-facing owned GUI API tests."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from playmind import owned_gui
from playmind.owned_gui import Handler, STATE
from playmind.planner_v2.model_registry import ModelRegistry
from playmind.planner_v2.modes import PlannerMode


class _Executor:
    def __init__(self) -> None:
        self.cleared = False

    def clear(self) -> None:
        self.cleared = True


class _Runtime:
    def __init__(self) -> None:
        self.mode = PlannerMode.SHADOW
        self.executor = _Executor()
        self.last_validation = None
        self.emergency_stop = False
        self.approved: bool | None = None

    def approve_plan(self, approved: bool) -> None:
        self.approved = approved

    def set_emergency_stop(self, active: bool) -> None:
        self.emergency_stop = active

    def snapshot(self) -> dict:
        return {
            "mode": self.mode.value,
            "emergency_stop": self.emergency_stop,
            "awaiting_approval": False,
            "can_send_input": False,
            "executor": {
                "plan": {
                    "goal": "test",
                    "skills": [{"name": "wait"}],
                    "confidence": 0.8,
                    "reason_code": "test",
                    "summary": "Wait safely",
                }
            },
            "latency_seconds": 0.01,
        }


def _server() -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server, int(server.server_address[1])


def _request(port: int, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_status_mode_planner_and_estop_endpoints(tmp_path: Path) -> None:
    runtime = _Runtime()
    STATE.running = False
    STATE.last_status = {"goal": "test", "life_phase": "alive"}
    STATE.mode = "shadow"
    STATE.planner_runtime = runtime
    STATE.registry = ModelRegistry(tmp_path / "registry.sqlite")
    STATE.emergency_stop = False
    STATE.demo_recorder = None
    server, port = _server()
    try:
        code, status = _request(port, "/api/status")
        assert code == 200
        assert status["mode"] == "shadow"
        assert status["planner"]["plan"]["goal"] == "test"
        assert status["learning_proof"]["verdict"] in {"YES", "NO", "INSUFFICIENT"}

        assert _request(port, "/api/mode", {"mode": "assist"})[1]["mode"] == "assist"
        assert runtime.mode is PlannerMode.ASSIST
        assert _request(port, "/api/planner/approve", {})[1]["approved"] is True
        assert runtime.approved is True
        assert _request(port, "/api/planner/reject", {})[1]["approved"] is False
        assert runtime.executor.cleared is True

        stopped = _request(port, "/api/emergency_stop", {})[1]
        assert stopped["emergency_stop"] is True
        assert runtime.emergency_stop is True
    finally:
        server.shutdown()
        STATE.planner_runtime = None
        STATE.registry = None
        STATE.emergency_stop = False
        STATE.stop_flag = False


def test_registry_models_and_learning_proof_endpoints(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry.sqlite")
    metrics = {
        "valid_plan_rate": 1.0,
        "illegal_skill_rate": 0.0,
        "scenario_count": 200,
        "benchmark_score": 0.6,
    }
    registry.register("production", status="candidate", eval_metrics=metrics)
    registry.promote("production")
    registry.register(
        "candidate",
        status="candidate",
        eval_metrics={**metrics, "benchmark_score": 0.8},
    )
    STATE.registry = registry
    server, port = _server()
    try:
        code, models = _request(port, "/api/registry/models")
        assert code == 200
        assert {row["model_id"] for row in models["models"]} == {"production", "candidate"}
        proof = _request(port, "/api/learning_proof")[1]
        assert proof["verdict"] == "YES"
        assert proof["candidate_score"] > proof["production_score"]
    finally:
        server.shutdown()
        STATE.registry = None
