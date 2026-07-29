from __future__ import annotations

import json
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from playmind import studio_gui
from playmind.gui.studio_dashboard import StudioGuiState


def _request(
    port: int,
    path: str,
    body: dict | None = None,
    *,
    method: str | None = None,
) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method=method or ("POST" if body is not None else "GET"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture
def studio_server(tmp_path: Path) -> tuple[int, StudioGuiState]:
    original = studio_gui.STATE
    state = StudioGuiState(
        projects_root=tmp_path / "projects",
        storage_root=tmp_path / "studio",
        data_root=tmp_path / "data",
        registry_path=tmp_path / "registry.sqlite",
    )
    state.app.store.create_project(
        project_id="review",
        profile="retail_wow_offline_only",
        provenance={
            "source_type": "synthetic",
            "rights_confirmed": True,
            "license_confirmed": True,
        },
    )
    state.app.select_project("review")
    studio_gui.STATE = state
    server = ThreadingHTTPServer(("127.0.0.1", 0), studio_gui.Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield int(server.server_address[1]), state
    finally:
        server.shutdown()
        server.server_close()
        studio_gui.STATE = original


def test_status_and_annotation_crud(
    studio_server: tuple[int, StudioGuiState],
) -> None:
    port, _state = studio_server
    code, status = _request(port, "/api/status")
    assert code == 200
    assert status["offline_only"] is True
    assert status["live_controls_blocked"] is True
    assert status["profile"]["name"] == "retail_wow_offline_only"

    code, created = _request(
        port,
        "/api/annotations",
        {"start": 1, "end": 2, "category": "wait"},
    )
    assert code == 201
    segment_id = created["annotation"]["segment_id"]
    reviewed = _request(
        port,
        "/api/annotations/review",
        {"segment_id": segment_id, "accepted": True},
    )[1]
    assert reviewed["annotation"]["review_status"] == "reviewed"
    assert _request(
        port, f"/api/annotations/{segment_id}", method="DELETE"
    )[0] == 200
    assert _request(port, "/api/annotations")[1]["annotations"] == []


def test_profile_blocks_live_controls(
    studio_server: tuple[int, StudioGuiState],
) -> None:
    port, _state = studio_server
    code, body = _request(port, "/api/live/start", {})
    assert code == 403
    assert body["offline_only"] is True
    assert "prohibits" in body["error"]


def test_smoke_model_is_prominently_labeled(
    studio_server: tuple[int, StudioGuiState],
) -> None:
    port, state = studio_server
    state.registry.register(
        "smoke-test",
        status="candidate",
        smoke=True,
        allowed_uses=["smoke_validation"],
    )
    model = _request(port, "/api/models")[1]["models"][0]
    assert model["display_label"] == "SMOKE / NO REAL WEIGHTS TRAINED"
    proof = _request(port, "/api/learning_proof")[1]
    assert proof["verdict"] == "INSUFFICIENT"
    assert proof["card_state"] == "smoke_only"
