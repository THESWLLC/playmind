from __future__ import annotations

from pathlib import Path

import pytest

from playmind.studio.frame_extractor import create_tiny_synthetic_fixture
from playmind.studio.media_probe import MediaToolUnavailableError, probe_media
from playmind.studio.project_store import ProjectStore
from playmind.studio.provenance import ProvenanceRecord, is_training_eligible


def test_provenance_json_and_conservative_eligibility() -> None:
    owned = ProvenanceRecord(
        "user_owned_recording", source_id="recording-1", rights_confirmed=True
    )
    assert ProvenanceRecord.from_json(owned.to_json()) == owned
    assert is_training_eligible(owned)
    assert not is_training_eligible(ProvenanceRecord("unknown"))
    assert not is_training_eligible(
        ProvenanceRecord("friend_provided", rights_confirmed=True)
    )
    assert is_training_eligible(
        ProvenanceRecord(
            "friend_provided",
            rights_confirmed=True,
            consent_confirmed=True,
        )
    )


def test_probe_refuses_to_fake_success_without_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"not actually a video")
    monkeypatch.setattr("playmind.studio.media_probe.shutil.which", lambda _name: None)
    with pytest.raises(MediaToolUnavailableError, match="Install FFmpeg"):
        probe_media(video)


def test_tiny_synthetic_fixture_is_a_real_project_when_pillow_exists(
    tmp_path: Path,
) -> None:
    fixture = create_tiny_synthetic_fixture(tmp_path)
    if fixture is None:
        pytest.skip("Pillow is optional")
    assert len(fixture["frames"]) == 3
    assert all(Path(item["path"]).is_file() for item in fixture["frames"])
    project = ProjectStore(tmp_path).load_project("synthetic-fixture")
    assert project["media"]["synthetic"] is True
