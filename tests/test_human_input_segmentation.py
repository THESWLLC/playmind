from __future__ import annotations

import json
from pathlib import Path

from playmind.demonstrations import DemonstrationRecorder, load_session_samples
from playmind.human_input import PhysicalInputCapture, partition_events_by_source
from playmind.segmentation import RuleBasedSkillSegmenter


class _Key:
    def __init__(self, char: str) -> None:
        self.char = char


def test_capture_duration_focus_and_source_separation() -> None:
    times = iter([1.0, 1.25])
    capture = PhysicalInputCapture(
        source="human",
        focus_provider=lambda: True,
        clock=lambda: next(times),
    )
    capture._on_key_press(_Key("x"))
    capture._on_key_release(_Key("x"))
    events = capture.snapshot_and_clear()
    assert [event["type"] for event in events] == ["key_down", "key_up"]
    assert events[1]["duration"] == 0.25
    assert all(event["source"] == "human" for event in events)
    grouped = partition_events_by_source(
        events + [{"type": "key_down", "source": "playmind_generated"}]
    )
    assert len(grouped["human"]) == 2
    assert len(grouped["playmind_generated"]) == 1


def test_generated_input_is_not_human_training_eligible(tmp_path: Path) -> None:
    recorder = DemonstrationRecorder(root=tmp_path, input_source="playmind_generated")
    recorder.start(episode_id="generated")
    generated = recorder.append(
        physical_events=[{"type": "key_down", "key": "w"}],
        inferred_skill="explore",
        training_eligible=True,
    )
    human = recorder.append(
        input_source="human",
        physical_events=[{"type": "key_down", "key": "w", "source": "human"}],
        inferred_skill="explore",
    )
    session = recorder.stop()
    assert generated["training_eligible"] is False
    assert generated["human_training_eligible"] is False
    assert generated["is_human_demonstration"] is False
    assert human["human_training_eligible"] is True
    assert load_session_samples(session)[0]["schema_version"] == 2
    assert json.loads((session / "session.json").read_text())["schema_version"] == 2


def test_rule_segmentation_sequences_confidence_and_override() -> None:
    segmenter = RuleBasedSkillSegmenter(confidence_threshold=0.9)
    combat = segmenter.segment(
        {
            "physical_events": [
                {"type": "key_down", "key": "tab"},
                {"type": "key_down", "key": "w"},
                {"type": "mouse_button", "button": "left", "pressed": True},
            ],
            "observation": {"player_hp": 0.9},
        }
    )
    assert [segment.skill for segment in combat] == [
        "acquire_target",
        "approach_target",
        "engage_target",
    ]
    assert combat[1].confidence < combat[0].confidence
    assert combat[1].training_eligible is True

    unstuck = segmenter.segment(
        {
            "physical_events": [{"type": "key_down", "key": "w"}],
            "observations": [{"motion": 0.0}] * 3,
        }
    )
    assert unstuck[0].skill == "unstuck"
    assert unstuck[0].training_eligible is False

    manual = segmenter.segment({}, manual_override="recover_health")
    assert manual[0].confidence == 1.0
    assert manual[0].manual_override is True
    assert manual[0].training_eligible is True
