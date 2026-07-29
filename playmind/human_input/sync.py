"""Utilities for aligning physical events with demonstration samples."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from playmind.human_input.capture import InputSource, PhysicalInputCapture


def partition_events_by_source(
    events: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Keep generated, physical, and unknown events explicitly separated."""
    grouped: dict[str, list[dict[str, Any]]] = {
        "human": [],
        "playmind_generated": [],
        "unknown": [],
    }
    for raw in events:
        event = dict(raw)
        source = str(event.get("source") or "unknown")
        if source not in grouped:
            source = "unknown"
            event["source"] = source
        grouped[source].append(event)
    return grouped


def events_between(
    events: Iterable[Mapping[str, Any]],
    start_timestamp: float,
    end_timestamp: float,
) -> list[dict[str, Any]]:
    """Return a stable timestamp-ordered half-open event window."""
    return sorted(
        (
            dict(event)
            for event in events
            if start_timestamp
            <= float(event.get("timestamp") or 0.0)
            < end_timestamp
        ),
        key=lambda event: float(event.get("timestamp") or 0.0),
    )


@dataclass
class InputSnapshot:
    physical_events: list[dict[str, Any]]
    input_source: InputSource

    @property
    def human_training_eligible(self) -> bool:
        return self.input_source == "human"

    def recorder_fields(self) -> dict[str, Any]:
        return {
            "physical_events": list(self.physical_events),
            "input_source": self.input_source,
        }


class PhysicalInputSynchronizer:
    """Drain a capture queue into recorder-ready sample fields."""

    def __init__(self, capture: PhysicalInputCapture) -> None:
        self.capture = capture

    def snapshot(self) -> InputSnapshot:
        events = self.capture.snapshot_and_clear()
        sources = {
            str(event.get("source") or "unknown")
            for event in events
        }
        if not events:
            source: InputSource = "unknown"
        elif sources == {"human"}:
            source = "human"
        elif sources == {"playmind_generated"}:
            source = "playmind_generated"
        else:
            source = "unknown"
        return InputSnapshot(physical_events=events, input_source=source)


__all__ = [
    "InputSnapshot",
    "PhysicalInputSynchronizer",
    "events_between",
    "partition_events_by_source",
]
