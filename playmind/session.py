"""Long-run session scheduler for owned-game agents."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field


@dataclass
class SessionConfig:
    play_minutes: float = 180.0
    break_minutes_min: float = 10.0
    break_minutes_max: float = 40.0
    max_wall_hours: float = 14.0
    logout_action: str = "wait"  # map to your game's logout hotkey via keymap later


@dataclass
class SessionScheduler:
    config: SessionConfig = field(default_factory=SessionConfig)
    started_at: float = field(default_factory=time.time)
    segment_started_at: float = field(default_factory=time.time)
    on_break: bool = False
    break_until: float = 0.0
    segments_completed: int = 0

    def wall_hours(self) -> float:
        return (time.time() - self.started_at) / 3600.0

    def should_stop(self) -> bool:
        return self.wall_hours() >= self.config.max_wall_hours

    def should_start_break(self) -> bool:
        if self.on_break:
            return False
        played = (time.time() - self.segment_started_at) / 60.0
        return played >= self.config.play_minutes

    def start_break(self) -> float:
        mins = random.uniform(self.config.break_minutes_min, self.config.break_minutes_max)
        self.on_break = True
        self.break_until = time.time() + mins * 60.0
        self.segments_completed += 1
        return mins

    def break_done(self) -> bool:
        return self.on_break and time.time() >= self.break_until

    def end_break(self) -> None:
        self.on_break = False
        self.break_until = 0.0
        self.segment_started_at = time.time()

    def status(self) -> dict:
        return {
            "wall_hours": round(self.wall_hours(), 3),
            "on_break": self.on_break,
            "segments_completed": self.segments_completed,
            "should_stop": self.should_stop(),
        }
