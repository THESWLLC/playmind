"""Rolling temporal history for Learning Architecture V2."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional, Sequence

from playmind.observations import Observation

HISTORY_MAXLEN = 16


@dataclass
class TemporalSummary:
    """Compact features derived from the last N ticks."""

    health_trend: float = 0.0
    target_health_trend: float = 0.0
    motion_trend: float = 0.0
    repeated_action_count: int = 0
    no_progress_duration: float = 0.0
    target_flicker_count: int = 0
    combat_flicker_count: int = 0
    recent_damage_dealt_est: float = 0.0
    recent_damage_received_est: float = 0.0
    current_skill_duration: float = 0.0
    recent_sensor_disagreement: int = 0


def _trend(values: Sequence[Optional[float]]) -> float:
    """Signed change from first known to last known value (0 if <2 points)."""
    known = [float(v) for v in values if v is not None]
    if len(known) < 2:
        return 0.0
    return known[-1] - known[0]


def _flicker_count(flags: Sequence[Optional[bool]]) -> int:
    """Count True↔False transitions; None breaks the chain (no flicker)."""
    count = 0
    prev: Optional[bool] = None
    for flag in flags:
        if flag is None:
            prev = None
            continue
        if prev is not None and flag != prev:
            count += 1
        prev = flag
    return count


def _damage_deltas(
    values: Sequence[Optional[float]],
) -> tuple[float, float]:
    """Return (drops_sum, rises_sum) across consecutive known pairs."""
    drops = 0.0
    rises = 0.0
    prev: Optional[float] = None
    for v in values:
        if v is None:
            prev = None
            continue
        cur = float(v)
        if prev is not None:
            delta = cur - prev
            if delta < -1e-9:
                drops += -delta
            elif delta > 1e-9:
                rises += delta
        prev = cur
    return drops, rises


@dataclass
class TemporalHistory:
    """Bounded rolling window of observations, actions, rewards, and timing."""

    maxlen: int = HISTORY_MAXLEN
    observations: Deque[Observation] = field(init=False)
    requested_actions: Deque[Optional[str]] = field(init=False)
    executed_actions: Deque[Optional[str]] = field(init=False)
    rewards: Deque[float] = field(init=False)
    outcomes: Deque[Any] = field(init=False)
    dt_seconds: Deque[float] = field(init=False)

    def __post_init__(self) -> None:
        n = max(1, int(self.maxlen))
        self.maxlen = n
        self.observations = deque(maxlen=n)
        self.requested_actions = deque(maxlen=n)
        self.executed_actions = deque(maxlen=n)
        self.rewards = deque(maxlen=n)
        self.outcomes = deque(maxlen=n)
        self.dt_seconds = deque(maxlen=n)

    def __len__(self) -> int:
        return len(self.observations)

    def clear(self) -> None:
        self.observations.clear()
        self.requested_actions.clear()
        self.executed_actions.clear()
        self.rewards.clear()
        self.outcomes.clear()
        self.dt_seconds.clear()

    def push(
        self,
        observation: Observation,
        *,
        requested_action: Optional[str] = None,
        executed_action: Optional[str] = None,
        reward: float = 0.0,
        outcome: Any = None,
        dt_seconds: float = 0.0,
    ) -> None:
        """Append one tick; older entries fall off when maxlen is exceeded."""
        self.observations.append(observation)
        self.requested_actions.append(requested_action)
        self.executed_actions.append(executed_action)
        self.rewards.append(float(reward))
        self.outcomes.append(outcome)
        self.dt_seconds.append(float(dt_seconds))

    def summarize(self) -> TemporalSummary:
        obs_list = list(self.observations)
        if not obs_list:
            return TemporalSummary()

        player_hps = [o.player_hp for o in obs_list]
        target_hps = [o.target_hp for o in obs_list]
        motions = [o.motion for o in obs_list]
        has_targets = [o.has_target for o in obs_list]
        in_combats = [o.in_combat for o in obs_list]

        health_trend = _trend(player_hps)
        target_health_trend = _trend(target_hps)
        motion_trend = _trend(motions)

        target_flicker_count = _flicker_count(has_targets)
        combat_flicker_count = _flicker_count(in_combats)

        target_drops, _ = _damage_deltas(target_hps)
        player_drops, _ = _damage_deltas(player_hps)
        recent_damage_dealt_est = target_drops
        recent_damage_received_est = player_drops

        # Trailing streak of the same non-None executed action.
        repeated_action_count = 0
        actions = list(self.executed_actions)
        if actions:
            last = actions[-1]
            if last is not None:
                for a in reversed(actions):
                    if a == last:
                        repeated_action_count += 1
                    else:
                        break

        # Accumulate dt while stagnating / no motion / no damage events this step.
        no_progress_duration = 0.0
        dts = list(self.dt_seconds)
        for i, o in enumerate(obs_list):
            dt = dts[i] if i < len(dts) else 0.0
            stagnant = o.stagnation_count > 0
            low_motion = o.motion is not None and o.motion < 2.0
            no_dmg = True
            if i > 0:
                prev = obs_list[i - 1]
                if (
                    prev.target_hp is not None
                    and o.target_hp is not None
                    and o.target_hp < prev.target_hp - 0.02
                ):
                    no_dmg = False
                if (
                    prev.player_hp is not None
                    and o.player_hp is not None
                    and abs(o.player_hp - prev.player_hp) > 0.02
                ):
                    no_dmg = False
            if stagnant or (low_motion and no_dmg):
                no_progress_duration += max(0.0, dt)

        recent_sensor_disagreement = sum(len(o.sensor_warnings) for o in obs_list)

        return TemporalSummary(
            health_trend=health_trend,
            target_health_trend=target_health_trend,
            motion_trend=motion_trend,
            repeated_action_count=repeated_action_count,
            no_progress_duration=no_progress_duration,
            target_flicker_count=target_flicker_count,
            combat_flicker_count=combat_flicker_count,
            recent_damage_dealt_est=recent_damage_dealt_est,
            recent_damage_received_est=recent_damage_received_est,
            current_skill_duration=0.0,
            recent_sensor_disagreement=recent_sensor_disagreement,
        )
