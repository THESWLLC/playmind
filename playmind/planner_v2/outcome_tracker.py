"""Plan and skill outcome accounting for evaluation and preferences."""

from __future__ import annotations

import time
import uuid
from typing import Any, Mapping

from .contract import Plan


class OutcomeTracker:
    """Keep plan traces and derive deterministic preference scores."""

    def __init__(self) -> None:
        self.active: dict[str, Any] | None = None
        self.completed: list[dict[str, Any]] = []

    def start_plan(
        self,
        plan: Plan,
        *,
        state: Mapping[str, Any] | Any | None = None,
        source: str = "llm",
        plan_id: str | None = None,
    ) -> str:
        if self.active is not None:
            self.end_plan("replaced")
        identifier = plan_id or uuid.uuid4().hex
        state_data = (
            state.to_dict()
            if state is not None and hasattr(state, "to_dict")
            else dict(state)
            if isinstance(state, Mapping)
            else state
        )
        self.active = {
            "plan_id": identifier,
            "plan": plan.to_dict(),
            "state": state_data,
            "source": str(source),
            "started_at": time.time(),
            "ended_at": None,
            "outcome": None,
            "skill_outcomes": [],
            "metrics": {},
            "score": None,
        }
        return identifier

    record_plan_start = start_plan

    def record_skill_outcome(
        self,
        skill_name: str,
        outcome: str,
        *,
        reward: float = 0.0,
        elapsed_seconds: float | None = None,
        reason: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.active is None:
            raise RuntimeError("no active plan")
        record = {
            "skill": str(skill_name),
            "outcome": str(outcome),
            "reward": float(reward),
            "elapsed_seconds": (
                None if elapsed_seconds is None else float(elapsed_seconds)
            ),
            "reason": str(reason),
            "metadata": dict(metadata or {}),
            "recorded_at": time.time(),
        }
        self.active["skill_outcomes"].append(record)
        return record

    def score(self, trace: Mapping[str, Any] | None = None) -> float:
        item = trace or self.active
        if item is None:
            return 0.0
        skills = list(item.get("skill_outcomes") or [])
        total = max(1, len(skills))
        successes = sum(
            1 for skill in skills if str(skill.get("outcome")).lower() == "success"
        )
        failures = sum(
            1
            for skill in skills
            if str(skill.get("outcome")).lower()
            in {"failed", "failure", "timeout", "cancelled"}
        )
        reward = sum(float(skill.get("reward") or 0.0) for skill in skills)
        outcome = str(item.get("outcome") or "").lower()
        outcome_bonus = {
            "success": 1.0,
            "complete": 1.0,
            "completed": 1.0,
            "death": -1.0,
            "emergency_stop": -1.0,
            "failed": -0.5,
            "replaced": -0.1,
        }.get(outcome, 0.0)
        metrics = item.get("metrics") or {}
        objective_delta = float(metrics.get("objective_progress_delta") or 0.0)
        return round(
            outcome_bonus
            + successes / total
            - 0.5 * failures / total
            + 0.25 * reward
            + objective_delta,
            6,
        )

    def end_plan(
        self,
        outcome: str,
        *,
        metrics: Mapping[str, Any] | None = None,
        score: float | None = None,
    ) -> dict[str, Any]:
        if self.active is None:
            raise RuntimeError("no active plan")
        self.active["ended_at"] = time.time()
        self.active["outcome"] = str(outcome)
        self.active["metrics"] = dict(metrics or {})
        self.active["score"] = (
            float(score) if score is not None else self.score(self.active)
        )
        finished = self.active
        self.completed.append(finished)
        self.active = None
        return finished

    record_plan_end = end_plan

    def preference_pairs(self, *, min_score_gap: float = 0.0) -> list[dict[str, Any]]:
        """Pair plans for the same goal, preferring the higher-scored trace."""
        pairs: list[dict[str, Any]] = []
        for left_index, left in enumerate(self.completed):
            left_goal = str((left.get("plan") or {}).get("goal") or "")
            for right in self.completed[left_index + 1 :]:
                right_goal = str((right.get("plan") or {}).get("goal") or "")
                if left_goal != right_goal:
                    continue
                left_score = float(left.get("score") or 0.0)
                right_score = float(right.get("score") or 0.0)
                if abs(left_score - right_score) <= float(min_score_gap):
                    continue
                chosen, rejected = (
                    (left, right) if left_score > right_score else (right, left)
                )
                pairs.append(
                    {
                        "state": chosen.get("state") or rejected.get("state"),
                        "chosen": chosen["plan"],
                        "rejected": rejected["plan"],
                        "chosen_score": float(chosen["score"]),
                        "rejected_score": float(rejected["score"]),
                    }
                )
        return pairs

    def snapshot(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "completed": list(self.completed),
            "preference_pairs": self.preference_pairs(),
        }


PlanOutcomeTracker = OutcomeTracker

__all__ = ["OutcomeTracker", "PlanOutcomeTracker"]
