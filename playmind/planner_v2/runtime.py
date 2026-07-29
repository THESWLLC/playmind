"""Event-driven orchestration for Planner V2."""

from __future__ import annotations

import inspect
import time
import urllib.error
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from playmind.policies.scripted import ScriptedPolicy

from .contract import ALLOWED_REPLAN_EVENTS, Plan, SkillStep
from .modes import PlannerMode, can_send_input
from .ollama_client import generate_plan
from .outcome_tracker import OutcomeTracker
from .plan_executor import PlanExecutor
from .plan_validator import PlanValidator
from .state_builder import build_planner_state

_CALL_EVENTS = frozenset(ALLOWED_REPLAN_EVENTS)
_EVENT_ALIASES = {
    "objective_new": "new_objective",
    "complete": "plan_complete",
    "completed": "plan_complete",
    "skill_failed": "skill_fail",
    "skill_failure": "skill_fail",
    "critical_health": "health_critical",
    "dead": "death",
    "is_dead": "death",
    "is_ghost": "ghost",
    "recovered": "controllable_after_recovery",
    "stuck": "severe_stuck",
    "invalid_target": "target_invalid",
    "progress_changed": "objective_progress_change",
    "expired": "plan_expiry",
    "blocking_modal": "modal",
    "periodic": "periodic_interval",
}
_FAILURE_STATUSES = frozenset({"failed", "failure", "timeout", "cancelled"})


def _event_name(value: Any) -> str:
    name = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _EVENT_ALIASES.get(name, name)


class PlannerV2Runtime:
    """Call the planner at meaningful boundaries and dispatch skill names."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        planner: Any = None,
        execute_skill: Callable[[str], Any] | None = None,
        validator: PlanValidator | None = None,
        executor: PlanExecutor | None = None,
        outcome_tracker: OutcomeTracker | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        full = dict(config or {})
        section = full.get("planner_v2")
        self.config = dict(section) if isinstance(section, Mapping) else full
        self.mode = PlannerMode.parse(self.config.get("mode", "shadow"))
        self.model = str(
            self.config.get("model")
            or self.config.get("production_model")
            or "llama3.2"
        )
        self.host = str(
            self.config.get("host") or "http://127.0.0.1:11434"
        ).rstrip("/")
        self.timeout = float(
            self.config.get(
                "timeout",
                self.config.get(
                    "timeout_s",
                    self.config.get("timeout_seconds", 60.0),
                ),
            )
        )
        self.periodic_interval = max(
            1.0,
            float(
                self.config.get(
                    "periodic_interval",
                    self.config.get(
                        "periodic_interval_seconds",
                        self.config.get("periodic_replan_seconds", 30.0),
                    ),
                )
            ),
        )
        self.minimum_confidence = float(self.config.get("minimum_confidence", 0.0))
        self.health_critical_threshold = float(
            self.config.get("health_critical_threshold", 0.15)
        )
        self.objective_progress_epsilon = max(
            0.0, float(self.config.get("objective_progress_epsilon", 1e-6))
        )
        self.auth_flags: dict[str, Any] = {
            "i_own_this_game": bool(
                self.config.get(
                    "i_own_this_game", full.get("i_own_this_game", False)
                )
            ),
            "enable_keyboard": bool(
                self.config.get(
                    "enable_keyboard", full.get("enable_keyboard", False)
                )
            ),
        }
        self.planner = planner
        self.execute_skill_callback = execute_skill
        self.validator = validator or PlanValidator(
            max_plan_length=int(
                self.config.get(
                    "max_plan_length",
                    self.config.get("maximum_plan_skills", 5),
                )
            )
        )
        self.clock = clock
        self.executor = executor or PlanExecutor(clock=clock)
        self.outcomes = outcome_tracker or OutcomeTracker()
        self.scripted = ScriptedPolicy()

        self.emergency_stop = False
        self.awaiting_approval = False
        self.last_state: Any = None
        self.last_validation: Any = None
        self.last_error = ""
        self.last_plan_source = ""
        self.last_trigger: str | None = None
        self.last_dispatched_skill: str | None = None
        self.last_latency_seconds: float | None = None
        self._last_dispatched_index: int | None = None
        self._last_planner_call: float | None = float(self.clock())
        self._last_goal: str | None = None
        self._last_objective_progress: float | None = None
        self._condition_edges: dict[str, bool] = {}

    def set_emergency_stop(self, active: bool = True) -> None:
        self.emergency_stop = bool(active)

    def approve_plan(self, approved: bool = True) -> None:
        self.awaiting_approval = not bool(approved)
        self.auth_flags["plan_approved"] = bool(approved)

    def _context_events(self, context: Mapping[str, Any]) -> set[str]:
        events: set[str] = set()
        goal = context.get("goal")
        if goal is not None:
            goal_text = str(goal)
            if self._last_goal is not None and goal_text != self._last_goal:
                events.add("new_objective")
            elif self._last_goal is None and goal_text:
                events.add("new_objective")
            self._last_goal = goal_text

        status = str(
            context.get("skill_status")
            or context.get("recent_action_outcome")
            or ""
        ).lower()
        conditions: dict[str, bool] = {
            "plan_complete": self.executor.complete,
            "skill_fail": status in _FAILURE_STATUSES
            or bool(context.get("skill_fail") or context.get("skill_failed")),
            "death": bool(context.get("is_dead")),
            "ghost": bool(context.get("is_ghost"))
            or str(context.get("life_phase") or "").lower() == "ghost",
            "controllable_after_recovery": bool(
                context.get("controllable_after_recovery")
            ),
            "severe_stuck": bool(context.get("severe_stuck"))
            or (
                str(context.get("stuck_hint") or "").lower()
                not in {"", "none", "false", "0"}
                and int(
                    context.get("stagnation_count")
                    or context.get("stagnant")
                    or 0
                )
                >= 8
            ),
            "target_invalid": bool(
                context.get("target_invalid")
                or context.get("target_veto")
                or context.get("invalid_target")
            ),
            "modal": bool(
                context.get("blocking_modal") or context.get("modal_menu")
            ),
            "plan_expiry": self.executor.is_expired(),
        }
        hp = context.get("vision_player_hp", context.get("player_hp"))
        if hp is None and isinstance(context.get("player"), Mapping):
            hp = context["player"].get("hp")
        try:
            conditions["health_critical"] = (
                hp is not None and float(hp) <= self.health_critical_threshold
            )
        except (TypeError, ValueError):
            conditions["health_critical"] = False

        for name, active in conditions.items():
            if active and not self._condition_edges.get(name, False):
                events.add(name)
            self._condition_edges[name] = active

        progress = context.get("objective_progress")
        if progress is not None:
            try:
                numeric_progress = float(progress)
            except (TypeError, ValueError):
                numeric_progress = None
            if numeric_progress is not None:
                if (
                    self._last_objective_progress is not None
                    and abs(numeric_progress - self._last_objective_progress)
                    > self.objective_progress_epsilon
                ):
                    events.add("objective_progress_change")
                self._last_objective_progress = numeric_progress
        return events

    def should_call_planner(
        self,
        event: Any = None,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Return true only at event boundaries or the periodic interval."""
        if self.mode is PlannerMode.OBSERVE or self.emergency_stop:
            return False
        now = float(self.clock())
        explicit: set[str] = set()
        if isinstance(event, str):
            explicit.add(_event_name(event))
        elif isinstance(event, Mapping):
            explicit.update(
                _event_name(name) for name, active in event.items() if active
            )
        elif event is not None:
            try:
                explicit.update(_event_name(item) for item in event)
            except TypeError:
                explicit.add(_event_name(event))
        derived = self._context_events(context or {})
        if (explicit | derived) & _CALL_EVENTS:
            return True
        return self._last_planner_call is None or (
            now - self._last_planner_call >= self.periodic_interval
        )

    def _invoke_planner(self, state: Any) -> str:
        if self.planner is None:
            return generate_plan(
                state,
                self.model,
                host=self.host,
                timeout=self.timeout,
            )
        fn = getattr(self.planner, "generate_plan", self.planner)
        if not callable(fn):
            raise TypeError("planner must be callable or expose generate_plan")
        kwargs = {
            "model": self.model,
            "host": self.host,
            "timeout": self.timeout,
        }
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return str(fn(state, **kwargs))
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        supported = (
            kwargs
            if accepts_kwargs
            else {name: value for name, value in kwargs.items() if name in signature.parameters}
        )
        return str(fn(state, **supported))

    def _fallback_plan(
        self,
        obs: Mapping[str, Any],
        *,
        goal: str,
        allowed_skills: Sequence[str],
        reason: str,
    ) -> Plan:
        allowed = list(dict.fromkeys(str(skill) for skill in allowed_skills))
        if not allowed:
            raise ValueError("cannot build fallback plan without available skills")
        decision = self.scripted.choose_skill(
            {"obs": obs, "goal": goal, "stuck": bool(obs.get("severe_stuck"))},
            allowed,
        )
        skill_name = (
            decision.skill if decision.skill in allowed else allowed[0]
        )
        timeout = int(
            max(
                1,
                min(
                    120,
                    float(
                        (self.config.get("skill_timeouts") or {}).get(
                            skill_name, 30
                        )
                    ),
                ),
            )
        )
        return Plan(
            goal=str(goal),
            skills=[
                SkillStep(
                    name=skill_name,
                    until=None,
                    max_seconds=timeout,
                    constraints={},
                )
            ],
            replan_on=[
                "skill_fail",
                "death",
                "ghost",
                "health_critical",
                "severe_stuck",
                "plan_expiry",
            ],
            confidence=float(getattr(decision, "confidence", 0.25)),
            reason_code="heuristic_fallback",
            summary=f"Scripted fallback after {reason}",
        )

    def request_plan(
        self,
        obs: Mapping[str, Any],
        *,
        goal: str,
        profile: Any = None,
        available_skills: Sequence[str],
        current_skill: Any = None,
        recent_skills: Sequence[Any] = (),
        previous_plan: Plan | Mapping[str, Any] | None = None,
        memory: Any = None,
        game_id: str = "",
        trigger: str = "periodic_interval",
    ) -> Plan | None:
        if self.mode is PlannerMode.OBSERVE or self.emergency_stop:
            return None
        state = build_planner_state(
            obs,
            goal=goal,
            profile=profile,
            available_skills=available_skills,
            current_skill=current_skill,
            recent_skills=recent_skills,
            previous_plan=previous_plan or self.executor.plan,
            memory=memory,
            game_id=game_id,
        )
        self.last_state = state
        self.last_trigger = _event_name(trigger)
        self._last_planner_call = float(self.clock())
        source = "llm"
        started_at = float(self.clock())
        try:
            raw = self._invoke_planner(state)
            validation = self.validator.validate_or_parse(raw, available_skills)
            self.last_validation = validation
            if not validation.ok or validation.plan is None:
                reason = "; ".join(validation.errors) or "validation failed"
                self.last_error = reason
                plan = self._fallback_plan(
                    obs,
                    goal=goal,
                    allowed_skills=available_skills,
                    reason=reason,
                )
                source = "heuristic_fallback"
            else:
                plan = validation.plan
                if float(plan.confidence) < self.minimum_confidence:
                    reason = (
                        f"plan confidence {plan.confidence:.3f} is below "
                        f"{self.minimum_confidence:.3f}"
                    )
                    self.last_error = reason
                    plan = self._fallback_plan(
                        obs,
                        goal=goal,
                        allowed_skills=available_skills,
                        reason=reason,
                    )
                    source = "heuristic_fallback"
                else:
                    self.last_error = ""
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
            TypeError,
        ) as exc:
            self.last_error = str(exc)
            plan = self._fallback_plan(
                obs,
                goal=goal,
                allowed_skills=available_skills,
                reason=f"{type(exc).__name__}: {exc}",
            )
            source = "heuristic_fallback"
        finally:
            self.last_latency_seconds = max(0.0, float(self.clock()) - started_at)

        fallback_validation = self.validator.validate(plan, available_skills)
        if not fallback_validation.ok:
            self.last_validation = fallback_validation
            self.last_error = "; ".join(fallback_validation.errors)
            return None
        self.executor.set_plan(plan)
        self._last_dispatched_index = None
        self.last_dispatched_skill = None
        self.last_plan_source = source
        self.awaiting_approval = self.mode is PlannerMode.ASSIST
        self.auth_flags["plan_approved"] = not self.awaiting_approval
        self.outcomes.start_plan(plan, state=state, source=source)
        return plan

    def execute_current(
        self,
        auth_flags: Mapping[str, Any] | None = None,
    ) -> str | None:
        if self.emergency_stop or self.execute_skill_callback is None:
            return None
        effective = dict(self.auth_flags)
        effective.update(dict(auth_flags or {}))
        if self.awaiting_approval or not can_send_input(self.mode, effective):
            return None
        skill = self.executor.next_skill()
        if skill is None:
            return None
        if self._last_dispatched_index == self.executor.index:
            return skill
        # The callback receives a validated skill name; Learning V2 owns all
        # low-level execution and action masking.
        self.execute_skill_callback(skill)
        self._last_dispatched_index = self.executor.index
        self.last_dispatched_skill = skill
        return skill

    def note_skill_outcome(
        self,
        status: str,
        *,
        reward: float = 0.0,
        elapsed_seconds: float | None = None,
        reason: str = "",
    ) -> str | None:
        skill = self.executor.next_skill()
        if skill is not None and self.outcomes.active is not None:
            self.outcomes.record_skill_outcome(
                skill,
                status,
                reward=reward,
                elapsed_seconds=elapsed_seconds,
                reason=reason,
            )
        if str(status).lower() == "success":
            next_name = self.executor.advance(True)
            self._last_dispatched_index = None
            if next_name is None and self.outcomes.active is not None:
                self.outcomes.end_plan("completed")
            return next_name
        return skill

    def step(
        self,
        obs: Mapping[str, Any],
        *,
        goal: str,
        available_skills: Sequence[str],
        event: Any = None,
        auth_flags: Mapping[str, Any] | None = None,
        **state_context: Any,
    ) -> dict[str, Any]:
        context = dict(obs)
        context["goal"] = goal
        if self.should_call_planner(event, context):
            trigger = _event_name(event) if isinstance(event, str) else "periodic_interval"
            self.request_plan(
                obs,
                goal=goal,
                available_skills=available_skills,
                trigger=trigger,
                **state_context,
            )
        dispatched = self.execute_current(auth_flags)
        result = self.snapshot()
        result["dispatched_skill"] = dispatched
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "emergency_stop": self.emergency_stop,
            "awaiting_approval": self.awaiting_approval,
            "plan_source": self.last_plan_source,
            "trigger": self.last_trigger,
            "error": self.last_error,
            "can_send_input": can_send_input(self.mode, self.auth_flags)
            and not self.emergency_stop
            and not self.awaiting_approval,
            "executor": self.executor.snapshot(),
            "last_dispatched_skill": self.last_dispatched_skill,
            "latency_seconds": self.last_latency_seconds,
        }


__all__ = ["PlannerV2Runtime"]
