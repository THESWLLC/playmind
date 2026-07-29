"""Offline replay environment — evaluate policies on saved demos without actuators."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from playmind.demonstrations import load_session_samples
from playmind.policies.base import PolicyDecision
from playmind.policies.scripted import DEFAULT_SKILL_ORDER, ScriptedPolicy
from playmind.skills.base import SkillContext, SkillStepResult


@dataclass
class ReplayStepResult:
    """One offline step: observation fed to policy, decision recorded (no actuation)."""

    index: int
    observation: dict[str, Any]
    decision: PolicyDecision
    sample: dict[str, Any]
    allowed_skills: list[str] = field(default_factory=list)
    runtime_result: SkillStepResult | None = None
    runtime_snapshot: dict[str, Any] | None = None
    runtime_error: str | None = None
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "observation": dict(self.observation),
            "decision": self.decision.to_dict(),
            "sample_id": self.sample.get("sample_id"),
            "episode_id": self.sample.get("episode_id"),
            "skill_label": self.sample.get("skill"),
            "allowed_skills": list(self.allowed_skills),
            "runtime_result": (
                {
                    "requested_action": self.runtime_result.requested_action,
                    "reason": self.runtime_result.reason,
                    "status": self.runtime_result.status,
                    "success_evidence": list(self.runtime_result.success_evidence),
                    "failure_evidence": list(self.runtime_result.failure_evidence),
                }
                if self.runtime_result is not None
                else None
            ),
            "runtime_snapshot": dict(self.runtime_snapshot or {}),
            "runtime_error": self.runtime_error,
            "done": self.done,
        }


@dataclass
class ReplayEnv:
    """Feed saved observation transitions through ``policy.choose_skill``.

    Does **not** call actuators, capture, or the live game — CPU-only offline eval.
    """

    samples: list[dict[str, Any]]
    policy: Any = field(default_factory=ScriptedPolicy)
    allowed_skills: Sequence[str] = field(default_factory=lambda: list(DEFAULT_SKILL_ORDER))
    context_builder: Optional[Callable[[dict[str, Any], dict[str, Any]], Mapping[str, Any]]] = None
    skill_runtime: Any | None = None
    runtime: Any | None = None  # compatibility alias for skill_runtime
    _index: int = field(default=0, init=False)
    last_decision: PolicyDecision | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.skill_runtime is None:
            self.skill_runtime = self.runtime
        elif self.runtime is not None and self.runtime is not self.skill_runtime:
            raise ValueError("runtime and skill_runtime must refer to the same runtime")

    @classmethod
    def from_session(
        cls,
        session_dir: str | Path,
        policy: Any | None = None,
        *,
        allowed_skills: Sequence[str] | None = None,
        skill_runtime: Any | None = None,
        runtime: Any | None = None,
    ) -> "ReplayEnv":
        samples = load_session_samples(session_dir)
        session_path = Path(session_dir) / "session.json"
        session_meta: dict[str, Any] = {}
        if session_path.exists():
            try:
                loaded = json.loads(session_path.read_text(encoding="utf-8"))
                if isinstance(loaded, Mapping):
                    session_meta = dict(loaded)
            except (OSError, ValueError):
                session_meta = {}
        for sample in samples:
            sample.setdefault("_session_outcome", session_meta.get("outcome"))
            sample.setdefault("_session_metadata", session_meta)
        return cls(
            samples=samples,
            policy=policy if policy is not None else ScriptedPolicy(),
            allowed_skills=list(allowed_skills) if allowed_skills is not None else list(DEFAULT_SKILL_ORDER),
            skill_runtime=skill_runtime if skill_runtime is not None else runtime,
        )

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[Mapping[str, Any]],
        policy: Any | None = None,
        *,
        allowed_skills: Sequence[str] | None = None,
        skill_runtime: Any | None = None,
        runtime: Any | None = None,
    ) -> "ReplayEnv":
        """Build a replay env from in-memory sample dicts (synthetic scenarios)."""
        return cls(
            samples=[dict(s) for s in samples],
            policy=policy if policy is not None else ScriptedPolicy(),
            allowed_skills=list(allowed_skills)
            if allowed_skills is not None
            else list(DEFAULT_SKILL_ORDER),
            skill_runtime=skill_runtime if skill_runtime is not None else runtime,
        )

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def done(self) -> bool:
        return self._index >= len(self.samples)

    def reset(self) -> dict[str, Any] | None:
        """Reset cursor to the first sample; return its observation (or None if empty)."""
        self._index = 0
        self.last_decision = None
        reset_state = getattr(self.policy, "reset_state", None)
        if callable(reset_state):
            reset_state()
        primary_reset = getattr(getattr(self.policy, "primary", None), "reset_state", None)
        if callable(primary_reset) and primary_reset is not reset_state:
            primary_reset()
        if self.skill_runtime is not None:
            clear = getattr(self.skill_runtime, "clear", None)
            if callable(clear):
                clear()
        if not self.samples:
            return None
        return dict(self.samples[0].get("observation") or {})

    def _build_context(self, sample: dict[str, Any]) -> Mapping[str, Any]:
        obs = dict(sample.get("observation") or {})
        if self.context_builder is not None:
            return self.context_builder(obs, sample)
        return {
            "obs": obs,
            "goal": sample.get("goal"),
            "episode_id": sample.get("episode_id"),
            "key_events": list(sample.get("key_events") or []),
            "profile": sample.get("profile"),
            "demo_skill": sample.get("skill"),
            "sample": sample,
            "replay_index": self._index,
        }

    def _allowed_for_sample(self, sample: Mapping[str, Any]) -> list[str]:
        per_sample = sample.get("allowed_skills") or sample.get("valid_skills")
        source = per_sample if isinstance(per_sample, Sequence) and not isinstance(
            per_sample, (str, bytes)
        ) else self.allowed_skills
        return list(dict.fromkeys(str(skill) for skill in source))

    @staticmethod
    def _coerce_decision(value: Any, allowed: Sequence[str]) -> PolicyDecision:
        if isinstance(value, PolicyDecision):
            return value
        if isinstance(value, Mapping):
            return PolicyDecision(
                skill=str(value.get("skill") or "wait"),
                confidence=float(value.get("confidence") or 0.0),
                reason=str(value.get("reason") or "mapping policy decision"),
                model_version=value.get("model_version"),
                allowed_skills=list(value.get("allowed_skills") or allowed),
                used_fallback=bool(value.get("used_fallback")),
                temporal_summary=value.get("temporal_summary"),
                debug_scores=dict(value.get("debug_scores") or {}),
            )
        raise TypeError(f"choose_skill must return PolicyDecision or mapping, got {type(value)!r}")

    def _step_runtime(
        self,
        decision: PolicyDecision,
        sample: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> tuple[SkillStepResult | None, dict[str, Any] | None, str | None]:
        """Dry-step an optional SkillRuntime; requested actions are only recorded."""
        if self.skill_runtime is None:
            return None, None, None
        try:
            timestamp = sample.get("timestamp", self._index)
            try:
                now = float(timestamp)
            except (TypeError, ValueError):
                now = float(self._index)
            ctx = SkillContext(
                obs=dict(observation),
                tick=self._index,
                goal=str(sample.get("goal") or ""),
                now=now,
                meta={"offline_replay": True, "dry_run": True},
            )
            if getattr(self.skill_runtime, "active_name", None) != decision.skill:
                self.skill_runtime.start(decision.skill, ctx)
            result = self.skill_runtime.step(ctx)
            snapshot_fn = getattr(self.skill_runtime, "snapshot", None)
            snapshot = snapshot_fn() if callable(snapshot_fn) else None
            return result, snapshot, None
        except Exception as exc:
            # A bad/unknown proposal is evaluation evidence, not a replay crash.
            return None, None, f"{type(exc).__name__}: {exc}"

    def step(self) -> ReplayStepResult:
        """Advance one sample: choose_skill only — never actuate."""
        if self.done:
            raise StopIteration("ReplayEnv exhausted; call reset()")
        sample = self.samples[self._index]
        obs = dict(sample.get("observation") or {})
        ctx = self._build_context(sample)
        allowed = self._allowed_for_sample(sample)
        decision = self._coerce_decision(
            self.policy.choose_skill(ctx, allowed),
            allowed,
        )
        runtime_result, runtime_snapshot, runtime_error = self._step_runtime(
            decision, sample, obs
        )
        self.last_decision = decision
        self._index += 1
        return ReplayStepResult(
            index=self._index - 1,
            observation=obs,
            decision=decision,
            sample=sample,
            allowed_skills=allowed,
            runtime_result=runtime_result,
            runtime_snapshot=runtime_snapshot,
            runtime_error=runtime_error,
            done=self.done,
        )

    def run(self) -> list[ReplayStepResult]:
        """Reset and replay the full session; returns all step results."""
        self.reset()
        results: list[ReplayStepResult] = []
        while not self.done:
            results.append(self.step())
        return results

    def iter_steps(self) -> Iterator[ReplayStepResult]:
        self.reset()
        while not self.done:
            yield self.step()

    def agreement_rate(self, results: Sequence[ReplayStepResult] | None = None) -> float:
        """Fraction of steps where policy skill matches demo ``skill`` label (if present)."""
        rows = list(results) if results is not None else self.run()
        labeled = [r for r in rows if r.sample.get("skill")]
        if not labeled:
            return 0.0
        hits = sum(1 for r in labeled if r.decision.skill == r.sample.get("skill"))
        return hits / float(len(labeled))

    def set_policy(self, policy: Any) -> None:
        """Swap the policy and reset the replay cursor."""
        self.policy = policy
        self.reset()


def compare_policies(
    samples: Sequence[Mapping[str, Any]],
    policies: Mapping[str, Any],
    *,
    allowed_skills: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run identical samples through policies and return sectioned comparisons."""
    from playmind.evaluation.metrics import summarize_replay_results

    out: dict[str, dict[str, Any]] = {}
    for name, policy in policies.items():
        env = ReplayEnv.from_samples(
            samples, policy=policy, allowed_skills=allowed_skills
        )
        results = env.run()
        out[name] = summarize_replay_results(results, policy_name=name)
        out[name]["skills"] = [r.decision.skill for r in results]
    return out
