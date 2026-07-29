"""Offline replay environment — evaluate policies on saved demos without actuators."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from playmind.demonstrations import load_session_samples
from playmind.policies.base import PolicyDecision
from playmind.policies.scripted import DEFAULT_SKILL_ORDER, ScriptedPolicy


@dataclass
class ReplayStepResult:
    """One offline step: observation fed to policy, decision recorded (no actuation)."""

    index: int
    observation: dict[str, Any]
    decision: PolicyDecision
    sample: dict[str, Any]
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "observation": dict(self.observation),
            "decision": self.decision.to_dict(),
            "sample_id": self.sample.get("sample_id"),
            "episode_id": self.sample.get("episode_id"),
            "skill_label": self.sample.get("skill"),
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
    _index: int = field(default=0, init=False)
    last_decision: PolicyDecision | None = field(default=None, init=False)

    @classmethod
    def from_session(
        cls,
        session_dir: str | Path,
        policy: Any | None = None,
        *,
        allowed_skills: Sequence[str] | None = None,
    ) -> "ReplayEnv":
        samples = load_session_samples(session_dir)
        return cls(
            samples=samples,
            policy=policy if policy is not None else ScriptedPolicy(),
            allowed_skills=list(allowed_skills) if allowed_skills is not None else list(DEFAULT_SKILL_ORDER),
        )

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[Mapping[str, Any]],
        policy: Any | None = None,
        *,
        allowed_skills: Sequence[str] | None = None,
    ) -> "ReplayEnv":
        """Build a replay env from in-memory sample dicts (synthetic scenarios)."""
        return cls(
            samples=[dict(s) for s in samples],
            policy=policy if policy is not None else ScriptedPolicy(),
            allowed_skills=list(allowed_skills)
            if allowed_skills is not None
            else list(DEFAULT_SKILL_ORDER),
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
        }

    def step(self) -> ReplayStepResult:
        """Advance one sample: choose_skill only — never actuate."""
        if self.done:
            raise StopIteration("ReplayEnv exhausted; call reset()")
        sample = self.samples[self._index]
        obs = dict(sample.get("observation") or {})
        ctx = self._build_context(sample)
        decision = self.policy.choose_skill(ctx, list(self.allowed_skills))
        self.last_decision = decision
        self._index += 1
        return ReplayStepResult(
            index=self._index - 1,
            observation=obs,
            decision=decision,
            sample=sample,
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
    """Run the same sample sequence through multiple policies; return per-name stats."""
    out: dict[str, dict[str, Any]] = {}
    for name, policy in policies.items():
        env = ReplayEnv.from_samples(
            samples, policy=policy, allowed_skills=allowed_skills
        )
        results = env.run()
        out[name] = {
            "agreement_rate": env.agreement_rate(results),
            "n_steps": len(results),
            "skills": [r.decision.skill for r in results],
        }
    return out
