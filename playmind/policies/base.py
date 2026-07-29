"""High-level policy interface for Learning Architecture V2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass
class PolicyDecision:
    """Result of choosing a skill under an allowed-skill mask."""

    skill: str
    confidence: float
    reason: str
    model_version: str | None = None
    allowed_skills: list[str] = field(default_factory=list)
    used_fallback: bool = False
    temporal_summary: str | None = None
    debug_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "confidence": self.confidence,
            "reason": self.reason,
            "model_version": self.model_version,
            "allowed_skills": list(self.allowed_skills),
            "used_fallback": self.used_fallback,
            "temporal_summary": self.temporal_summary,
            "debug_scores": dict(self.debug_scores),
        }


@runtime_checkable
class HighLevelPolicy(Protocol):
    """Selects a reusable skill rather than a raw key each tick."""

    def choose_skill(
        self,
        context: Mapping[str, Any],
        allowed_skills: Sequence[str],
    ) -> PolicyDecision:
        ...
