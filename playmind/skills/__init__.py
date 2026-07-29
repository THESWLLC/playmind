"""Skill registry and public exports."""

from __future__ import annotations

from typing import Iterable

from playmind.skills.base import Skill, SkillContext, SkillStepResult
from playmind.skills.combat import (
    AcquireTargetSkill,
    ApproachTargetSkill,
    BasicCombatRotationSkill,
    DisengageSkill,
    EngageTargetSkill,
    LootTargetSkill,
    ValidateTargetSkill,
)
from playmind.skills.death import DeathRecoverySkill, GhostRunbackSkill
from playmind.skills.recovery import (
    ClearModalSkill,
    ExploreSkill,
    InteractSkill,
    RecoverHealthSkill,
    UnstuckSkill,
    WaitSkill,
)

__all__ = [
    "Skill",
    "SkillContext",
    "SkillStepResult",
    "SkillRegistry",
    "default_registry",
    "get_skill",
    "list_skills",
    "AcquireTargetSkill",
    "ValidateTargetSkill",
    "ApproachTargetSkill",
    "EngageTargetSkill",
    "BasicCombatRotationSkill",
    "LootTargetSkill",
    "DisengageSkill",
    "RecoverHealthSkill",
    "ExploreSkill",
    "UnstuckSkill",
    "ClearModalSkill",
    "DeathRecoverySkill",
    "GhostRunbackSkill",
    "InteractSkill",
    "WaitSkill",
]


class SkillRegistry:
    """Name → factory for skill instances."""

    def __init__(self) -> None:
        self._factories: dict[str, type[Skill]] = {}

    def register(self, skill_cls: type[Skill]) -> type[Skill]:
        name = getattr(skill_cls, "name", None) or skill_cls.__name__
        self._factories[str(name)] = skill_cls
        return skill_cls

    def get(self, name: str) -> Skill:
        key = str(name).strip()
        if key not in self._factories:
            raise KeyError(f"unknown skill: {name!r}")
        return self._factories[key]()

    def create(self, name: str) -> Skill:
        return self.get(name)

    def names(self) -> list[str]:
        return sorted(self._factories.keys())

    def __contains__(self, name: object) -> bool:
        return str(name) in self._factories

    def __len__(self) -> int:
        return len(self._factories)


_DEFAULT: SkillRegistry | None = None


def default_registry() -> SkillRegistry:
    global _DEFAULT
    if _DEFAULT is None:
        reg = SkillRegistry()
        for cls in (
            AcquireTargetSkill,
            ValidateTargetSkill,
            ApproachTargetSkill,
            EngageTargetSkill,
            BasicCombatRotationSkill,
            LootTargetSkill,
            DisengageSkill,
            RecoverHealthSkill,
            ExploreSkill,
            UnstuckSkill,
            ClearModalSkill,
            DeathRecoverySkill,
            GhostRunbackSkill,
            InteractSkill,
            WaitSkill,
        ):
            reg.register(cls)
        _DEFAULT = reg
    return _DEFAULT


def get_skill(name: str) -> Skill:
    return default_registry().get(name)


def list_skills() -> list[str]:
    return default_registry().names()


def register_extra(skills: Iterable[type[Skill]]) -> None:
    reg = default_registry()
    for cls in skills:
        reg.register(cls)
