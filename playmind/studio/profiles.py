"""Offline Studio game profiles and their enforceable capabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass


PROFILE_RETAIL_WOW_OFFLINE_ONLY = "retail_wow_offline_only"


@dataclass(frozen=True)
class ProtectedProfile:
    name: str
    live_perception: bool = False
    live_capture: bool = False
    physical_input_logging: bool = False
    live_planning: bool = False
    process_access: bool = False
    generated_input: bool = False
    prohibited_planning_modes: tuple[str, ...] = (
        "shadow",
        "assist",
        "hybrid",
        "autonomous",
    )
    allowed_uses: tuple[str, ...] = (
        "offline_video_import",
        "offline_frame_analysis",
        "human_annotation",
        "offline_dataset_export",
        "offline_evaluation",
    )

    def __post_init__(self) -> None:
        prohibited = (
            self.live_perception,
            self.live_capture,
            self.physical_input_logging,
            self.live_planning,
            self.process_access,
            self.generated_input,
        )
        if any(prohibited):
            raise ValueError("a ProtectedProfile cannot enable live or input capabilities")
        required_modes = {"shadow", "assist", "hybrid", "autonomous"}
        if not required_modes.issubset(self.prohibited_planning_modes):
            raise ValueError("a ProtectedProfile must prohibit every live planning mode")

    @property
    def offline_only(self) -> bool:
        return True

    @property
    def live_use_prohibited(self) -> bool:
        return True

    @property
    def disallow_live_perception(self) -> bool:
        return True

    @property
    def disallow_live_capture(self) -> bool:
        return True

    @property
    def disallow_physical_input_logging_for_gameplay(self) -> bool:
        return True

    @property
    def disallow_process_access(self) -> bool:
        return True

    @property
    def disallow_generated_input(self) -> bool:
        return True

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["allowed_uses"] = list(self.allowed_uses)
        result["prohibited_planning_modes"] = list(self.prohibited_planning_modes)
        result["offline_only"] = True
        result["live_use_prohibited"] = True
        return result


RETAIL_WOW_OFFLINE_ONLY = ProtectedProfile(PROFILE_RETAIL_WOW_OFFLINE_ONLY)
PROTECTED_PROFILES = {RETAIL_WOW_OFFLINE_ONLY.name: RETAIL_WOW_OFFLINE_ONLY}


def get_profile(name: str = PROFILE_RETAIL_WOW_OFFLINE_ONLY) -> ProtectedProfile:
    try:
        return PROTECTED_PROFILES[str(name)]
    except KeyError as exc:
        raise KeyError(f"unknown Studio profile: {name!r}") from exc


__all__ = [
    "PROFILE_RETAIL_WOW_OFFLINE_ONLY",
    "PROTECTED_PROFILES",
    "ProtectedProfile",
    "RETAIL_WOW_OFFLINE_ONLY",
    "get_profile",
]
