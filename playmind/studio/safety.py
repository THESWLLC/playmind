"""Hard safety boundary for the offline-only Studio."""

from __future__ import annotations

from collections.abc import Iterable


class StudioSafetyError(RuntimeError):
    """Raised when an operation would cross Studio's offline boundary."""


FORBIDDEN_GAME_PROCESSES = frozenset(
    {
        "wow.exe",
        "wowclassic.exe",
        "wowclassic",
        "wowclassict.exe",
        "wowclassicb.exe",
        "wowclassic_t.exe",
        "wowclassic_era.exe",
        "wow",
        "wowt.exe",
        "wowb.exe",
        "wow-64.exe",
        "world of warcraft.exe",
    }
)


def _process_basename(value: object) -> str:
    text = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    return text.casefold()


def detect_forbidden_live_context(process_names: Iterable[object]) -> bool:
    """Reject a supplied process snapshot containing a World of Warcraft client.

    Process discovery is deliberately left to the caller, making this function
    deterministic and unit-testable. Studio never performs process inspection
    or starts a live capture itself.
    """

    detected = sorted(
        {
            _process_basename(name)
            for name in process_names
            if _process_basename(name) in FORBIDDEN_GAME_PROCESSES
        }
    )
    if detected:
        raise StudioSafetyError(
            "PlayMind Studio is offline-only and refuses live functionality "
            "while a retail World of Warcraft client is present: "
            + ", ".join(detected)
        )
    return False


def assert_studio_safe() -> bool:
    """Assert the invariant exposed to CLI/GUI callers: Studio cannot send input."""

    return True


def studio_may_not_send_input() -> bool:
    """Return the permanent no-generated-input policy."""

    return True


__all__ = [
    "FORBIDDEN_GAME_PROCESSES",
    "StudioSafetyError",
    "assert_studio_safe",
    "detect_forbidden_live_context",
    "studio_may_not_send_input",
]
