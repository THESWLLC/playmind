"""Strict action + skill masking for Learning Architecture V2.

Rejects invented Q-table strings, death clicks while alive, casts without a
target, loot on living targets, and free movement during confirm modals.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from playmind.learning import OWNED_ACTIONS

# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

# Canonical low-level actions beyond OWNED_ACTIONS that skills / heuristics use.
_EXTRA_STATIC: tuple[str, ...] = (
    "key:1",
    "key:2",
    "key:3",
    "key:4",
    "key:5",
    "key:tab",
    "key:esc",
    "key:escape",
    "key:enter",
    "key:y",
    "key:c",
    "key:space",
    "hold:w:0.5",
    "hold:w:0.6",
    "hold:w:0.8",
    "hold:w:1.0",
    "hold:w:1.1",
    "hold:w:1.2",
    "hold:w:1.4",
    "hold:a:0.4",
    "hold:a:0.6",
    "hold:a:1.1",
    "hold:a:1.2",
    "hold:d:0.4",
    "hold:d:0.5",
    "hold:d:0.6",
    "hold:d:1.1",
    "hold:d:1.2",
    "hold:s:0.6",
    "hold:s:0.8",
    "hold:s:1.0",
    "hold:s:1.1",
    "hold:s:1.2",
    "click_label:Close",
    "click_label:Accept",
    "click_label:Continue",
    "click_label:Yes",
    "click_label:yes",
    "click_label:ok",
    "click_label:OK",
    "click_label:Release Spirit",
    "click_label:Return to Graveyard",
    "click_label:Resurrect in a Safe Zone",
    "click_label:Closest Town",
    "click_label:Closest City",
)

REGISTERED_ACTIONS: frozenset[str] = frozenset(list(OWNED_ACTIONS) + list(_EXTRA_STATIC))

# Patterns for dynamic but still-allowed strings (must match exactly these shapes).
_KEY_RE = re.compile(r"^key:(?:[1-5]|tab|esc|escape|enter|y|c|space)$", re.I)
_HOLD_RE = re.compile(r"^hold:[wasd]:([0-9]*\.?[0-9]+)$", re.I)
_CLICK_LABEL_RE = re.compile(r"^click_label:(.+)$", re.I)

# Death / rez UI labels — forbidden while alive.
_DEATH_LABEL_MARKERS = (
    "release spirit",
    "return to graveyard",
    "graveyard",
    "closest town",
    "closest city",
    "safe zone",
    "resurrect",
    "sanctuary",
)

_CONFIRM_LABEL_MARKERS = ("yes", "ok", "accept")

# Combat presses that require a target.
_TARGET_REQUIRED = frozenset(
    {
        "attack",
        "key:1",
        "key:2",
        "key:3",
        "key:4",
        "key:5",
    }
)

_MOVE_PREFIXES = ("move_", "hold:")

# Skills that are only legal in certain life states.
_DEATH_SKILLS = frozenset({"death_recovery"})
_GHOST_SKILLS = frozenset({"ghost_runback"})
_ALIVE_ONLY_SKILLS = frozenset(
    {
        "acquire_target",
        "validate_target",
        "approach_target",
        "engage_target",
        "basic_combat_rotation",
        "loot_target",
        "disengage",
        "recover_health",
        "explore",
        "unstuck",
        "interact",
    }
)


def _obs_map(obs: Mapping[str, Any] | dict[str, Any]) -> Mapping[str, Any]:
    return obs


def _phase(obs: Mapping[str, Any]) -> str:
    return str(obs.get("life_phase") or "alive")


def _alive(obs: Mapping[str, Any]) -> bool:
    phase = _phase(obs)
    if phase in {"dead_dialog", "confirm", "rez_picker", "ghost"}:
        return False
    return not bool(obs.get("is_dead")) and not bool(obs.get("is_ghost"))


def _is_dead_side(obs: Mapping[str, Any]) -> bool:
    phase = _phase(obs)
    return (
        bool(obs.get("is_dead"))
        or phase in {"dead_dialog", "confirm", "rez_picker"}
    )


def _is_ghost(obs: Mapping[str, Any]) -> bool:
    return bool(obs.get("is_ghost")) or _phase(obs) == "ghost"


def _confirm_modal(obs: Mapping[str, Any]) -> bool:
    return bool(obs.get("confirm_pending")) or _phase(obs) == "confirm"


def _is_deathish_action(action: str) -> bool:
    low = (action or "").strip().lower()
    if low == "release_spirit":
        return True
    if low.startswith("click_label:"):
        label = low.split(":", 1)[1]
        if any(m in label for m in _DEATH_LABEL_MARKERS):
            return True
        # Bare Yes/OK often means death confirm — treat as deathish when checking alive.
        if label in _CONFIRM_LABEL_MARKERS or label.startswith("yes"):
            return True
    if low.startswith("click:"):
        # Top-center death dialog band clicks.
        try:
            fx_s, fy_s = low.split(":", 1)[1].split(",")
            fx, fy = float(fx_s), float(fy_s)
            if 0.25 <= fx <= 0.70 and 0.05 <= fy <= 0.35:
                return True
        except Exception:
            return True
    return False


def _is_movement(action: str) -> bool:
    low = (action or "").strip().lower()
    return low.startswith(_MOVE_PREFIXES)


def _pattern_allowed(action: str) -> bool:
    """True if action matches a carefully constrained dynamic pattern."""
    a = (action or "").strip()
    if not a:
        return False
    if a in REGISTERED_ACTIONS:
        return True
    if _KEY_RE.match(a):
        return True
    m = _HOLD_RE.match(a)
    if m:
        try:
            sec = float(m.group(1))
        except ValueError:
            return False
        return 0.05 <= sec <= 3.0
    m = _CLICK_LABEL_RE.match(a)
    if m:
        label = m.group(1).strip()
        if not label or len(label) > 64:
            return False
        # Reject world-mob-looking freeform unless it is a known UI phrase.
        # Allowlist-ish: must look like UI (letters/spaces) and not invent junk.
        if re.search(r"[^\w\s\-'.]", label):
            return False
        return True
    return False


def is_registered_action(action: str) -> bool:
    return _pattern_allowed(action)


def validate_action(obs: Mapping[str, Any], action: str) -> tuple[bool, str]:
    """Return (ok, reason). Reject unknown / illegal actions for this obs."""
    a = (action or "").strip()
    if not a:
        return False, "empty_action"
    if not _pattern_allowed(a):
        return False, "unknown_or_invented_action"

    low = a.lower()
    alive = _alive(obs)
    dead_side = _is_dead_side(obs)
    ghost = _is_ghost(obs)

    # No death / release / rez clicks while alive.
    if alive and _is_deathish_action(a):
        return False, "death_action_while_alive"
    if alive and low == "release_spirit":
        return False, "release_while_alive"

    # No inventing arbitrary Q keys — already covered by pattern allowlist.
    if low.startswith("ability:") or low.startswith("bind:") or low.startswith("invent:"):
        return False, "invented_ability_action"

    # Target-required casts.
    if low in _TARGET_REQUIRED or (low.startswith("key:") and low in _TARGET_REQUIRED):
        if alive and not obs.get("has_target") and not obs.get("in_combat"):
            return False, "cast_without_target"

    # Loot only when target is dead / absent with loot flag.
    if low == "loot":
        thp = obs.get("target_hp_est")
        try:
            thp_f = float(thp) if thp is not None else None
        except (TypeError, ValueError):
            thp_f = None
        if obs.get("has_target") and thp_f is not None and thp_f > 0.05:
            return False, "loot_while_target_alive"

    # Confirm modal: block free movement unless explicitly needed (we disallow by default).
    if _confirm_modal(obs) and _is_movement(a):
        return False, "movement_during_confirm_modal"

    # While dead dialog / confirm / rez, only allow death UI + wait (+ enter).
    if dead_side and not ghost:
        if _is_movement(a) and low not in {"wait"}:
            return False, "movement_while_dead"
        if low in _TARGET_REQUIRED:
            return False, "combat_while_dead"
        if low in {"loot", "attack", "target_nearest", "key:tab"}:
            return False, "farm_while_dead"

    # Ghost: no living combat / loot / release again.
    if ghost:
        if low in _TARGET_REQUIRED or low == "loot":
            return False, "combat_while_ghost"
        if low == "release_spirit" or (
            low.startswith("click_label:") and "release" in low
        ):
            return False, "release_while_ghost"

    # Esc opens Options mid-fight — reject unless modal already open.
    if alive and low in {"key:esc", "key:escape"} and not obs.get("modal_menu"):
        return False, "esc_without_modal"

    return True, "ok"


def mask_actions(
    obs: Mapping[str, Any],
    candidates: Sequence[str] | Iterable[str],
) -> list[str]:
    """Filter candidates to those legal under validate_action."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        a = str(raw).strip()
        if not a or a in seen:
            continue
        ok, _ = validate_action(obs, a)
        if ok:
            out.append(a)
            seen.add(a)
    return out


def mask_skills(
    obs: Mapping[str, Any],
    skill_names: Sequence[str] | Iterable[str],
) -> list[str]:
    """Filter high-level skill names by life-state + simple preconditions."""
    alive = _alive(obs)
    ghost = _is_ghost(obs)
    dead_side = _is_dead_side(obs)
    modal = bool(obs.get("modal_menu"))
    has_target = bool(obs.get("has_target"))
    out: list[str] = []
    for name in skill_names:
        n = str(name).strip()
        if not n:
            continue
        if n in _DEATH_SKILLS:
            if dead_side and not ghost:
                out.append(n)
            continue
        if n in _GHOST_SKILLS:
            if ghost:
                out.append(n)
            continue
        if n == "clear_modal":
            if modal and alive:
                out.append(n)
            continue
        if n == "wait":
            out.append(n)
            continue
        if n in _ALIVE_ONLY_SKILLS and not alive:
            continue
        if not alive:
            continue
        # Alive-side soft preconditions.
        if n == "acquire_target" and has_target:
            continue
        if n in {"validate_target", "approach_target", "engage_target", "basic_combat_rotation"}:
            if not has_target:
                continue
        if n == "loot_target":
            thp = obs.get("target_hp_est")
            try:
                thp_f = float(thp) if thp is not None else None
            except (TypeError, ValueError):
                thp_f = None
            if has_target and thp_f is not None and thp_f > 0.05:
                continue
            if not (
                (thp_f is not None and thp_f <= 0.05)
                or obs.get("loot_available")
                or obs.get("recent_kill")
                or (not has_target and thp_f is not None and thp_f <= 0.05)
            ):
                # Allow when no living target and loot flags unset — still mask if
                # clearly fighting a living target (handled above). Otherwise skip
                # unless kill/loot signals present.
                if has_target:
                    continue
                if not (obs.get("loot_available") or obs.get("recent_kill")):
                    continue
        if n == "recover_health":
            try:
                hp = float(obs.get("vision_player_hp") or obs.get("player", {}).get("hp") or 1.0)
            except (TypeError, ValueError, AttributeError):
                hp = 1.0
            if hp >= 0.45:
                continue
        if n == "unstuck":
            if not (obs.get("stuck_hint") or int(obs.get("stagnant") or 0) >= 4):
                continue
        if modal and n not in {"clear_modal", "wait"}:
            continue
        out.append(n)
    return out
