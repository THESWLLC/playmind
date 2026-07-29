"""Interactive teach-mode helpers."""

from __future__ import annotations

from typing import Any

from playmind.demo_world import ACTIONS


HELP = """Teach commands:
  <action>     use one of: {actions}
  dir <text>   set a high-level directive (farm / stop / turn in)
  accept       accept the agent's suggestion
  skip         skip teaching this step
  help         show this help
  quit         stop the episode
""".format(actions=", ".join(ACTIONS))


def prompt_teacher(suggestion: str, obs: dict[str, Any]) -> tuple[str, str | None]:
    """Return (command, optional_action_override).

    command in: accept | action | directive | skip | quit | help
    """
    print("--- teach mode ---")
    print(
        f"hp={obs['player']['hp']} pos=({obs['player']['x']},{obs['player']['y']}) "
        f"kills={obs.get('quest_kills')}/{obs.get('quest_kills_needed')} "
        f"npc_here={obs.get('npc_here')}"
    )
    if obs.get("vision_quest_text"):
        print("vision quest:", obs["vision_quest_text"])
    elif obs.get("quest_text"):
        print("quest:", obs["quest_text"])
    print(f"agent suggests: {suggestion}")
    raw = input("> ").strip()
    if not raw:
        return "accept", suggestion
    low = raw.lower()
    if low in {"accept", "y", "yes", ""}:
        return "accept", suggestion
    if low in {"skip", "s"}:
        return "skip", None
    if low in {"quit", "q", "exit"}:
        return "quit", None
    if low in {"help", "h", "?"}:
        print(HELP)
        return "help", None
    if low.startswith("dir "):
        return "directive", raw[4:].strip()
    normalized = low.replace("-", "_")
    if normalized in ACTIONS:
        return "action", normalized
    print(f"Unrecognized: {raw}")
    print(HELP)
    return "help", None
