#!/usr/bin/env python3
"""Generate synthetic demonstration sessions for pipeline smoke tests.

These are NOT human gameplay. They exist so train/eval/hybrid wiring can be
exercised without a live game. Real competence still requires local demos.

Usage:
  python3 scripts/generate_synthetic_demos.py \\
    --out-dir data/playmind/demonstrations \\
    --sessions 6 --episodes-per-session 2 --steps 24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playmind.demonstrations import DemonstrationRecorder

# Simple scripted trajectories: skill sequences that look episode-like.
TRAJECTORIES: list[tuple[str, list[str]]] = [
    (
        "explore_and_pull",
        [
            "explore",
            "explore",
            "acquire_target",
            "validate_target",
            "approach_target",
            "engage_target",
            "basic_combat_rotation",
            "basic_combat_rotation",
            "loot_target",
            "explore",
        ],
    ),
    (
        "death_recovery",
        [
            "basic_combat_rotation",
            "death_recovery",
            "death_recovery",
            "ghost_runback",
            "ghost_runback",
            "wait",
            "explore",
        ],
    ),
    (
        "stuck_clear",
        [
            "explore",
            "unstuck",
            "unstuck",
            "clear_modal",
            "explore",
            "acquire_target",
            "wait",
        ],
    ),
    (
        "heal_and_reengage",
        [
            "basic_combat_rotation",
            "recover_health",
            "recover_health",
            "disengage",
            "acquire_target",
            "approach_target",
            "engage_target",
            "basic_combat_rotation",
        ],
    ),
]


def _obs_for_skill(skill: str, step: int, episode: int) -> dict:
    """Build a coherent-enough observation for the labeled skill."""
    base = {
        "vision_player_hp": 0.85,
        "player_hp_confidence": 0.9,
        "target_hp": None,
        "target_hp_confidence": None,
        "has_target": False,
        "has_target_confidence": 0.9,
        "in_combat": False,
        "in_combat_confidence": 0.8,
        "is_dead": False,
        "is_ghost": False,
        "life_phase": "alive",
        "motion": 2.0 + (step % 5),
        "motion_confidence": 0.7,
        "hostiles_near": True,
        "hostile_count": 1,
        "hostile_count_confidence": 0.6,
        "blocking_modal": False,
        "modal_menu": False,
        "objective_progress": min(1.0, 0.05 * step),
        "stagnation_count": 0,
        "failed_action_streak": 0,
        "progress_stage": "farm",
        "stuck_hint": "none",
        "sensor_warnings": [],
    }
    if skill in {"acquire_target", "validate_target", "approach_target"}:
        base.update(
            {
                "has_target": skill != "acquire_target",
                "target_hp": 1.0 if skill != "acquire_target" else None,
                "target_hp_confidence": 0.8 if skill != "acquire_target" else None,
                "hostiles_near": True,
            }
        )
    if skill in {"engage_target", "basic_combat_rotation", "loot_target"}:
        base.update(
            {
                "has_target": True,
                "has_target_confidence": 0.95,
                "in_combat": skill != "loot_target",
                "target_hp": 0.2 if skill == "loot_target" else max(0.1, 0.9 - 0.1 * step),
                "target_hp_confidence": 0.9,
                "motion": 1.0,
            }
        )
    if skill == "loot_target":
        base["in_combat"] = False
        base["target_hp"] = 0.0
    if skill in {"death_recovery"}:
        base.update(
            {
                "is_dead": True,
                "vision_player_hp": 0.0,
                "life_phase": "dead_dialog",
                "has_target": False,
                "in_combat": False,
                "hostiles_near": False,
            }
        )
    if skill == "ghost_runback":
        base.update(
            {
                "is_dead": False,
                "is_ghost": True,
                "life_phase": "ghost",
                "vision_player_hp": 0.0,
                "motion": 4.0,
            }
        )
    if skill == "unstuck":
        base.update({"stuck_hint": "severe", "stagnation_count": 8, "motion": 0.0})
    if skill == "clear_modal":
        base.update({"blocking_modal": True, "modal_menu": True})
    if skill == "recover_health":
        base.update({"vision_player_hp": 0.25, "in_combat": False, "has_target": False})
    if skill == "disengage":
        base.update({"in_combat": True, "has_target": True, "vision_player_hp": 0.3})
    if skill == "explore":
        base.update({"has_target": False, "hostiles_near": episode % 2 == 0, "motion": 5.0})
    if skill == "wait":
        base.update({"motion": 0.0, "life_phase": "alive"})
    return base


def generate(
    out_dir: Path,
    *,
    sessions: int,
    episodes_per_session: int,
    steps: int,
    seed: int = 0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"sessions": [], "total_samples": 0, "out_dir": str(out_dir)}
    t0 = time.time()
    for s_i in range(sessions):
        traj_name, skill_cycle = TRAJECTORIES[s_i % len(TRAJECTORIES)]
        rec = DemonstrationRecorder(root=out_dir, input_source="playmind_generated")
        rec.start(
            goal=f"synthetic:{traj_name}",
            profile="synthetic_smoke",
        )
        assert rec.session_dir is not None
        samples = 0
        for e_i in range(episodes_per_session):
            ep = f"ep-{s_i}-{e_i}-{uuid.uuid4().hex[:8]}"
            n = max(8, steps)
            for step in range(n):
                skill = skill_cycle[step % len(skill_cycle)]
                obs = _obs_for_skill(skill, step, e_i)
                rec.append(
                    observation=obs,
                    skill=skill,
                    label="success",
                    key_events=[],
                    goal=f"synthetic:{traj_name}",
                    profile="synthetic_smoke",
                    notes=f"synthetic seed={seed} session={s_i}",
                    timestamp=t0 + s_i * 1000 + e_i * 100 + step,
                    episode_id=ep,
                )
                samples += 1
        outcome = "success" if s_i % 5 else "failure"
        rec.mark(outcome, notes="synthetic session end")
        rec.stop()
        status_path = rec.session_dir / "session.json"
        if status_path.exists():
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
                payload["synthetic"] = True
                payload["trajectory"] = traj_name
                status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            except (OSError, json.JSONDecodeError):
                pass
        summary["sessions"].append(
            {
                "session_dir": str(rec.session_dir),
                "trajectory": traj_name,
                "samples": samples,
                "outcome": outcome,
            }
        )
        summary["total_samples"] += samples
    manifest = out_dir / "SYNTHETIC_MANIFEST.json"
    manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["manifest"] = str(manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate synthetic PlayMind demos")
    p.add_argument("--out-dir", default="data/playmind/demonstrations")
    p.add_argument("--sessions", type=int, default=6)
    p.add_argument("--episodes-per-session", type=int, default=2)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    summary = generate(
        Path(args.out_dir),
        sessions=max(1, args.sessions),
        episodes_per_session=max(1, args.episodes_per_session),
        steps=max(8, args.steps),
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"Wrote {summary['total_samples']} samples across "
            f"{len(summary['sessions'])} sessions → {summary['out_dir']}"
        )
        print("NOTE: synthetic only — replace with real demos before expecting skillful play.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
