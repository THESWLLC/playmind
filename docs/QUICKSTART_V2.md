# Learning Architecture V2 — Quick start

See also: [LEARNING_ARCHITECTURE_V2.md](./LEARNING_ARCHITECTURE_V2.md)

## Enable skill-based policy (no training required)

In `config/owned_game.json`:

```json
"learning_v2": {
  "enabled": true,
  "policy_mode": "hybrid",
  "legacy_q_fallback": false,
  "use_rewards_v2": true,
  "track_episodes": true
}
```

Modes:
- `scripted` — deterministic skills only
- `hybrid` — emergencies + scripted (BC stub falls back until a checkpoint exists)
- `legacy_q` — experimental raw tabular Q bridge

## Run

```bash
PYTHONPATH=. python3 scripts/run_owned_loop.py --config config/owned_game.json --max-ticks 30
# or GUI
PYTHONPATH=. python3 -m playmind.owned_gui
```

Status logs include `active_skill`, `learning_v2`, `reward_v2`, and `episode_id`.

## Legacy Q

Existing `policy.json` is retained. With V2 enabled and `policy_mode` ≠ `legacy_q`, raw Q no longer chooses actions every tick. Set `"legacy_q_fallback": true` only for experiments.

## Still coming (follow-on)

- Demonstration recorder GUI
- Behavior-cloning training
- Replay env / evaluation reports
- Sensor labeling tool
