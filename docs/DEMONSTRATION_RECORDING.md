# Demonstration recording

Record human (or teleop) play for Learning Architecture V2 behavior cloning.

Sessions land under `data/playmind/demonstrations/<session_id>/` with:

- `meta.jsonl` — one sample per line (`schema_version=1`)
- `session.json` — outcome / notes
- optional `frames/` screenshots

## Record a session (library)

```bash
PYTHONPATH=. python3 - <<'PY'
from playmind.demonstrations import DemonstrationRecorder

rec = DemonstrationRecorder(root="data/playmind/demonstrations")
rec.start(goal="farm", profile="human", episode_id="ep-1")
rec.append(
    observation={"vision_player_hp": 0.9, "has_target": False, "life_phase": "alive"},
    key_events=["hold:w:0.8", "key:tab"],
    skill="explore",
)
rec.mark("success", notes="clean walk")
print(rec.stop())
PY
```

## Review recorded data

```bash
# List sessions
PYTHONPATH=. python3 - <<'PY'
from playmind.demonstrations import list_sessions, load_session_samples
for s in list_sessions():
    rows = load_session_samples(s)
    print(s.name, "samples=", len(rows))
PY

# Validate windows for BC
PYTHONPATH=. python3 scripts/train_behavior_clone.py --dry-validate-only \
  --data-dir data/playmind/demonstrations
```

## Config knobs

In `config/owned_game.json` under `learning_v2.demonstration`:

```json
"demonstration": {
  "enabled": false,
  "root": "data/playmind/demonstrations",
  "save_frames": true,
  "max_session_samples": 10000
}
```

See also: [TRAINING.md](./TRAINING.md) · [QUICKSTART_V2.md](./QUICKSTART_V2.md)
