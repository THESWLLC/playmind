# Demonstration recording

Record human (or teleop) play for Learning Architecture V2 behavior cloning.

Sessions land under `data/playmind/demonstrations/<session_id>/` with:

- `meta.jsonl` — one sample per line (`schema_version=1`)
- `session.json` — outcome / notes
- optional `frames/` screenshots

Recording schema 1 is the on-disk demonstration format. Training converts structured observation dictionaries to [feature schema v2](./FEATURE_SCHEMA.md); saved frames are not consumed by the current model.

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

## Episode and quality rules

- Give each contiguous gameplay episode a stable `episode_id`; change it at true lifecycle boundaries.
- The dataset groups rows by `(session_id, episode_id)`, so sequence windows never cross an episode or concatenate same-named episodes from different sessions.
- Train/validation/test assignment is deterministic and episode-wise by default. A given episode ID stays in one split.
- Mark unusable sessions or samples `bad`. Bad and unlabeled rows are excluded by default; failed-but-valid demonstrations remain useful labeled data.
- Do not label death dialog, ghost/runback, loading, or not-yet-controllable recovery as normal gameplay. See [EPISODE_LIFECYCLE.md](./EPISODE_LIFECYCLE.md).

## Sequence windows

The recurrent dataset creates one causal window per eligible target row (subject to `--stride` and `--min-sequence-length`). Each window contains at most the latest 16 rows by default, remains inside one episode, and is left-padded with zero rows plus a validity mask.

Keep timestamps and labels ordered. Longer continuous recordings are acceptable when episode IDs are correct; a recorder session does not itself guarantee one episode.

## Review recorded data

```bash
# List sessions
PYTHONPATH=. python3 - <<'PY'
from playmind.demonstrations import list_sessions, load_session_samples
for s in list_sessions():
    rows = load_session_samples(s)
    print(s.name, "samples=", len(rows))
PY

# Validate recurrent windows and episode-separated split
PYTHONPATH=. python3 scripts/train_behavior_clone.py --dry-validate-only \
  --data-dir data/playmind/demonstrations \
  --history-length 16
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

Recording frames does not imply visual learning: `VisualObservationEncoder` is currently a placeholder.

See [TRAINING.md](./TRAINING.md), [RECURRENT_POLICY.md](./RECURRENT_POLICY.md), and [QUICKSTART_V2.md](./QUICKSTART_V2.md).
