# Learning Architecture V2 — Quick start

See also: [LEARNING_ARCHITECTURE_V2.md](./LEARNING_ARCHITECTURE_V2.md)

**Docs:** [RECURRENT_POLICY](./RECURRENT_POLICY.md) · [SKILL_COMMITMENT](./SKILL_COMMITMENT.md) · [EPISODE_LIFECYCLE](./EPISODE_LIFECYCLE.md) · [FEATURE_SCHEMA](./FEATURE_SCHEMA.md) · [DEMONSTRATION_RECORDING](./DEMONSTRATION_RECORDING.md) · [TRAINING](./TRAINING.md) · [EVALUATION](./EVALUATION.md)

The next-phase runtime is implemented and tested on this branch. A useful learned policy still requires real demonstrations and measured evaluation; visual learning and live improvement are not established.

## Enable skill-based policy (no training required)

In `config/owned_game.json`:

```json
"learning_v2": {
  "enabled": true,
  "policy_mode": "hybrid",
  "legacy_q_fallback": false,
  "use_rewards_v2": true,
  "track_episodes": true,
  "history_length": 16,
  "confidence_threshold": 0.45,
  "commitment_confidence_margin": 0.15,
  "minimum_commitment_seconds": 0.4,
  "maximum_commitment_seconds": 25.0,
  "controllable_frames": 3,
  "device": "cpu",
  "seed": 0
}
```

Modes:
- `scripted` — deterministic skills only
- `hybrid` — emergencies scripted; loads `bc_checkpoint` when set; else scripted fallback
- `legacy_q` — experimental raw tabular Q bridge
- `behavior_clone` — BC primary (same fallbacks as hybrid if checkpoint missing/low-confidence)

Set `"bc_checkpoint": "models/checkpoints/recurrent_skill_policy_v2.json"` after training to use the recurrent policy. Legacy MLP checkpoints remain loadable explicitly.

Also see [LOCAL_NEXT_STEPS](./LOCAL_NEXT_STEPS.md) for the game-PC calibration checklist.

Also see [LOCAL_NEXT_STEPS](./LOCAL_NEXT_STEPS.md) for the game-PC calibration checklist.

Validate settings:

```bash
PYTHONPATH=. python3 - <<'PY'
import json
from pathlib import Path
from playmind.config_v2 import LearningV2Settings
raw = json.loads(Path("config/owned_game.example.json").read_text())
s = LearningV2Settings.load_from_owned_config(raw)
s.validate()
print("ok", s.policy_mode, s.history_length)
PY
```

---

## Eight core flows

### 1. Running scripted mode

```bash
# Set learning_v2.policy_mode to "scripted" in config/owned_game.json
PYTHONPATH=. python3 scripts/run_owned_loop.py --config config/owned_game.json --max-ticks 30
# or GUI
PYTHONPATH=. python3 -m playmind.owned_gui
```

### 2. Recording demonstrations

Owned GUI (Phases 7 + 15): open Advanced V2 → Start/Stop recording, name/goal/profile/notes,
Mark Success/Failure/Bad. Optional **F9** toggle when `pynput` is installed. Policy mode,
model path, episode reset, clear legacy Q, and export diagnostics live on the same page.
Demo samples append from status ticks via GUI-owned `DemonstrationRecorder`; live-loop
`learning_v2` applies from `/api/v2/config` (or config file) on the next Start.

```bash
PYTHONPATH=. python3 -m playmind.owned_gui
# or API/scripted recorder:
PYTHONPATH=. python3 - <<'PY'
from playmind.demonstrations import DemonstrationRecorder
rec = DemonstrationRecorder(root="data/playmind/demonstrations")
rec.start(goal="farm", profile="human", episode_id="ep-1")
rec.append(
    observation={"vision_player_hp": 0.9, "has_target": False, "life_phase": "alive"},
    key_events=["hold:w:0.8"],
    skill="explore",
)
rec.mark("success")
print(rec.stop())
PY
```

Details: [DEMONSTRATION_RECORDING.md](./DEMONSTRATION_RECORDING.md)

### 3. Reviewing recorded data

```bash
PYTHONPATH=. python3 - <<'PY'
from playmind.demonstrations import list_sessions, load_session_samples
for s in list_sessions():
    print(s.name, len(load_session_samples(s)))
PY

PYTHONPATH=. python3 scripts/train_behavior_clone.py --dry-validate-only \
  --data-dir data/playmind/demonstrations \
  --history-length 16
```

### 4. Training behavior cloning

```bash
PYTHONPATH=. python3 scripts/train_behavior_clone.py \
  --data-dir data/playmind/demonstrations \
  --history-length 16 --batch-size 32 --epochs 30 \
  --checkpoint models/checkpoints/recurrent_skill_policy_v2.json
```

Details: [TRAINING.md](./TRAINING.md)

### 5. Evaluating a checkpoint

```bash
# Sequence-aware held-out evaluation
PYTHONPATH=. python3 scripts/evaluate_behavior_clone.py \
  --data-dir data/playmind/demonstrations \
  --checkpoint models/checkpoints/recurrent_skill_policy_v2.json \
  --history-length 16 --split test \
  --json-out data/playmind/evaluation/recurrent-test.json

# Baselines + evidence-separated comparative report
PYTHONPATH=. python3 scripts/run_evaluation.py \
  --data-dir data/playmind/demonstrations \
  --checkpoints models/checkpoints/recurrent_skill_policy_v2.json \
  --output-dir data/playmind/evaluation/recurrent-comparison
```

Details: [EVALUATION.md](./EVALUATION.md)

### 6. Running hybrid mode

```bash
# learning_v2.policy_mode = "hybrid" and set bc_checkpoint above
PYTHONPATH=. python3 scripts/run_owned_loop.py --config config/owned_game.json --max-ticks 30
```

Without a trained checkpoint, hybrid safely falls back to scripted skills.
Live actuation remains separately gated by ownership/configuration and `--live`.

### 7. Reverting to legacy mode

```bash
# Migrate / mark existing Q-table first
PYTHONPATH=. python3 scripts/migrate_legacy_learning.py --data-dir data/playmind/owned

# Then set learning_v2.policy_mode to "legacy_q" (experimental)
PYTHONPATH=. python3 scripts/run_owned_loop.py --config config/owned_game.json --max-ticks 30
```

Or disable V2 entirely: `"learning_v2": { "enabled": false }` to restore the pre-V2 priority stack with tabular Q.

Details: [MIGRATION.md](./MIGRATION.md)

### 8. Exporting diagnostics

```bash
PYTHONPATH=. python3 scripts/export_diagnostics.py \
  --owned-dir data/playmind/owned \
  --config config/owned_game.json
# → data/playmind/diagnostics/<timestamp>/ (+ .zip)
```

---

## Legacy Q notes

Existing `policy.json` is retained. With V2 enabled and `policy_mode` ≠ `legacy_q`, raw Q no longer chooses actions every tick. Set `"legacy_q_fallback": true` only for experiments.

Status logs include the active skill, commitment diagnostics, lifecycle/episode fields, `learning_v2`, and `reward_v2`.
