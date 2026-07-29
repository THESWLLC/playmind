# Learning Architecture V2 — Quick start

See also: [LEARNING_ARCHITECTURE_V2.md](./LEARNING_ARCHITECTURE_V2.md)

**Docs:** [DEMONSTRATION_RECORDING](./DEMONSTRATION_RECORDING.md) · [TRAINING](./TRAINING.md) · [EVALUATION](./EVALUATION.md) · [SENSOR_LABELING](./SENSOR_LABELING.md) · [SKILLS](./SKILLS.md) · [MIGRATION](./MIGRATION.md)

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
  "device": "cpu",
  "seed": 0
}
```

Modes:
- `scripted` — deterministic skills only
- `hybrid` — emergencies + scripted (BC stub falls back until a checkpoint exists)
- `legacy_q` — experimental raw tabular Q bridge
- `behavior_clone` — BC primary (falls back like hybrid until a real checkpoint loads)

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
  --data-dir data/playmind/demonstrations
```

### 4. Training behavior cloning

```bash
PYTHONPATH=. python3 scripts/train_behavior_clone.py \
  --data-dir data/playmind/demonstrations \
  --window-size 4 --batch-size 8 --epochs 1 \
  --checkpoint models/checkpoints/skill_policy_v2.json
```

Details: [TRAINING.md](./TRAINING.md)

### 5. Evaluating a checkpoint

```bash
PYTHONPATH=. python3 - <<'PY'
from playmind.models.policy_v2 import SkillPolicyV2
from playmind.demonstrations import list_sessions
from playmind.replay_env import ReplayEnv

policy = SkillPolicyV2.load("models/checkpoints/skill_policy_v2.json")
sessions = list_sessions()
print("trained=", policy.trained)
if sessions:
    env = ReplayEnv.from_session(sessions[0], policy=policy)
    env.reset()
    n = 0
    while not env.done:
        if env.step() is None:
            break
        n += 1
    print("replayed=", n)
PY
```

Details: [EVALUATION.md](./EVALUATION.md)

### 6. Running hybrid mode

```bash
# learning_v2.policy_mode = "hybrid" (default), optional bc_checkpoint
PYTHONPATH=. python3 scripts/run_owned_loop.py --config config/owned_game.json --max-ticks 30
```

Without a trained checkpoint, hybrid safely falls back to scripted skills.

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

Status logs include `active_skill`, `learning_v2`, `reward_v2`, and `episode_id`.
