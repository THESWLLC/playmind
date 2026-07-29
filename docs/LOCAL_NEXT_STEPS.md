# Local next steps (your game PC)

Cloud agents **cannot** see your game window, press your keys, or record real demos.
Do these steps on the machine that runs the game (Cursor Desktop / terminal).

## What is already done in the repo

- Learning Architecture V2 + recurrent GRU BC
- Skill commitment, episode lifecycle, feature schema v2
- Outcome-based offline evaluation
- Helpers:
  - `scripts/setup_owned_game.py` — copy/edit `config/owned_game.json`
  - `scripts/generate_synthetic_demos.py` — fake demos for pipeline smoke only
  - `scripts/smoke_local_pipeline.py` — generate → train → eval smoke

## Checklist

### 1. Calibrate capture + keymap

```bash
git pull origin main
python3 scripts/setup_owned_game.py --window-title "YourGameTitle"
python3 scripts/capture_once.py --list-windows
python3 scripts/capture_once.py --window "YourGameTitle" --out data/playmind/capture_sample.png
```

Edit `config/owned_game.json`:

- `capture.window_title`
- `rois` (at least `hp_roi`; add target/combat ROIs as you calibrate)
- `keymap_path` / key bindings in `config/keymap.example.json`

Keep `i_own_this_game=false` and `enable_keyboard=false` until dry-runs look sane.

### 2. Dry-run the owned loop (no keys)

```bash
python3 scripts/run_owned_loop.py --config config/owned_game.json --max-ticks 30
# or
python3 -m playmind.owned_gui
```

Confirm: window capture works, HP/OCR/life phase look plausible, skills commit without thrashing.

### 3. Record real demonstrations

In owned GUI → Advanced V2:

1. Start recording  
2. Play normally (or teleop)  
3. Mark success / failure / bad  
4. Stop  

Store under `data/playmind/demonstrations/` (gitignored).

Optional smoke with **synthetic** demos (not gameplay quality):

```bash
python3 scripts/smoke_local_pipeline.py --device cpu
```

### 4. Train recurrent BC

```bash
python3 scripts/train_behavior_clone.py \
  --data-dir data/playmind/demonstrations \
  --checkpoint models/checkpoints/recurrent_skill_policy_v2.json \
  --history-length 16 \
  --model-type recurrent \
  --device auto
```

Point config:

```json
"learning_v2": {
  "enabled": true,
  "policy_mode": "hybrid",
  "bc_checkpoint": "models/checkpoints/recurrent_skill_policy_v2.json"
}
```

### 5. Evaluate offline

```bash
python3 scripts/run_evaluation.py \
  --data-dir data/playmind/demonstrations \
  --checkpoints models/checkpoints/recurrent_skill_policy_v2.json \
  --compare-scripted \
  --output-dir data/playmind/eval/latest
```

Read `report.md`. Treat counterfactual sections as estimates only.

### 6. Hybrid dry-run, then live (owned games only)

```bash
python3 scripts/run_owned_loop.py --config config/owned_game.json --max-ticks 50
```

Only when ownership gates are intentional:

```json
"i_own_this_game": true,
"enable_keyboard": true
```

```bash
python3 scripts/run_owned_loop.py --config config/owned_game.json --live --max-ticks 50
```

## Still future work

- Visual encoder training (interfaces exist; no frame model yet)
- Large-scale human-like competence (needs many real demos + iteration)
- Stateful GRU live inference (optional; default is safe last-16 stateless)

## Docs

- [QUICKSTART_V2](./QUICKSTART_V2.md)
- [RECURRENT_POLICY](./RECURRENT_POLICY.md)
- [DEMONSTRATION_RECORDING](./DEMONSTRATION_RECORDING.md)
- [TRAINING](./TRAINING.md)
- [EVALUATION](./EVALUATION.md)
- [COMPLIANCE_BOUNDARIES](./COMPLIANCE_BOUNDARIES.md)
