# PlayMind

Local AI agent framework for **games you own** (or are allowed to automate).

> **Not for World of Warcraft or other ToS-restricted live clients.**  
> Do not point this at official MMO clients with automated keyboard/mouse control.

## Status

**Learning Architecture V2 is implemented on `main`** (skills, hybrid/scripted/BC policies, demos, train/eval, sensor metrics, owned GUI V2, config validation, migration, diagnostics).

| Layer | State |
|-------|--------|
| Framework / V2 learning stack | Done — see [docs/QUICKSTART_V2.md](docs/QUICKSTART_V2.md) |
| Competent play on *your* game | Local work — calibrate window/ROIs/keymap, record demos, optionally train BC |

Without a BC checkpoint, `policy_mode: hybrid` runs **scripted skills** (safe default).

## Quick start (no extra installs)

```bash
python3 playmind_onefile.py --episodes 5
python3 -m playmind --episodes 5
```

### GUIs

```bash
# Demo world + live log
python3 -m playmind.web_gui
# open http://127.0.0.1:8765

# Owned-game brain monitor (Learning V2 controls)
python3 -m playmind.owned_gui
# open http://127.0.0.1:8777
```

## Owned-game loop

```bash
cp config/owned_game.example.json config/owned_game.json
# edit: i_own_this_game, capture.window_title, rois, keymap
python3 scripts/run_owned_loop.py --config config/owned_game.json --max-ticks 10
```

Keyboard is **off** unless `i_own_this_game=true`, `enable_keyboard=true`, and you pass `--live`.

## Learning V2

```json
"learning_v2": {
  "enabled": true,
  "policy_mode": "hybrid",
  "bc_checkpoint": null
}
```

```bash
# Scripted / hybrid dry-run
python3 scripts/run_owned_loop.py --config config/owned_game.json --max-ticks 30

# Validate demos / train BC (needs demos; torch optional for real MLP train)
python3 scripts/train_behavior_clone.py --dry-validate-only
python3 scripts/train_behavior_clone.py --checkpoint models/checkpoints/skill_policy_v2.json

# Eval + diagnostics
python3 scripts/run_evaluation.py
python3 scripts/export_diagnostics.py
python3 scripts/migrate_legacy_learning.py --data-dir data/playmind/owned
```

Full flows: [docs/QUICKSTART_V2.md](docs/QUICKSTART_V2.md)

### Local next steps (your PC)

1. Set `capture.window_title` and ROIs for your game  
2. Dry-run with `learning_v2.enabled=true` / `policy_mode=scripted`  
3. Record demos in owned GUI → train BC → set `bc_checkpoint`  
4. Only then enable `--live` keyboard  

## Other features

```bash
python3 -m playmind --teach --episodes 1
python3 -m playmind --vision --episodes 3
python3 scripts/self_play_train.py --episodes 50
python3 scripts/capture_once.py --ocr   # needs mss Pillow pytesseract
```

Optional Ollama planner:

```bash
ollama pull dolphin-llama3
ollama create playmind-planner -f models/Modelfile.playmind
python3 -m playmind --ollama --ollama-model playmind-planner --episodes 1
```

## Docs

- [QUICKSTART_V2](docs/QUICKSTART_V2.md) · [LEARNING_ARCHITECTURE_V2](docs/LEARNING_ARCHITECTURE_V2.md) · [SKILLS](docs/SKILLS.md)
- [DEMONSTRATION_RECORDING](docs/DEMONSTRATION_RECORDING.md) · [TRAINING](docs/TRAINING.md) · [EVALUATION](docs/EVALUATION.md)
- [SENSOR_LABELING](docs/SENSOR_LABELING.md) · [MIGRATION](docs/MIGRATION.md)
- [FULL_STACK](docs/FULL_STACK.md) · [COMPLIANCE_BOUNDARIES](docs/COMPLIANCE_BOUNDARIES.md)

## Layout

```text
playmind/     agent, owned loop, skills, policies, training, GUI
scripts/      owned loop, BC train/eval, diagnostics, teleop
config/       owned_game + keymap examples
docs/         architecture + V2 guides
tests/
```

## License

MIT — see [LICENSE](LICENSE)
