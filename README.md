# PlayMind

Local AI agent framework for **games you own** (or are allowed to automate).

PlayMind can:
- Play a built-in demo world
- Learn from its own runs (tabular policy + experience logs)
- Ask you questions in teach mode
- Capture screens + OCR/vision ROIs
- Drive an owned-game keyboard actuator (opt-in)
- Schedule long sessions with breaks
- Export / build local LLM planner artifacts (Ollama Modelfile, optional LoRA)

> **Not for World of Warcraft or other ToS-restricted live clients.**  
> Do not point this at official MMO clients with automated keyboard/mouse control.

## Quick start (no extra installs)

```bash
python3 playmind_onefile.py --episodes 5
python3 -m playmind --episodes 5
```

### GUI with live logging

```bash
PYTHONPATH=. python3 -m playmind.web_gui
# open http://127.0.0.1:8765
```

Shows the world map, current action/HP/kills, and a scrolling event log.

## Full feature commands

```bash
# Teach mode
python3 -m playmind --teach --episodes 1

# Demo vision frames
python3 -m playmind --vision --episodes 3

# Self-play + fine-tune export
PYTHONPATH=. python3 scripts/self_play_train.py --episodes 50
PYTHONPATH=. python3 scripts/finetune_export_check.py
PYTHONPATH=. python3 scripts/build_ollama_modelfile.py

# Screen capture / OCR (needs: pip install mss Pillow pytesseract)
PYTHONPATH=. python3 scripts/capture_once.py --ocr

# Owned-game loop (dry-run by default)
cp config/owned_game.example.json config/owned_game.json
# edit: i_own_this_game, capture, rois, keymap
PYTHONPATH=. python3 scripts/run_owned_loop.py --config config/owned_game.json --max-ticks 10
```

Optional local LLM planner:

```bash
ollama pull dolphin-llama3
ollama create playmind-planner -f models/Modelfile.playmind
python3 -m playmind --ollama --ollama-model playmind-planner --episodes 1
```

Docs: [docs/FULL_STACK.md](docs/FULL_STACK.md) · [docs/NEXT_STACK.md](docs/NEXT_STACK.md) · [docs/QUICKSTART_V2.md](docs/QUICKSTART_V2.md) · [docs/LEARNING_ARCHITECTURE_V2.md](docs/LEARNING_ARCHITECTURE_V2.md) · [docs/DEMONSTRATION_RECORDING.md](docs/DEMONSTRATION_RECORDING.md) · [docs/TRAINING.md](docs/TRAINING.md) · [docs/EVALUATION.md](docs/EVALUATION.md) · [docs/SENSOR_LABELING.md](docs/SENSOR_LABELING.md) · [docs/SKILLS.md](docs/SKILLS.md) · [docs/MIGRATION.md](docs/MIGRATION.md)

## Learning V2 (skills / hybrid)

Default path can use hierarchical skills instead of raw tabular Q:

```json
"learning_v2": { "enabled": true, "policy_mode": "hybrid" }
```

See [docs/QUICKSTART_V2.md](docs/QUICKSTART_V2.md) for the eight flows (scripted, demos, review, train, eval, hybrid, legacy, diagnostics). Legacy Q remains as an optional fallback.

```bash
# 1) Scripted
# config: learning_v2.policy_mode=scripted
PYTHONPATH=. python3 scripts/run_owned_loop.py --config config/owned_game.json --max-ticks 30

# 2–3) Record + review demos — docs/DEMONSTRATION_RECORDING.md
PYTHONPATH=. python3 scripts/train_behavior_clone.py --dry-validate-only

# 4) Train BC
PYTHONPATH=. python3 scripts/train_behavior_clone.py --checkpoint models/checkpoints/skill_policy_v2.json

# 5) Eval / replay — docs/EVALUATION.md

# 6) Hybrid (default when enabled)
PYTHONPATH=. python3 scripts/run_owned_loop.py --config config/owned_game.json --max-ticks 30

# 7) Legacy migrate + policy_mode=legacy_q
PYTHONPATH=. python3 scripts/migrate_legacy_learning.py --data-dir data/playmind/owned

# 8) Diagnostics bundle
PYTHONPATH=. python3 scripts/export_diagnostics.py
```

## Safety defaults

- Keyboard sending is **off** unless you set `i_own_this_game=true` and `enable_keyboard=true` and pass `--live`
- Demo paths never touch remote MMO clients

## Project layout

```text
playmind/     agent, vision, capture, actuators, owned loop, session
scripts/      capture, owned loop, self-play, fine-tune helpers
config/       keymap + owned_game examples
docs/         architecture + compliance notes
tests/
```

## License

MIT — see [LICENSE](LICENSE)
