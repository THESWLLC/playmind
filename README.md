# PlayMind

Local AI agent framework for **games you own** (or are allowed to automate).

> **Not for World of Warcraft or other ToS-restricted live clients.**  
> Do not point this at official MMO clients with automated keyboard/mouse control.

For recordings of protected games, use the separate, file-only
**PlayMind Offline Studio** workflow. Start with
[START_HERE.md](START_HERE.md).

## Status

**Planner V2, the owned-game learning-proof GUI, and the Offline Studio backend
are implemented on this branch:** structured LLM skill plans, strict validation
and runtime modes, physical demonstration capture/segmentation for authorized
owned-game labs, offline video/provenance/annotation/data building, QLoRA/DPO
entry points, evaluation indexing, and an audited model registry. Real useful
behavior still needs permissioned demonstrations and held-out evaluation; no
live improvement claim is made.

| Layer | State |
|-------|--------|
| Planner V2 / learning-proof GUI | Implemented and automated-test covered — see [docs/MMO_LLM_QUICKSTART.md](docs/MMO_LLM_QUICKSTART.md) |
| Framework / recurrent V2 learning stack | Implemented and automated-test covered — see [docs/QUICKSTART_V2.md](docs/QUICKSTART_V2.md) |
| Offline Studio backend | Implemented and core unit-test covered — see [docs/PLAYMIND_STUDIO_QUICKSTART.md](docs/PLAYMIND_STUDIO_QUICKSTART.md) |
| Offline Studio GUI / launchers | Implemented — `scripts/start_studio.py` and `start_playmind_studio.bat` |
| Planner and recurrent trained models | Need real, episode-labeled demonstrations and held-out evaluation |
| Live gameplay improvement | Not measured; no improvement claim |
| Offline visual analysis | Still-image heuristics implemented; visual-model training and GUI review are not |

Without a BC checkpoint, `policy_mode: hybrid` runs **scripted skills** (safe default).

## Choose the correct surface

### Offline Studio — recordings of protected games

Studio processes imported files only. It has no live capture, process access,
physical gameplay input logging, or generated-input surface.

```powershell
winget install Gyan.FFmpeg
.\setup_playmind_studio.ps1
.\start_playmind_studio.bat
# opens http://127.0.0.1:8787/
```

Do not replace the Studio launcher with `start_playmind.bat`; that starts the
separate owned-game lab. Instructions: [START_HERE.md](START_HERE.md).

### Owned-game lab — only games/environments you may automate

```powershell
Copy-Item config\owned_game.example.json config\owned_game.json
.\setup_windows.ps1
.\start_playmind.bat
```

This starts the owned-game Control Center in safe shadow mode with keyboard
input blocked. It is not the Offline Studio and must not be used with the
official World of Warcraft client.
First-time workflow: [MMO LLM planner quickstart](docs/MMO_LLM_QUICKSTART.md).

## Quick start (no extra installs)

```bash
python3 playmind_onefile.py --episodes 5
python3 -m playmind --episodes 5
```

### GUIs

```bash
# Offline Studio (recording import, review, datasets, evaluation)
python3 scripts/start_studio.py
# open http://127.0.0.1:8787

# Demo world + live log
python3 -m playmind.web_gui
# open http://127.0.0.1:8765

# OWNED-GAME LAB brain monitor (not Studio)
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

# Validate demos / train recurrent BC (history length defaults to 16)
python3 scripts/train_behavior_clone.py --history-length 16 --dry-validate-only
python3 scripts/train_behavior_clone.py --history-length 16 \
  --checkpoint models/checkpoints/recurrent_skill_policy.json

# Actuator-free evaluation + diagnostics
python3 scripts/run_evaluation.py \
  --checkpoints models/checkpoints/recurrent_skill_policy.json
python3 scripts/export_diagnostics.py
python3 scripts/migrate_legacy_learning.py --data-dir data/playmind/owned
```

Full flows: [docs/QUICKSTART_V2.md](docs/QUICKSTART_V2.md)

### Local next steps (your PC)

Cloud agents cannot see your game. On the machine that runs it:

```bash
python3 scripts/setup_owned_game.py --window-title "YourGameTitle"
python3 scripts/smoke_local_pipeline.py --device cpu   # optional synthetic smoke
```

Then calibrate ROIs, record **real** demos, re-train, dry-run, and only then `--live`.  
Full checklist: [docs/LOCAL_NEXT_STEPS.md](docs/LOCAL_NEXT_STEPS.md)  

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

- [START_HERE](START_HERE.md) · [STUDIO_QUICKSTART](docs/PLAYMIND_STUDIO_QUICKSTART.md) · [OFFLINE_VIDEO_IMPORT](docs/OFFLINE_VIDEO_IMPORT.md) · [OFFLINE_ANNOTATION](docs/OFFLINE_ANNOTATION.md)
- [DATA_PROVENANCE_AND_PERMISSION](docs/DATA_PROVENANCE_AND_PERMISSION.md) · [REAL_BENCHMARK_BUILDER](docs/REAL_BENCHMARK_BUILDER.md) · [FIRST_MMO_MODEL_TRAINING](docs/FIRST_MMO_MODEL_TRAINING.md)
- [CORRECTION_DRIVEN_LEARNING](docs/CORRECTION_DRIVEN_LEARNING.md) · [LEARNING_PROOF_DASHBOARD](docs/LEARNING_PROOF_DASHBOARD.md) · [RETAIL_WOW_OFFLINE_WORKFLOW](docs/RETAIL_WOW_OFFLINE_WORKFLOW.md) · [ACCOUNT_SAFETY_ARCHITECTURE](docs/ACCOUNT_SAFETY_ARCHITECTURE.md)
- [MMO_LLM_QUICKSTART](docs/MMO_LLM_QUICKSTART.md) · [MMO_LLM_ARCHITECTURE](docs/MMO_LLM_ARCHITECTURE.md) · [HUMAN_DEMONSTRATIONS](docs/HUMAN_DEMONSTRATIONS.md)
- [PLANNER_DATASET](docs/PLANNER_DATASET.md) · [PLANNER_TRAINING](docs/PLANNER_TRAINING.md) · [PLANNER_EVALUATION](docs/PLANNER_EVALUATION.md) · [MODEL_PROMOTION](docs/MODEL_PROMOTION.md)
- [WINDOWS_SETUP](docs/WINDOWS_SETUP.md) · [WSL2_TRAINING](docs/WSL2_TRAINING.md) · [TROUBLESHOOTING](docs/TROUBLESHOOTING.md)
- [LOCAL_NEXT_STEPS](docs/LOCAL_NEXT_STEPS.md) · [QUICKSTART_V2](docs/QUICKSTART_V2.md) · [LEARNING_ARCHITECTURE_V2](docs/LEARNING_ARCHITECTURE_V2.md) · [RECURRENT_POLICY](docs/RECURRENT_POLICY.md)
- [SKILL_COMMITMENT](docs/SKILL_COMMITMENT.md) · [EPISODE_LIFECYCLE](docs/EPISODE_LIFECYCLE.md) · [FEATURE_SCHEMA](docs/FEATURE_SCHEMA.md)
- [DEMONSTRATION_RECORDING](docs/DEMONSTRATION_RECORDING.md) · [TRAINING](docs/TRAINING.md) · [EVALUATION](docs/EVALUATION.md)
- [SENSOR_LABELING](docs/SENSOR_LABELING.md) · [MIGRATION](docs/MIGRATION.md)
- [FULL_STACK](docs/FULL_STACK.md) · [COMPLIANCE_BOUNDARIES](docs/COMPLIANCE_BOUNDARIES.md)

## Layout

```text
playmind/     agent, owned loop, offline Studio, skills, policies, training, GUI
scripts/      owned loop, BC train/eval, diagnostics, teleop
config/       owned_game + keymap examples
docs/         architecture + V2 guides
tests/
```

## Compliance

Offline access is not automatic permission to train or redistribute. Use only
recordings you are authorized to possess and use; record consent/license,
exclude unknown or unreviewed data, protect personal information, and honor
deletion/attribution terms. Never use PlayMind to automate the official World
of Warcraft client, read its memory, inject code, manipulate packets, or evade
anti-cheat. See [Compliance Boundaries](docs/COMPLIANCE_BOUNDARIES.md) and
[Data Provenance and Permission](docs/DATA_PROVENANCE_AND_PERMISSION.md).

## License

MIT — see [LICENSE](LICENSE)
