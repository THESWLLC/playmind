# PlayMind full stack

## What’s included now

| Piece | How to run |
|-------|------------|
| Demo agent | `python3 -m playmind --episodes 5` |
| Teach mode | `python3 -m playmind --teach --episodes 1` |
| Vision frames (demo) | `python3 -m playmind --vision --episodes 3` |
| Screen capture | `PYTHONPATH=. python3 scripts/capture_once.py --ocr` |
| Owned-game loop (dry-run) | copy config → set `i_own_this_game` → `scripts/run_owned_loop.py` |
| Session breaks | configured in `config/owned_game.json` `session` block |
| Self-play data | `scripts/self_play_train.py` |
| Ollama Modelfile | `scripts/build_ollama_modelfile.py` |
| Optional LoRA train | `scripts/finetune_lora.py` (needs torch/peft) |

## Owned-game setup

1. Copy `config/owned_game.example.json` → `config/owned_game.json`
2. Set `"i_own_this_game": true` only for a game you own
3. Calibrate capture / `hp_roi`
4. Edit keymap
5. Dry-run first:
   ```bash
   PYTHONPATH=. python3 scripts/run_owned_loop.py --config config/owned_game.json --max-ticks 10
   ```
6. Live keys (optional):
   ```bash
   pip install pynput mss Pillow pytesseract
   # set enable_keyboard true
   PYTHONPATH=. python3 scripts/run_owned_loop.py --config config/owned_game.json --live --max-ticks 10
   ```

## Local LLM

```bash
PYTHONPATH=. python3 scripts/self_play_train.py --episodes 50
PYTHONPATH=. python3 scripts/build_ollama_modelfile.py
ollama create playmind-planner -f models/Modelfile.playmind
ollama run playmind-planner
```

## Hard limits

- Default paths **do not** send OS keys
- Owned loop refuses to start unless `i_own_this_game=true`
- Do not use with official WoW / restricted MMO clients
