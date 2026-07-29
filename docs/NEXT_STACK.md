# PlayMind next stack (implemented)

## A — Teach mode
```bash
python3 -m playmind --teach --episodes 1
```
Commands at the prompt: action name, `dir farm`, `accept`, `skip`, `quit`.

## C — Vision / OCR
```bash
python3 -m playmind --vision --episodes 3
```
Demo writes ASCII frames to `data/playmind/frames/latest.txt` and parses quest text.
For real screenshots later: point `read_frame()` at a PNG (optional Pillow + pytesseract).

## B/D — Game / Parsec actuators
```bash
python3 -m playmind --actuator dry-run --episodes 1
python3 -m playmind --actuator parsec-stub --episodes 1
```
- `demo` — in-process world (default)
- `dry-run` — logs intended keys to JSONL (no OS input)
- `parsec-stub` — placeholder for owned-game Parsec control (disabled)

## E — Self-play + fine-tune export
```bash
PYTHONPATH=. python3 scripts/self_play_train.py --episodes 50
PYTHONPATH=. python3 scripts/finetune_export_check.py
```
Uses `data/playmind/finetune.jsonl` for a future local LLM fine-tune (Ollama/Unsloth/Axolotl).

## Safety
Owned games / demo only. Do not wire actuators to official MMO clients.
