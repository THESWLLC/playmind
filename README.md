# PlayMind

Local AI agent framework for **games you own** (or are allowed to automate).

PlayMind can:
- Play a built-in demo world
- Learn from its own runs (tabular policy + experience logs)
- Optionally ask you questions in teach mode
- Optionally use a local Ollama LLM as planner
- Export fine-tune JSONL for your own model later

> **Not for World of Warcraft or other ToS-restricted live clients.**  
> Do not point this at official MMO clients with automated keyboard/mouse control.

## Quick start (no extra installs)

```bash
python3 playmind_onefile.py --episodes 5
```

Or the package CLI:

```bash
python3 -m playmind --episodes 5
```

### Teach / vision / actuators / training

```bash
# A) Teach mode — agent asks you while playing
python3 -m playmind --teach --episodes 1

# C) Vision path (demo ASCII frames + quest text parse)
python3 -m playmind --vision --episodes 3

# B/D) Keyboard actuator stubs (no OS key injection by default)
python3 -m playmind --actuator dry-run --episodes 1

# E) Self-play to grow learning data + fine-tune export
PYTHONPATH=. python3 scripts/self_play_train.py --episodes 50
PYTHONPATH=. python3 scripts/finetune_export_check.py
```

Optional local LLM planner (requires [Ollama](https://ollama.com)):

```bash
ollama pull dolphin-llama3
python3 -m playmind --ollama --episodes 1 --interactive
```

More detail: [docs/NEXT_STACK.md](docs/NEXT_STACK.md)

## What gets saved

Under `data/playmind/`:
- `policy.json` — learned action values
- `experience.jsonl` — self-play + teacher labels
- `finetune.jsonl` — export for later LLM fine-tuning

## Project layout

```text
playmind/           Agent, demo world, learning, planners
playmind_onefile.py Shareable single-file demo
docs/               Research + setup notes
tests/              Automated tests
```

## Create the GitHub repo

This cloud environment has **no GitHub login**, so publish from your machine:

```bash
# from this project root
gh auth login
gh repo create playmind --private --source=. --remote=origin --push
```

Details: [docs/GITHUB_SETUP.md](docs/GITHUB_SETUP.md)

## Roadmap

1. ~~Demo world + self-learning + teach mode~~
2. ~~Vision frame path + actuator stubs + fine-tune export~~
3. Real window capture + OCR on **your** game screenshots
4. Enable Parsec/keyboard backend for **your** game only
5. Fine-tune local LLM on `finetune.jsonl`

## License

MIT — see [LICENSE](LICENSE)
