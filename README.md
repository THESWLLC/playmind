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

Teach mode (agent asks you when unsure):

```bash
python3 -m playmind --teach --interactive --episodes 1
```

Optional local LLM planner (requires [Ollama](https://ollama.com)):

```bash
ollama pull dolphin-llama3
python3 -m playmind --ollama --episodes 1 --interactive
```

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

1. Demo world + self-learning + teach mode *(this repo)*
2. Window capture + OCR/CV adapters
3. Parsec/keyboard actuator for **your** game
4. Fine-tune local LLM on `finetune.jsonl`

## License

MIT — see [LICENSE](LICENSE)
