# Models

- `Modelfile.playmind` — generated few-shot Ollama Modelfile (safe to regenerate)
- `playmind-lora/` — optional LoRA output from `scripts/finetune_lora.py` (gitignored)

```bash
ollama create playmind-planner -f models/Modelfile.playmind
```
