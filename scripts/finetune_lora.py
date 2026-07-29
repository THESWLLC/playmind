#!/usr/bin/env python3
"""Optional LoRA fine-tune entrypoint (requires torch + transformers + peft + datasets).

This script exits with install instructions if ML deps are missing.
It is intentionally conservative and trains on PlayMind finetune.jsonl only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/playmind/finetune.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out-dir", default="models/playmind-lora")
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError:
        raise SystemExit(
            "Missing ML deps. Install (GPU recommended):\n"
            "  pip install torch transformers datasets peft accelerate\n"
            "Or use the lighter path:\n"
            "  python scripts/build_ollama_modelfile.py\n"
            "  ollama create playmind-planner -f models/Modelfile.playmind"
        )

    rows = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            msgs = row["messages"]
            text = ""
            for m in msgs:
                text += f"<|{m['role']}|>\n{m['content']}\n"
            rows.append({"text": text})

    ds = Dataset.from_list(rows)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=512)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = get_peft_model(
        model,
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"]),
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    training = TrainingArguments(
        output_dir=str(out),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        logging_steps=5,
        save_steps=args.max_steps,
        learning_rate=1e-4,
        report_to=[],
    )

    def collate(features):
        batch = {k: torch.tensor([f[k] for f in features]) for k in features[0] if k != "text"}
        batch["labels"] = batch["input_ids"].clone()
        return batch

    trainer = Trainer(model=model, args=training, train_dataset=ds, data_collator=collate)
    trainer.train()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"Saved LoRA adapter to {out}")


if __name__ == "__main__":
    main()
