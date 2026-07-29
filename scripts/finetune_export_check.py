#!/usr/bin/env python3
"""Validate and summarize finetune.jsonl for local LLM training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="data/playmind/finetune.jsonl")
    args = p.parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python -m playmind --episodes 10")

    actions: Counter[str] = Counter()
    n = 0
    bad = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                msg = row["messages"][-1]["content"]
                actions[msg] += 1
                n += 1
            except Exception:
                bad += 1
    print(f"samples={n} bad_lines={bad}")
    print("top actions:")
    for action, count in actions.most_common(12):
        print(f"  {action:16} {count}")
    print(
        "\nNext: fine-tune a local model (Unsloth/Axolotl/Ollama create) using this JSONL."
    )


if __name__ == "__main__":
    main()
