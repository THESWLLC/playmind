#!/usr/bin/env python3
"""Build an Ollama Modelfile few-shot adapter from finetune.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM = (
    "You are PlayMind, an action planner for an owned video game. "
    "Given a JSON game state, reply with exactly one action name."
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/playmind/finetune.jsonl")
    p.add_argument("--base-model", default="dolphin-llama3")
    p.add_argument("--out", default="models/Modelfile.playmind")
    p.add_argument("--examples", type=int, default=12)
    p.add_argument("--name", default="playmind-planner")
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}. Run scripts/self_play_train.py first.")

    examples = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            if len(examples) >= args.examples:
                break
            row = json.loads(line)
            msgs = row.get("messages", [])
            if len(msgs) < 3:
                continue
            user = msgs[-2]["content"]
            assistant = msgs[-1]["content"]
            examples.append((user, assistant))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"FROM {args.base_model}",
        f'SYSTEM """{SYSTEM}"""',
        "PARAMETER temperature 0.2",
        "",
    ]
    for user, assistant in examples:
        lines.append("MESSAGE user " + json.dumps(user))
        lines.append("MESSAGE assistant " + json.dumps(assistant))
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out} with {len(examples)} examples")
    print("Create locally with:")
    print(f"  ollama create {args.name} -f {out}")
    print(f"  ollama run {args.name}")


if __name__ == "__main__":
    main()
