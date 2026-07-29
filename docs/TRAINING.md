# Behavior-cloning training

Train (or dry-validate) a skill policy from demonstration sessions. CUDA is optional; without PyTorch the CLI still validates the dataset and exits 0.

## Prerequisites

1. Record demos — [DEMONSTRATION_RECORDING.md](./DEMONSTRATION_RECORDING.md)
2. Optional: `pip install torch` for the training skeleton

## Commands

```bash
# Dry-validate demos (no torch required)
PYTHONPATH=. python3 scripts/train_behavior_clone.py --dry-validate-only \
  --data-dir data/playmind/demonstrations

# Train skeleton → checkpoint metadata (uses torch when installed)
PYTHONPATH=. python3 scripts/train_behavior_clone.py \
  --data-dir data/playmind/demonstrations \
  --window-size 4 \
  --batch-size 8 \
  --epochs 1 \
  --checkpoint models/checkpoints/skill_policy_v2.json
```

Point hybrid mode at the checkpoint:

```json
"learning_v2": {
  "enabled": true,
  "policy_mode": "hybrid",
  "bc_checkpoint": "models/checkpoints/skill_policy_v2.json",
  "confidence_threshold": 0.45,
  "device": "cpu",
  "seed": 0
}
```

## Notes

- Splits are **episode-wise** (`train` / `val` / `test`) via `DemonstrationDataset`
- Untrained models report low confidence so HybridPolicy falls back to scripted skills
- Device defaults to `cpu`; set `"device": "cuda"` only when CUDA torch is available

See also: [EVALUATION.md](./EVALUATION.md) · [SKILLS.md](./SKILLS.md)
