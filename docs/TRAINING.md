# Recurrent behavior-cloning training

The default trainer builds a `RecurrentSkillPolicyV2` from episode-local demonstration sequences. The path is implemented and tested, but the repository does not include evidence that a trained policy improves live gameplay.

## Prerequisites

1. Record demos — [DEMONSTRATION_RECORDING.md](./DEMONSTRATION_RECORDING.md)
2. Install PyTorch for training. Dry validation does not require it.

## Commands

```bash
# Validate default 16-step windows (no torch required)
PYTHONPATH=. python3 scripts/train_behavior_clone.py --dry-validate-only \
  --data-dir data/playmind/demonstrations \
  --history-length 16

# Train recurrent policy (the default --model-type)
PYTHONPATH=. python3 scripts/train_behavior_clone.py \
  --data-dir data/playmind/demonstrations \
  --model-type recurrent \
  --history-length 16 \
  --batch-size 32 \
  --epochs 30 \
  --checkpoint models/checkpoints/recurrent_skill_policy.json

# Resume compatible recurrent checkpoint/training state
PYTHONPATH=. python3 scripts/train_behavior_clone.py \
  --data-dir data/playmind/demonstrations \
  --history-length 16 --resume \
  --checkpoint models/checkpoints/recurrent_skill_policy.json
```

Point hybrid mode at the checkpoint:

```json
"learning_v2": {
  "enabled": true,
  "policy_mode": "hybrid",
  "history_length": 16,
  "bc_checkpoint": "models/checkpoints/recurrent_skill_policy.json",
  "confidence_threshold": 0.45,
  "device": "cpu",
  "seed": 0
}
```

## Notes

- Recurrent training uses feature schema v2 (`FEATURE_SCHEMA_VERSION=2`), a 54-D value/known/confidence representation.
- Train/validation/test assignment is episode-wise; windows never cross `(session_id, episode_id)` boundaries.
- Bad and unlabeled samples are excluded by default. The CLI does not expose overrides.
- Windows are left-padded to 16 by default and carry a validity mask; `--stride` and `--min-sequence-length` control window generation.
- Normalization statistics are fit on the training split only and stored in checkpoint metadata.
- Validation loss selects the checkpoint; the test split is evaluated after selection. Metrics are also appended to `data/playmind/training/metrics.csv` by default.
- `--device auto` selects CUDA when available, otherwise CPU. `--no-amp` disables CUDA mixed precision.
- `--model-type mlp` remains available only for explicit legacy comparisons; it does not use recurrent schema-v2 sequences.

See [RECURRENT_POLICY.md](./RECURRENT_POLICY.md), [FEATURE_SCHEMA.md](./FEATURE_SCHEMA.md), and [EVALUATION.md](./EVALUATION.md).
