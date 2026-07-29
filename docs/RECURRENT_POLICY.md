# Recurrent skill policy

## Status

- **Implemented and unit-tested on this branch:** structured sequence encoding, GRU inference, padding masks, pre-softmax skill masking, checkpoint round-trips, and schema rejection.
- **Needs real demonstrations:** training a useful checkpoint and measuring it on held-out episodes.
- **Not established:** live gameplay improvement or visual learning. The current model is structured-input only.

## Architecture

`RecurrentSkillPolicyV2` consumes schema-v2 structured features:

```text
(B,T,54) features
  → Linear + LayerNorm + ReLU encoder (default 96-D)
  → GRU (default 128 hidden units, one layer)
  → skill logits + auxiliary heads
```

The skill head classifies the final valid GRU state. Auxiliary heads cover target validity, combat, death, HP/progress deltas, and skill success when labels are available. Invalid skills are masked at the logits before softmax.

## Stateless last-16 live inference

The controller keeps a rolling `TemporalHistory` and supplies at most the latest `history_length` observations; the default is 16. `BehaviorCloningPolicy` resets recurrent state before each decision, so live inference recomputes from that bounded window instead of carrying a hidden state indefinitely. This avoids persistent hidden-state leakage. Stateful support exists in the network API but is not the default live path.

Training windows are episode-local, left-padded with zeros, and accompanied by a `True == valid` padding mask. Padding is ignored by the GRU.

## Checkpoints

A recurrent checkpoint is JSON metadata plus a sibling `.pt` weights file. Metadata includes:

- `checkpoint_schema_version`, `model_type`, `model_version`
- `feature_schema_version`, ordered `feature_names`, `feature_dim`
- `skill_names`, `aux_names`, `history_length`
- `architecture`: encoder/hidden dimensions, layers, dropout, directionality, stateful flag
- `training_config`, train-split `normalization_stats`
- `input_modalities` (`["structured"]`), `trained`, `seed`, `git_commit`, `weights_file`

The loader requires `model_type: recurrent_skill_policy`, checkpoint schema 2, feature schema 2, and the exact feature ordering. Metadata alone is not evidence of model quality; retain the `.pt` file and evaluate the checkpoint.

## Legacy MLP versus recurrent

| | Legacy `SkillPolicyV2` | `RecurrentSkillPolicyV2` |
|---|---|---|
| Input | Schema-v1 38-D, last frame | Schema-v2 54-D sequence |
| Temporal model | None | GRU over valid timesteps |
| Checkpoint type | `structured_mlp_legacy` | `recurrent_skill_policy` |
| Loading | `SkillPolicyV2.load()` / legacy adapter | `RecurrentSkillPolicyV2.load()` |

Weights are never reinterpreted across architectures. Hybrid loading detects checkpoint type and falls back to scripted behavior on missing or incompatible files.

## Train and evaluate

```bash
# Validate episode-local windows without training
PYTHONPATH=. python3 scripts/train_behavior_clone.py \
  --data-dir data/playmind/demonstrations \
  --history-length 16 --dry-validate-only

# Recurrent is the default model type
PYTHONPATH=. python3 scripts/train_behavior_clone.py \
  --data-dir data/playmind/demonstrations \
  --history-length 16 --epochs 30 \
  --checkpoint models/checkpoints/recurrent_skill_policy.json

# Sequence-aware held-out classification evaluation
PYTHONPATH=. python3 scripts/evaluate_behavior_clone.py \
  --data-dir data/playmind/demonstrations \
  --checkpoint models/checkpoints/recurrent_skill_policy.json \
  --history-length 16 --split test \
  --json-out data/playmind/evaluation/recurrent-test.json

# Comparative dry replay and outcome-section report
PYTHONPATH=. python3 scripts/run_evaluation.py \
  --data-dir data/playmind/demonstrations \
  --checkpoints models/checkpoints/recurrent_skill_policy.json \
  --output-dir data/playmind/evaluation/recurrent-comparison
```

PyTorch is required for training and weighted inference; dry dataset validation remains available without it.

## Limitations

- No bundled real demonstration corpus or measured live improvement is claimed.
- `run_evaluation.py` is actuator-free replay. Its policy choices do not alter recorded outcomes, and its generic replay path does not reconstruct recurrent training windows.
- The live controller bounds history but does not currently clear `TemporalHistory` at each lifecycle edge, so a short window can include observations from both sides of an episode transition.
- Visual encoders are placeholders; frames are not learned by this model.
- Sensor quality, labels, class balance, and episode boundaries determine model quality.
- CPU training works but can be slow; CUDA is optional.
