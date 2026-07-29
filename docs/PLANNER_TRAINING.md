# Planner training

Planner SFT and DPO entry points are implemented and automated-test covered
through dependency-free synthetic smoke runs. Real GPU QLoRA is hardware- and
model-dependent and has not been validated by CI.

## Dependencies

Python 3.10+ is required. Core recording uses
`requirements-playmind.txt`. Real planner training needs:

```bash
python -m pip install -r requirements-playmind-ml.txt
python -m pip install trl bitsandbytes
```

The trainer imports `torch`, `transformers`, `datasets`, `peft`, `trl`, and
`accelerate`; CUDA QLoRA additionally needs a working CUDA-enabled
`bitsandbytes`. `trl` and `bitsandbytes` are currently not listed in
`requirements-playmind-ml.txt`, so install them explicitly. Install a PyTorch
build compatible with the host NVIDIA driver/CUDA environment.

Always review the base model's authoritative license and terms. A local
`config.json` license field is copied into the run manifest when available, but
that is not a legal determination.

## Presets

| Preset | Intended use | Key settings |
|---|---|---|
| `cpu_tiny_smoke` | Synthetic pipeline check | 128 tokens, microbatch 1, no quantization |
| `rtx_4070_ti_3b_qlora` | 12 GB RTX 4070 Ti | 1024 tokens, microbatch 1, accumulation 16, checkpointing, rank 16/alpha 32, NF4 |
| `rtx_4070_ti_7b_qlora_experimental` | Experimental 12 GB attempt | 768 tokens, accumulation 32; may OOM |

GPU presets intentionally have no default model. Pass `--base-model` after
choosing a compatible, licensed causal language model.

## Smoke versus real training

```bash
# Dependency-free synthetic artifact; does not read train.jsonl.
python3 scripts/train_planner_sft.py --smoke --max-steps 2

# Real SFT using episode-safe planner exports.
python3 scripts/train_planner_sft.py \
  --base-model /path/to/licensed-3b-hf-model \
  --preset rtx_4070_ti_3b_qlora \
  --train-file data/playmind/planner/sft/train.jsonl \
  --eval-file data/playmind/planner/sft/val.jsonl
```

`--dry-run` is an alias for the same synthetic smoke path; it is not a
configuration-only validation. Smoke mode silently switches the default 4070
preset to `cpu_tiny_smoke` when no base model was provided.

Every successful run writes:

```text
models/playmind/runs/<run-id>/
├── adapter/
├── checkpoints/              # real training
├── metrics.csv
└── training_manifest.json
```

It registers the adapter as `candidate` in
`data/playmind/planner/registry.sqlite` unless `--no-register` is passed.
Training never sets production status.

Real SFT applies loss only to the assistant completion, uses validation loss
for best-model selection when validation data exists, supports early stopping,
and supports `--resume-from-checkpoint [PATH]`.

## DPO

Create preference splits first, then:

```bash
# Synthetic plumbing check
python3 scripts/train_planner_dpo.py --smoke --max-steps 2

# Real DPO
python3 scripts/train_planner_dpo.py \
  --base-model /path/to/licensed-3b-hf-model \
  --preset rtx_4070_ti_3b_qlora \
  --train-file data/playmind/planner/preferences/train.jsonl \
  --eval-file data/playmind/planner/preferences/val.jsonl \
  --beta 0.1
```

The implemented DPO CLI starts from `--base-model`; it does not accept or chain
an SFT adapter. Therefore it is not yet a complete “SFT adapter then DPO”
workflow. Smoke DPO writes a synthetic placeholder candidate. If ML
dependencies are missing, real DPO writes a `status: skipped` manifest and
returns nonzero.

## QLoRA behavior and memory

With CUDA and `bitsandbytes`, the 4070 presets load the base model in 4-bit NF4
with double quantization and train LoRA adapters on all linear layers. Compute
uses BF16 when supported, otherwise FP16.

If CUDA or `bitsandbytes` is unavailable, the current trainer falls back to
full LoRA with FP16/FP32 weights and emits a warning. That fallback uses much
more memory and is likely to OOM on a 12 GB card; treat the warning as a setup
failure for the 4070 preset.

For CUDA OOM:

1. Use the 3B preset, not experimental 7B.
2. Keep microbatch at 1.
3. Reduce `max_seq_length` in a copied/custom preset; the current CLI has no
   direct sequence-length override.
4. Close GPU-heavy applications and verify no stale trainer remains.
5. Increase gradient accumulation only when changing the preset in code.
6. Do not mistake the non-QLoRA fallback for a memory-saving path.

## WSL2

WSL2 is recommended for NVIDIA QLoRA on Windows. Use the Windows host driver;
do not install a separate Linux NVIDIA kernel driver inside WSL. Confirm both
`nvidia-smi` and `torch.cuda.is_available()` before training. See
[WSL2_TRAINING.md](./WSL2_TRAINING.md).

Evaluation and explicit promotion are mandatory after training. See
[PLANNER_EVALUATION.md](./PLANNER_EVALUATION.md) and
[MODEL_PROMOTION.md](./MODEL_PROMOTION.md).
