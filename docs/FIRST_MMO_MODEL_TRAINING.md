# First offline MMO planner model training

This workflow trains a planner from reviewed offline recording annotations.
It does not train a vision model, control a live client, or establish gameplay
improvement.

## Evidence/status

| Part | State |
|---|---|
| SFT/DPO configuration, manifests, registry, smoke artifacts | Implemented and automated-test covered |
| Studio reviewed-data export | Implemented and unit-tested at core boundaries |
| Real Hugging Face/TRL training | Implemented, dependency/hardware dependent |
| RTX 4070 Ti 3B QLoRA run | Not tested on this CI host |
| 7B/12 GB preset | Experimental; may OOM |
| SFT-adapter-to-DPO chaining | Deferred; DPO currently starts from `--base-model` |

## 1. Install the training stack

```bash
python -m pip install -r requirements-playmind-ml.txt
python -m pip install trl bitsandbytes
```

The trainer imports `torch`, `transformers`, `datasets`, `peft`, `trl`, and
`accelerate`; CUDA QLoRA additionally needs working `bitsandbytes`. Install the
PyTorch build compatible with the host driver/CUDA.

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY
```

Review the base model's authoritative model card, license, acceptable-use
terms, architecture compatibility, and redistribution rules. Local
`config.json` license metadata is informative, not a legal determination.

## 2. Require real reviewed data

Studio export paths are:

```text
data/playmind/planner/sft/studio/{train,val,test}.jsonl
data/playmind/planner/preferences/studio/{train,val,test}.jsonl
data/playmind/planner/manifests/studio/
```

Inspect line counts and manifests. Confirm:

- no ineligible/rejected project;
- no leakage violation;
- benchmark source projects are absent from training;
- examples contain real reviewed plans rather than synthetic/suggested labels;
- validation is non-empty and source-disjoint;
- at least one immutable real benchmark version exists.

The readiness defaults classify 10 reviewed examples plus a frozen benchmark
as experimental; normal defaults are 1,000 reviewed examples, 100 preference
examples, a frozen benchmark, eligible provenance/license, no leakage, at
least 2 GiB free, and a GPU. These are engineering gates, not quality proof.

## 3. Smoke first

```bash
python scripts/train_planner_sft.py \
  --preset cpu_tiny_smoke \
  --smoke \
  --max-steps 2 \
  --run-id first-studio-smoke
```

Smoke mode does not read the supplied real train file. It writes synthetic
rows and `adapter/smoke_artifact.json`, sets `smoke: true`, and registers a
restricted candidate unless `--no-register` is used.

**SMOKE / NO REAL WEIGHTS:** never export, evaluate as a learned candidate, or
compare smoke loss to real training. Registry promotion hard-rejects smoke
artifacts, even with manual override.

`--dry-run` invokes the same synthetic smoke path; it is not configuration-only
validation.

## 4. Run the 4070 Ti 3B preset

```bash
python scripts/train_planner_sft.py \
  --base-model /path/to/licensed-3b-hf-model \
  --preset rtx_4070_ti_3b_qlora \
  --train-file data/playmind/planner/sft/studio/train.jsonl \
  --eval-file data/playmind/planner/sft/studio/val.jsonl \
  --run-id first-studio-3b-sft \
  --seed 42 \
  --early-stopping-patience 2 \
  --no-register
```

Verified SFT CLI flags:

```text
--base-model --preset --train-file --eval-file --runs-root --registry-path
--run-id --seed --max-steps --early-stopping-patience
--resume-from-checkpoint [PATH] --smoke --dry-run --no-register
```

The preset uses 1024 tokens, microbatch 1, accumulation 16, gradient
checkpointing, rank 16/alpha 32, dropout 0.05, three epochs, and 4-bit NF4 with
BF16 when supported (otherwise FP16).

Stop if the trainer warns that QLoRA is unavailable and falls back to full
LoRA. That path loads much larger FP16/FP32 weights and is likely to OOM on a
12 GB GPU. There is no CLI sequence-length override; reducing it requires a
code/custom-preset change.

To resume:

```bash
python scripts/train_planner_sft.py \
  --base-model /path/to/licensed-3b-hf-model \
  --preset rtx_4070_ti_3b_qlora \
  --train-file data/playmind/planner/sft/studio/train.jsonl \
  --eval-file data/playmind/planner/sft/studio/val.jsonl \
  --run-id first-studio-3b-sft-resumed \
  --resume-from-checkpoint models/playmind/runs/first-studio-3b-sft/checkpoints/<checkpoint>
```

Use a new run ID because run directories are created exclusively.

## 5. Inspect artifacts

```text
models/playmind/runs/<run-id>/
├── adapter/
├── checkpoints/
├── metrics.csv
└── training_manifest.json
```

Check `status`, `smoke`, dataset path, base model/license metadata,
quantization, fallback message, preset, train/validation metrics, and registry
ID. Without `--no-register`, training registers `candidate`; it never sets
`production`. The command above deliberately defers registration because the
generic trainer does not propagate protected Studio lineage.

Falling train loss proves fitting only. Stop or investigate when validation
loss worsens, outputs collapse to one plan, illegal skills rise, or results are
sensitive to trivial wording changes.

## 6. Optional DPO

Reviewed corrections export preference splits:

```bash
python scripts/train_planner_dpo.py \
  --base-model /path/to/licensed-3b-hf-model \
  --preset rtx_4070_ti_3b_qlora \
  --train-file data/playmind/planner/preferences/studio/train.jsonl \
  --eval-file data/playmind/planner/preferences/studio/val.jsonl \
  --beta 0.1 \
  --run-id first-studio-3b-dpo
```

Verified DPO flags are `--base-model`, `--preset`, `--train-file`,
`--eval-file`, `--runs-root`, `--registry-path`, `--run-id`, `--seed`,
`--beta`, `--max-steps`, `--resume-from-checkpoint [PATH]`, `--smoke`,
`--dry-run`, and `--no-register`.

The implemented CLI does not accept an SFT adapter. Do not describe this as
SFT-then-DPO continuation.

## 7. Evaluate before any judgment

Expose the actual adapter/merged artifact as an Ollama model, then register the
exact Ollama tag as `display_name` on a separate candidate with
`live_use_prohibited=True`, profile `retail_wow_offline_only`, and allowed use
`offline_evaluation`. Run the frozen real suite against that restricted ID.
The evaluator does not load `adapter_path` directly and otherwise prefers
`display_name` before `base_model`.
Follow [Real Benchmark Builder](./REAL_BENCHMARK_BUILDER.md) and
[Learning Proof Dashboard](./LEARNING_PROOF_DASHBOARD.md).

For the protected retail-WoW offline profile, keep derived models marked
`live_use_prohibited`; they cannot be promoted to production by the registry.
