# Learning proof dashboard

The learning-proof view is an evidence checklist, not a “model learned” badge.
Use it to connect source permission, reviewed examples, training manifests,
held-out evaluation, and human judgment.

## Current implementation

- Evaluator report normalization and canonical
  `data/playmind/planner/evaluation/index.json`: implemented and unit-tested.
- Evaluator writes legacy timestamped reports plus
  `runs/<run-id>/report.json`: implemented.
- Existing owned-game GUI learning-proof foundation: implemented, but separate
  from Offline Studio.
- Offline Studio **Learning Proof** view: newly implemented. It reads the latest
  normalized report and registry, excludes smoke as proof, and labels offline
  evidence.
- Automatic proof verdict: implemented as a simple candidate-score greater-than
  production comparison when comparable scores exist.
- Automatic promotion: intentionally not implemented.

Start it with `scripts/start_studio.py` / `start_playmind_studio.bat`. Use the
manual files/commands below to audit what the dashboard summarizes.

## Refresh and inspect evaluation discovery

Evaluation refreshes the index automatically. To rebuild it:

```bash
python - <<'PY'
import json
from playmind.studio.eval_index import write_index
print(json.dumps(write_index(), indent=2))
PY
```

Artifacts:

```text
data/playmind/planner/evaluation/
├── index.json
├── planner_benchmark_<timestamp>.json
├── planner_benchmark_<timestamp>.csv
├── planner_benchmark_<timestamp>.md
└── runs/planner_benchmark_<timestamp>/report.json
```

The index normalizes evaluator `backends` into `comparisons`, so Studio and the
owned-game GUI can discover new and legacy reports. It exposes `smoke` when the
report contains that field; current planner evaluation does not itself add a
smoke flag.

The dashboard's `YES` means only that the selected real candidate score exceeds
the selected production score in comparable recorded evidence. It does not by
itself enforce 100 scenarios, category coverage, source separation, statistical
significance, or human acceptance. Inspect the report and gates yourself.

## Evidence ladder

| Level | Required evidence | What it establishes |
|---|---|---|
| 0. Data | eligible provenance, human-reviewed labels, source-safe splits | data may enter this experiment |
| 1. Smoke | successful synthetic smoke manifest labeled `smoke: true` | command/artifact/registry plumbing only |
| 2. Fit | real run manifest, non-smoke adapter, train and validation metrics | model fitting occurred |
| 3. Held out | frozen real suite, candidate vs scripted/generic report | offline benchmark evidence |
| 4. Replicated | gains on a second untouched source group/run seed | reduced chance of one-suite luck/leakage |
| 5. Judged | blind human failure review and documented acceptance/rejection | practical offline judgment |

No level proves improvement in a live game. The protected retail-WoW profile
prohibits live/generated-input use.

## Read training proof

Open `models/playmind/runs/<run-id>/training_manifest.json` and verify:

- `status` is `completed`;
- `smoke` is `false` for a real claim;
- dataset path is the intended reviewed split;
- base model and license review match the run;
- quantization is expected and fallback message is empty;
- adapter path exists and is not `smoke_artifact.json`;
- validation metrics exist and do not diverge badly;
- registry ID resolves to this run.

Smoke manifests are **SMOKE / NO REAL WEIGHTS** regardless of reassuring loss
or completion fields.

## Read evaluation proof

```bash
python - <<'PY'
import json
from pathlib import Path

index = json.loads(Path(
    "data/playmind/planner/evaluation/index.json"
).read_text(encoding="utf-8"))
latest = index["reports"][0]
print(json.dumps(latest, indent=2))
PY
```

For each backend inspect:

- `valid_plan_rate`;
- `illegal_skill_rate`;
- `json_fail_rate`;
- `first_skill_correct`;
- `full_plan_score`;
- aggregate `benchmark_score`;
- p50/p95 latency;
- duplicate-output, diversity, and exact-match signals;
- promotion gate errors and scenario count.

Open the CSV and review every candidate error. Aggregate gains can hide a
critical death/recovery regression or a baseline outage.

## Promotion gates versus proof

Default registry gates require:

- valid plan rate at least 0.99;
- illegal skill rate at most 0.005;
- at least 100 scenarios; and
- benchmark score strictly better than production.

Passing gates does not promote; evaluation leaves status unchanged. The built-
in 18-scenario synthetic suite cannot pass the count gate.

Smoke and `live_use_prohibited` artifacts are hard-blocked from promotion even
with manual override. This is expected for models derived under
`retail_wow_offline_only`.

## Personal judgment checklist

- Was the benchmark frozen before this candidate was tuned?
- Are train/validation/benchmark source groups disjoint?
- Were expected plans written without seeing candidate output?
- Are alternatives broad enough to avoid penalizing valid plans?
- Does the candidate improve meaningful categories, not only formatting?
- Are all illegal skills and lifecycle failures understood?
- Does a second reviewer agree on disputed examples?
- Does improvement repeat on untouched projects and seeds?
- Is output diversity plausible rather than collapsed?
- Can you explain what remains untested?

Record a decision as “accepted for further offline experiment,” “needs
corrections,” or “rejected.” Avoid “proven for gameplay.”

See [First Model Training](./FIRST_MMO_MODEL_TRAINING.md),
[Real Benchmark Builder](./REAL_BENCHMARK_BUILDER.md), and
[Correction-Driven Learning](./CORRECTION_DRIVEN_LEARNING.md).
