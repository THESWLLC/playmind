# Planner evaluation

The planner benchmark is actuator-free. It compares plan text against frozen or
user-supplied expected scenarios; it does not execute plans or establish live
gameplay improvement.

## Run it

Start Ollama and make the requested model tags available:

```bash
ollama list
python3 scripts/evaluate_planner.py \
  --generic-model llama3.2 \
  --candidate-id <registry-model-id> \
  --output-dir data/playmind/planner/evaluation
```

`--suite PATH` accepts JSONL scenarios. Without it, the command uses 18
in-code, frozen synthetic scenarios. `--ollama-host`, `--timeout`, and
`--registry-path` select the service and registry.

## Backends compared

The default command always includes:

- `scripted` — deterministic existing policy; no Ollama dependency.
- `ollama_generic` — `--generic-model` through Ollama.

It additionally includes registry `production` and `candidate` records when
available. For a registry row, the evaluator chooses the first available value
from `gguf_path`, `merged_path`, `display_name`, or `base_model` as the Ollama
model argument. It does **not** load `adapter_path`. A freshly trained adapter
therefore is not automatically evaluated as that adapter and may resolve to
its base model or fail. Deploy an unambiguous Ollama model and verify which
artifact/tag the registry backend resolves before using results for promotion.

Backend exceptions are isolated and counted as failed outputs, so a report can
still be written when Ollama is unavailable.

## Metrics

Per scenario the evaluator records:

- structural validity and available-skill legality
- JSON parse failure
- first-skill correctness
- normalized longest-common-subsequence full-plan similarity
- p50/p95 generation latency
- actual/expected skills and backend errors

The documented benchmark score is:

```text
0.30 × valid_plan_rate
+ 0.20 × legal_skill_rate
+ 0.15 × JSON_success_rate
+ 0.15 × first_skill_correct
+ 0.20 × full_plan_score
```

Inspect components as well as the aggregate. A high format score can coexist
with poor task choices, and expected-plan agreement is only as good as the
scenario labels.

## Overfitting signals

Reports include duplicate-output rate, output-diversity rate, exact-match rate,
and a warning. The implemented warning requires at least 10 scenarios and
fires when duplicate output exceeds 80%, or when exact match exceeds 99% while
output diversity is below 20%.

Also investigate manually:

- train/validation loss diverging while training loss falls
- high frozen-suite agreement but weak held-out real episodes
- one plan repeated across different lifecycle states
- sensitivity to wording or sensor omissions
- exact examples duplicated across episode splits
- candidate gains limited to synthetic categories represented in training

## Reports and registry effects

Each run writes timestamped:

```text
data/playmind/planner/evaluation/
├── planner_benchmark_<timestamp>.json
├── planner_benchmark_<timestamp>.csv
└── planner_benchmark_<timestamp>.md
```

JSON contains backend metrics, score formula, and artifact paths; CSV contains
scenario-level rows; Markdown is the comparison summary.

When a backend maps to a registry model ID, evaluation updates its
`eval_metrics`, appends an audit event, and reports promotion gate errors.
Status remains unchanged. Evaluation never promotes.

The GUI searches `data/playmind/eval`, `data/playmind/evaluation`, and
`data/playmind/planner/eval` for `report.json`, while this benchmark writes
timestamped JSON directly under `data/playmind/planner/evaluation`. Therefore
the current **Model Comparison** “latest report” panel may not discover these
artifacts; review the generated Markdown/JSON directly. This is an implemented
CLI with a deferred GUI path-alignment fix.

## Promotion interpretation

Default gates require at least 100 scenarios, valid-plan rate ≥ 0.99,
illegal-skill rate ≤ 0.005, and a benchmark score better than production. The
built-in 18-scenario suite cannot satisfy the count gate. Build a frozen,
held-out suite of at least 100 independently reviewed scenarios before normal
promotion. Synthetic scores are pipeline evidence, not gameplay evidence.
