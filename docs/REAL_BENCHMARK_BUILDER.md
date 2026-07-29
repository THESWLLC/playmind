# Real benchmark builder

A real benchmark is a held-out, human-reviewed set of planner states and
acceptable plans derived from authorized recordings. It measures offline plan
agreement; it does not execute gameplay or prove live improvement.

## Implementation status

- `StudioScenario`, category coverage checks, provenance/review gates,
  immutable versioning, and SHA-256 content hashes: implemented and unit-tested.
- GUI raw-JSON benchmark form/freezer: implemented; it is not a structured
  per-field scenario editor.
- Automatic scenario creation from annotations: not implemented.
- Evaluator input bridge: not implemented as one command. The builder writes a
  pretty JSON envelope; `evaluate_planner.py --suite` reads JSONL. Use the
  explicit conversion step below.

## Prevent leakage first

Choose benchmark projects before training. Do not export their segments into
SFT or preference data. Keep the whole source family held out:

- same project/source SHA-256;
- clips or re-encodes of the same recording;
- adjacent footage from the same session;
- copied annotations or transcript-derived answers;
- near-duplicate states from the same encounter.

Use a separate reviewer where possible. Write expected plans before looking at
candidate output.

## Scenario schema

Prepare `benchmark_scenarios.jsonl`, one JSON object per line:

```json
{"scenario_id":"session42-loading-001","category":"loading","planner_state":{"available_skills":["wait","clear_modal"],"sensors":{"is_loading":{"value":true,"known":true}}},"expected_plan":{"skills":["wait"]},"acceptable_alternative_plans":[{"skills":["clear_modal","wait"]}],"project_id":"session42","source_id":"sha256-or-source-id","reviewed":true,"provenance_eligible":true,"notes":"Reviewed by A and B; held out before training."}
```

Required fields are `scenario_id`, `category`, `planner_state`, and
`expected_plan`. For a `frozen_real` suite, `reviewed` and
`provenance_eligible` must both be true. IDs must be unique.

`expected_plan` and alternatives are normalized planner plans. At minimum they
need a non-empty `skills` list. Include `available_skills` in planner state so
legality is evaluated against the intended situation.

## Category coverage

The implemented recommended category set is:

```text
combat, recovery, multi_enemy, target_loss, death, ghost, loading, inventory,
quest, modal, nav, stuck, skill_fail, conflicting_sensors, unknown_sensors,
long_horizon
```

The promotion gate requires at least 100 scenarios. That count is a minimum,
not a guarantee of representative coverage. Include ordinary and difficult
cases, sensor ambiguity, acceptable alternatives, and failure recovery.

## Validate and freeze

The Studio **Benchmark Builder** tab accepts a benchmark ID and JSON list of
scenarios, then freezes it. The UI does not request the recommended required
category set, so use the backend command below when enforcing full coverage.

```bash
python - <<'PY'
import json
from pathlib import Path
from playmind.studio.benchmark_builder import (
    BenchmarkBuilder,
    REQUIRED_BENCHMARK_CATEGORIES,
)

source = Path("benchmark_scenarios.jsonl")
scenarios = [
    json.loads(line)
    for line in source.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
result = BenchmarkBuilder(
    "data/playmind/planner/evaluation"
).freeze(
    scenarios,
    benchmark_id="studio_real_benchmark",
    tier="frozen_real",
    required_categories=REQUIRED_BENCHMARK_CATEGORIES,
)
print(json.dumps(result, indent=2))
PY
```

This creates `studio_real_benchmark_v1.json`; subsequent freezes create `v2`,
`v3`, and so on. Existing versions cannot be overwritten through the builder.
The envelope records category names, count, creation time, tier, and a hash of
canonical scenario rows.

Never edit a frozen file. Correct an error in source records, rerun review, and
freeze a new version with notes describing the change.

## Convert the frozen envelope for the evaluator

`evaluate_planner.py --suite` currently requires JSONL:

```bash
python - <<'PY'
import json
from pathlib import Path

source = Path("data/playmind/planner/evaluation/studio_real_benchmark_v1.json")
destination = source.with_suffix(".jsonl")
payload = json.loads(source.read_text(encoding="utf-8"))
with destination.open("x", encoding="utf-8") as handle:
    for scenario in payload["scenarios"]:
        handle.write(json.dumps(scenario, sort_keys=True) + "\n")
print(destination)
PY
```

Using open mode `x` prevents accidental overwrite. If the JSONL needs to
change, delete it only after confirming the matching frozen JSON version and
regenerate it byte-for-byte.

## Evaluate

```bash
python scripts/evaluate_planner.py \
  --suite data/playmind/planner/evaluation/studio_real_benchmark_v1.jsonl \
  --candidate-id <registry-candidate-id> \
  --generic-model llama3.2 \
  --output-dir data/playmind/planner/evaluation
```

Verified CLI options are `--suite`, `--output-dir`, `--registry-path`,
`--generic-model`, `--candidate-id`, `--ollama-host`, and `--timeout`.

The command writes timestamped JSON/CSV/Markdown, a canonical
`runs/<run-id>/report.json`, and `index.json`. Evaluation updates candidate
metrics but never promotes.

## Human acceptance

Inspect:

- scenario count and category balance;
- valid-plan, illegal-skill, JSON failure, first-skill, and full-plan metrics;
- every candidate error and disagreement;
- baseline/generic failures that might make a relative gain misleading;
- duplicate output, exact match, and diversity warnings;
- cases where an unlisted alternative is actually reasonable;
- robustness to omitted/unknown/conflicting sensors;
- a second untouched source set.

If review changes the expected answer after seeing model output, record the
reason and freeze a new suite. Do not tune training repeatedly against the
held-out suite and continue calling it held out.
