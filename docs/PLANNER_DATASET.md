# Planner datasets

## Layout

Planner artifacts live under `data/playmind/planner/`:

```text
planner/
├── sft/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── preferences/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── evaluation/
│   ├── eval_suite.jsonl
│   └── planner_benchmark_<timestamp>.{json,csv,md}
├── manifests/
│   ├── sft.manifest.json
│   ├── preferences.manifest.json
│   └── evaluation.manifest.json
└── registry.sqlite
```

Training runs are separate under `models/playmind/runs/<run-id>/`.

## SFT schema

Export demonstrations with:

```bash
python3 scripts/export_planner_sft.py \
  --input data/playmind/demonstrations \
  --output-dir data/playmind/planner/sft \
  --manifest-dir data/playmind/planner/manifests \
  --seed 0
```

Each schema-v1 JSONL object contains:

```json
{
  "schema_version": 1,
  "example_id": "stable sample id",
  "episode_id": "episode id",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "{\"planner_state\":{...}}"},
    {"role": "assistant", "content": "{\"skills\":[...],\"rationale\":\"...\"}"}
  ],
  "split": "train",
  "eligible": true,
  "metadata": {
    "sample_id": "...",
    "session_id": "...",
    "input_source": "human",
    "skill": "explore"
  },
  "planner_state": {},
  "plan": {},
  "input_source": "human"
}
```

Planner state preserves sensor `{value, known, confidence}`, unknown sensor
names, lifecycle, warnings, recent action/outcome, objective text, and known
abilities. The trainer consumes `messages`; retained state/plan fields support
audit and manifests.

Normal SFT eligibility requires a non-empty plan, an eligible segment/sample,
and a source other than `playmind_generated`. The exporter does not require
`input_source=human`; legacy `unknown` rows can pass, so audit source coverage.

## Preference schema

Prepare a JSONL source with planner state (or an observation), distinct chosen
and rejected plans, an episode ID, source, and optional outcomes:

```json
{
  "episode_id": "ep-42",
  "input_source": "human",
  "planner_state": {"goal": "survive", "available_skills": ["recover_health", "engage_target"]},
  "chosen": {"skills": ["recover_health"], "rationale": "low health"},
  "rejected": {"skills": ["engage_target"], "rationale": "unsafe"},
  "outcomes": {"chosen": "success", "rejected": "unsafe"},
  "training_eligible": true
}
```

Export it with:

```bash
python3 scripts/export_planner_preferences.py \
  --input data/playmind/planner/preference_source.jsonl
```

Exported rows contain `example_id`, `episode_id`, `planner_state`, `chosen`,
`rejected`, `outcomes`, `split`, `eligible`, `input_source`, and
`schema_version`. Empty/equal pairs, generated input, and ineligible rows are
excluded by default.

## Evaluation schema

An evaluation scenario contains:

```json
{
  "schema_version": 1,
  "scenario_id": "planner-recovery-v1",
  "category": "recovery",
  "planner_state": {},
  "expected_plan": {"skills": ["recover_health"], "rationale": ""},
  "invalid_plan": {"skills": ["engage_target"]}
}
```

The code includes 18 frozen, synthetic lifecycle/uncertainty scenarios. Export
them when a versioned file is needed:

```bash
python3 - <<'PY'
from playmind.planner_data.export_eval_suite import export_eval_suite
print(export_eval_suite())
PY
```

Without `--suite`, `evaluate_planner.py` reads the same in-code frozen suite.
These mocked cases test plan formatting and expected-skill agreement, not
gameplay competence.

## Manifests and splits

SFT and preference rows use deterministic SHA-256 episode bucketing with seed
`0` by default and ratios `70/15/15`. Every row sharing an `episode_id` stays in
one split. The assertion catches cross-split episode leakage, but the fallback
ID `"unknown"` places all missing-ID rows together; fix missing IDs rather than
accepting that bucket.

Each manifest records schema/type, creation time, total and split counts,
SHA-256 file hashes, eligibility, and coverage for skills, categories,
lifecycle states, unknown sensors, and input sources. Exporters add source
counts, split seed, and ratios. Commit or archive manifests with an experiment
so its exact input can be reconstructed.

`--include-ineligible` is an audit/debug option. It writes excluded rows with
`eligible=false`; trainers do not independently filter that field, so never
point normal training at such an export.
