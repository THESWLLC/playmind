# Model promotion

The SQLite registry is
`data/playmind/planner/registry.sqlite`. Training registers `candidate`;
evaluation updates metrics without changing status; only an explicit operator
action promotes.

## States

| State | Meaning |
|---|---|
| `experimental` | Unqualified development artifact |
| `candidate` | Trained artifact awaiting evaluation and review |
| `production` | Sole selected production record |
| `rejected` | Reviewed and rejected; promotion requires manual override |
| `archived` | Retained but inactive/superseded |

The database enforces at most one production row.

## Default gates

Normal promotion requires:

- `valid_plan_rate >= 0.99`
- `illegal_skill_rate <= 0.005`
- `scenario_count >= 100`
- candidate `benchmark_score` strictly greater than production (margin `0.0`)

Metrics may come from evaluation or training metadata, but promotion decisions
should use held-out evaluation. The built-in 18-case synthetic suite cannot
pass the scenario-count gate.

Inspect gates without changing state:

```bash
python3 - <<'PY'
from playmind.planner_v2.model_registry import ModelRegistry
r = ModelRegistry()
model_id = "<candidate-id>"
print(r.get(model_id))
print(r.promotion_errors(model_id))
PY
```

## Promote or reject

The GUI **Models** tab exposes Promote, Reject, Archive, and Rollback. Its
actions use the fixed reason `GUI operator action`; promotion obeys gates and
does not offer manual override.

For an auditable custom reason:

```bash
python3 - <<'PY'
from playmind.planner_v2.model_registry import ModelRegistry
r = ModelRegistry()
print(r.promote("<candidate-id>", reason="held-out suite and shadow review passed"))
PY
```

Reject a non-production candidate:

```bash
python3 - <<'PY'
from playmind.planner_v2.model_registry import ModelRegistry
r = ModelRegistry()
print(r.reject("<candidate-id>", reason="illegal skill rate exceeded gate"))
PY
```

Production cannot be rejected directly; promote a replacement or roll back
first.

## Manual override audit

Manual override can bypass failed gates and can revive rejected/archived
records. It is intentionally not exposed by the GUI:

```bash
python3 - <<'PY'
from playmind.planner_v2.model_registry import ModelRegistry
r = ModelRegistry()
print(r.promote(
    "<candidate-id>",
    reason="emergency operator override; accepted risks documented in ticket X",
    manual_override=True,
))
PY
```

The promotion audit row has `warning=true` and records
`manual_override`, every gate error, and the previous production ID. An
override is not a substitute for evidence; use it only with a specific reason
and rollback plan.

Review recent audit history:

```bash
python3 - <<'PY'
import json
from playmind.planner_v2.model_registry import ModelRegistry
print(json.dumps(ModelRegistry().audit_log(limit=20), indent=2))
PY
```

Registration, metric updates, status changes, promotion, and rollback are
audited with timestamps, previous/new status, reason, warning, and details.

## Rollback

Promotion archives the previous production model and stores its ID in the
audit details. Roll back to that model:

```bash
python3 - <<'PY'
from playmind.planner_v2.model_registry import ModelRegistry
r = ModelRegistry()
print(r.rollback(reason="candidate regression during shadow review"))
PY
```

Pass `rollback("<model-id>", reason=...)` to select an explicit retained model.
Rollback archives current production and makes the target production without
re-running promotion gates; it is itself audited. It fails when no production
or previous target exists.

Registry state does not by itself rewrite `config/owned_game.json` or prove an
Ollama tag exists. Verify runtime model resolution after every promotion or
rollback, then use `shadow` before enabling any authorized input.
