# Migration (legacy → Learning V2)

Existing owned-game data is **kept**. Migration marks tabular Q as legacy, stamps `schema_version`, and backs up corrupt JSON to `*.bak`.

## What migrates

| Artifact | Action |
|----------|--------|
| `policy.json` | Copy → `policy.legacy.json`, stamp `legacy: true` |
| `experience.jsonl` | Sidecar `experience.meta.json` with `schema_version` |
| `process_memory.json` | Add `schema_version` if missing |
| `ui_memory.json` / `ability_memory.json` / `travel_memory.json` | Same |
| Corrupt JSON | Copied to `.bak` (timestamped if needed) |

Atomic writes use temp file + `os.replace`.

## Command

```bash
PYTHONPATH=. python3 scripts/migrate_legacy_learning.py \
  --data-dir data/playmind/owned

# Machine-readable report
PYTHONPATH=. python3 scripts/migrate_legacy_learning.py --json

# Replace an existing policy.legacy.json
PYTHONPATH=. python3 scripts/migrate_legacy_learning.py --overwrite-legacy-policy
```

## After migration

- Hybrid / scripted modes do **not** invent actions from Q keys by default
- Set `"policy_mode": "legacy_q"` only for experiments
- Optional `"legacy_q_fallback": true` under hybrid

## Diagnostics export

```bash
PYTHONPATH=. python3 scripts/export_diagnostics.py \
  --owned-dir data/playmind/owned \
  --config config/owned_game.json
```

Bundles write to `data/playmind/diagnostics/<timestamp>/` (+ `.zip`). Home paths are redacted.

See also: [QUICKSTART_V2.md](./QUICKSTART_V2.md) · [LEARNING_ARCHITECTURE_V2.md](./LEARNING_ARCHITECTURE_V2.md)
