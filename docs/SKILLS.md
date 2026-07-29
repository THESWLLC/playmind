# Skills

Hierarchical skills emit **masked** low-level actions (`key:`, `hold:`, `wait`, …). The high-level policy selects a skill; `SkillRuntime` steps it until success, failure, or timeout.

## Registry

```bash
PYTHONPATH=. python3 - <<'PY'
from playmind.skills import list_skills
print("\n".join(list_skills()))
PY
```

Core skills include:

| Skill | Role |
|-------|------|
| `acquire_target` / `validate_target` | Tab / confirm target |
| `approach_target` / `engage_target` | Close distance / open combat |
| `basic_combat_rotation` | Attack loop |
| `loot_target` / `disengage` | Aftermath / back off |
| `recover_health` / `unstuck` / `explore` | Survival / motion |
| `clear_modal` / `interact` / `wait` | UI / idle |
| `death_recovery` / `ghost_runback` | Death pipeline |

## Timeouts & retries

Defaults live in `playmind/config_v2.py` and may be overridden:

```json
"learning_v2": {
  "skill_timeouts": { "basic_combat_rotation": 20.0, "death_recovery": 25.0 },
  "skill_retry_limits": { "acquire_target": 4, "unstuck": 4 }
}
```

## Policy modes

- **scripted** — deterministic skill order / emergencies only
- **hybrid** — emergencies + BC when confident, else scripted (legacy Q opt-in last)
- **legacy_q** — experimental raw tabular Q bridge

```bash
# Scripted smoke run
PYTHONPATH=. python3 scripts/run_owned_loop.py --config config/owned_game.json --max-ticks 30
```

Ensure `learning_v2.policy_mode` is set appropriately in the config (see [QUICKSTART_V2.md](./QUICKSTART_V2.md)).
