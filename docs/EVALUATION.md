# Evaluation & offline replay

Evaluate skill policies on saved demonstrations **without** sending keys or requiring the game.

## Replay a session

```bash
PYTHONPATH=. python3 - <<'PY'
from pathlib import Path
from playmind.demonstrations import list_sessions
from playmind.policies.scripted import ScriptedPolicy
from playmind.replay_env import ReplayEnv

sessions = list_sessions()
assert sessions, "record demos first — see docs/DEMONSTRATION_RECORDING.md"
env = ReplayEnv.from_session(sessions[0], policy=ScriptedPolicy())
obs = env.reset()
n = 0
while not env.done:
    step = env.step()
    n += 1
    if step is None:
        break
print("replayed_steps=", n, "last=", env.last_decision)
PY
```

## Evaluate a checkpoint (metadata / BC stub)

```bash
PYTHONPATH=. python3 - <<'PY'
from playmind.models.policy_v2 import SkillPolicyV2
from playmind.demonstrations import list_sessions
from playmind.replay_env import ReplayEnv

ckpt = "models/checkpoints/skill_policy_v2.json"
policy = SkillPolicyV2.load(ckpt)
sessions = list_sessions()
env = ReplayEnv.from_session(sessions[0], policy=policy) if sessions else None
print("trained=", policy.trained, "skills=", policy.skill_names)
if env is not None:
    env.reset()
    matches = 0
    total = 0
    while not env.done:
        step = env.step()
        if step is None:
            break
        total += 1
        label = step.sample.get("skill")
        if label and step.decision.skill == label:
            matches += 1
    print("skill_match_rate=", (matches / total) if total else None, "n=", total)
PY
```

## Config

```json
"learning_v2": {
  "evaluation": {
    "enabled": false,
    "report_dir": "data/playmind/eval",
    "max_replay_samples": 5000
  }
}
```

Metrics targets (kill rate, deaths/hour, skill timeout rate, etc.) are sketched in [EVALUATION_PLAN.md](./EVALUATION_PLAN.md) and [LEARNING_ARCHITECTURE_V2.md](./LEARNING_ARCHITECTURE_V2.md).
