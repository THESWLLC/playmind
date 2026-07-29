# Correction-driven learning

Correction-driven learning records a candidate plan and a human's corrected
plan for the same offline planner state. One reviewed correction can produce:

- an SFT row whose target is the corrected plan; and
- a preference row where corrected is `chosen` and candidate is `rejected`.

The correction store, eligibility checks, and dataset bridge are implemented
and unit-tested. The Studio **Model Review/Corrections** tab can add JSON
candidate/corrected plans and accept/reject them. Automatic candidate
generation and clip-synchronized editing are not implemented.

## When a correction is eligible

All conditions must hold:

- `review_status == "reviewed"`;
- candidate and corrected plans both have skills;
- corrected plan differs from candidate plan; and
- project provenance is training-eligible.

A suggested or rejected correction remains in the project but is excluded.

## Record a correction

Use an offline state tied to an imported project's timestamp:

```bash
python - <<'PY'
from playmind.studio.corrections import CorrectionStore, PlanCorrection

PROJECT_ID = "<project-id>"
store = CorrectionStore(PROJECT_ID)
item = store.add(PlanCorrection(
    project_id=PROJECT_ID,
    timestamp=42.5,
    planner_state={
        "available_skills": ["engage_target", "recover_health", "wait"],
        "sensors": {
            "player_hp": {"value": 0.12, "known": True, "confidence": 0.9}
        },
    },
    candidate_plan={"skills": ["engage_target"]},
    corrected_plan={"skills": ["recover_health"]},
    notes="Candidate ignored critically low health; reviewed against clip.",
))
store.review(item.correction_id, accepted=True)
print(item.correction_id)
PY
```

Plans are normalized by the same planner data schema used for training.
Corrections are stored in
`data/playmind/studio/projects/<project-id>/corrections.json`.

## Human correction protocol

1. Freeze the exact planner state and source timestamp.
2. Save the candidate output before editing it.
3. Watch sufficient clip context and inspect known/unknown sensor confidence.
4. Write the smallest defensible corrected plan. Do not add unavailable skills.
5. Explain the behavioral error, not merely “wrong.”
6. Add acceptable alternatives to a benchmark scenario rather than forcing one
   arbitrary answer when several plans are valid.
7. Have another reviewer inspect high-impact death/recovery, modal, target-loss,
   and long-horizon corrections.
8. Reject ambiguous examples rather than laundering guesses as preferences.

Candidate/corrected pairs from the same prompt should stay in the same split.

## Export

```bash
python - <<'PY'
import json
from playmind.studio.dataset_bridge import export_reviewed_projects
result = export_reviewed_projects(["<project-id>"])
print(json.dumps(result, indent=2))
PY
```

Outputs are under:

```text
data/playmind/planner/sft/studio/
data/playmind/planner/preferences/studio/
```

Check that `counts.preferences` increased and that `rejected_projects` and
`leakage` are empty. Do not use `--include-ineligible` to bypass review.

## Train

Use corrections in SFT:

```bash
python scripts/train_planner_sft.py \
  --base-model /path/to/licensed-3b-hf-model \
  --preset rtx_4070_ti_3b_qlora \
  --train-file data/playmind/planner/sft/studio/train.jsonl \
  --eval-file data/playmind/planner/sft/studio/val.jsonl \
  --no-register
```

Or train preferences from the base model:

```bash
python scripts/train_planner_dpo.py \
  --base-model /path/to/licensed-3b-hf-model \
  --preset rtx_4070_ti_3b_qlora \
  --train-file data/playmind/planner/preferences/studio/train.jsonl \
  --eval-file data/playmind/planner/preferences/studio/val.jsonl \
  --beta 0.1 \
  --no-register
```

Current DPO does not chain from an SFT adapter. That limitation prevents a
complete iterative SFT-adapter-then-DPO loop on this branch.
For protected-profile data, register only a later offline-evaluation package
with `live_use_prohibited=True`; generic training registration does not inherit
Studio provenance automatically.

## Prove that corrections helped

Maintain three disjoint sets:

1. training corrections;
2. validation corrections used for stopping/model selection; and
3. frozen real benchmark scenarios never used for tuning.

Compare pre-correction and post-correction candidates on the same frozen suite,
including category-level errors. A lower training loss or memorized corrected
example is not proof. Look for gains on new source projects, no illegal-skill
regression, no output collapse, and human-reviewed improvement on the error
class targeted by corrections.

Never collect corrections by running or controlling the official retail WoW
client. Studio corrections are post-recording, offline records only.
