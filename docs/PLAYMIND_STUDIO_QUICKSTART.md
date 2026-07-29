# PlayMind Offline Studio quickstart

Offline Studio turns authorized recording files into reviewed planner/vision
datasets and held-out evaluations. It has no live capture or generated-input
surface.

For the complete install-to-export path, start with
[START_HERE.md](../START_HERE.md).

## Availability

The `playmind.studio` backend, local browser GUI, and separate launchers are
implemented. Validate without opening a port, then launch:

```bash
python scripts/start_studio.py --dry-run
python scripts/start_studio.py
```

```powershell
.\setup_playmind_studio.ps1
.\start_playmind_studio.bat
```

The default URL is `http://127.0.0.1:8787/`. Verified script flags are
`--host`, `--port`, `--config`, `--no-browser`, and `--dry-run`. Windows
launcher parameters are `-Port`, `-Config`, and `-NoBrowser`.
`start_playmind.bat` opens the owned-game lab and is not a substitute.

## Prerequisites

- Python 3.10+
- `ffmpeg` and `ffprobe` on `PATH`
- `requirements.txt` and `requirements-playmind.txt`
- a local `.mp4`, `.mkv`, `.mov`, `.avi`, or `.webm` recording
- documented rights/permission for the intended training use

Verify:

```bash
python - <<'PY'
from playmind.studio.media_probe import media_tools_status
from playmind.studio.safety import studio_may_not_send_input
print(media_tools_status())
assert studio_may_not_send_input()
PY
```

## Five-step workflow

### 1. Import and extract

In **Projects/Import**, enter a local path, choose copy/reference, check the
rights and local-ML-use confirmations, and import. In **Analysis**, select
`overview` and extract. The current GUI is path-based, not a file picker. Its
`uniform` and `scene_change` options do not match backend strategy names and
currently error.

Equivalent backend:

```bash
python - <<'PY'
from playmind.studio.app import StudioApp
from playmind.studio.provenance import ProvenanceRecord

app = StudioApp()
project = app.import_video(
    "recordings/session.mp4",
    provenance=ProvenanceRecord(
        "user_owned_recording",
        source_id="session-001",
        rights_confirmed=True,
        training_use_allowed=True,
        private_use_only=True,
    ),
    mode="copy",
)
print("PROJECT_ID=" + project["project_id"])
print(app.extract_frames("overview", interval_seconds=10))
PY
```

The default project root is `data/playmind/studio/projects/`.

### 2. Analyze and annotate

Click **Analyze offline frames**. In **Annotation timeline**, add a time range,
select it, then accept/reject. F7/F8 are optional and work only while the review
panel is focused. There is no playable video timeline yet.

Equivalent backend:

```bash
python - <<'PY'
from playmind.studio.app import StudioApp
from playmind.studio.annotations import TimelineSegment

PROJECT_ID = "<project-id>"
app = StudioApp()
app.select_project(PROJECT_ID)
app.analyze(do_ocr=False)
segment = app.add_annotation(TimelineSegment(
    10.0, 16.0, "recover_health",
    review_status="suggested",
    notes="Requires human confirmation.",
))
app.annotations().review(segment.segment_id)
PY
```

Automated detections and transcript suggestions remain `suggested`; a person
must review them. See [Offline Annotation](./OFFLINE_ANNOTATION.md).

### 3. Export and check readiness

Use **Datasets → Export reviewed project**, then **Training Readiness →
Refresh readiness**. Equivalent backend:

```bash
python - <<'PY'
import json
from playmind.studio.app import StudioApp

app = StudioApp()
app.select_project("<project-id>")
exported = app.export_datasets()
print(json.dumps(exported, indent=2))
print(json.dumps(app.readiness(leakage=exported["leakage"]), indent=2))
PY
```

Do not proceed if `rejected_projects` or `leakage` is non-empty. “Ready for
experimental” is not the same as “Ready for normal”; the normal defaults are
1,000 reviewed examples, 100 preferences, at least one frozen real benchmark,
a GPU, license confirmation, and no blockers.

### 4. Smoke, then real train

**Training → Start smoke train** launches only the smoke path. CLI equivalent:

```bash
python scripts/train_planner_sft.py \
  --preset cpu_tiny_smoke --smoke --max-steps 2 --run-id studio-smoke
```

The result is **SMOKE / NO REAL WEIGHTS**.

```bash
python scripts/train_planner_sft.py \
  --base-model /path/to/licensed-3b-hf-model \
  --preset rtx_4070_ti_3b_qlora \
  --train-file data/playmind/planner/sft/studio/train.jsonl \
  --eval-file data/playmind/planner/sft/studio/val.jsonl \
  --run-id studio-3b-sft \
  --no-register
```

Real 4070 Ti QLoRA is implemented but was not exercised on the CI host.

### 5. Evaluate and judge

```bash
python scripts/evaluate_planner.py \
  --suite data/playmind/planner/evaluation/studio_real_benchmark_v1.jsonl \
  --candidate-id studio-3b-sft-eval \
  --generic-model llama3.2
```

Before this command, create a held-out suite and expose the candidate through
an Ollama-resolvable artifact/tag. Register that exact tag as a separate
candidate with `live_use_prohibited=True`,
`source_game_profile="retail_wow_offline_only"`, and
`allowed_uses=["offline_evaluation"]`; the generic trainer does not propagate
those fields automatically. The Studio **Evaluate offline** button runs the
default synthetic suite without a candidate selector, so use this CLI for a
real claim. Read
[Real Benchmark Builder](./REAL_BENCHMARK_BUILDER.md),
[First MMO Model Training](./FIRST_MMO_MODEL_TRAINING.md), and
[Learning Proof Dashboard](./LEARNING_PROOF_DASHBOARD.md).

## What success does not mean

- Import success proves media/provenance persistence, not permission.
- Suggested labels are not reviewed truth.
- Smoke success proves plumbing only.
- Falling training loss does not prove generalization.
- A synthetic benchmark does not prove real-recording quality.
- Offline plan agreement does not prove live gameplay improvement.
- A registry `candidate` is not automatically deployable or promotable.
