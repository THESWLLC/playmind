# PlayMind Offline Studio: start here

PlayMind Offline Studio is the file-only path for learning from recordings. It
does not capture a live game, inspect a game process, log gameplay input, or
send keyboard/mouse input. For the official World of Warcraft client, use only
the `retail_wow_offline_only` workflow described here.

## What is available on this branch

| State | Capability |
|---|---|
| Implemented and unit-tested | Video probing/import, FFmpeg frame extraction, provenance gates, project storage, annotation storage, transcript suggestions, offline analysis, dataset export, versioned benchmark builder, readiness checks, correction records, evaluation indexing, and model restrictions |
| Newly implemented | Local Studio web GUI, first-run wizard, import/analysis/annotation/dataset/benchmark/readiness/smoke/evaluation/correction/model/doctor views, `scripts/start_studio.py`, and Windows launcher/setup scripts |
| Implemented and smoke-tested | Planner SFT/DPO command plumbing and synthetic no-weights artifacts |
| Implemented but hardware-dependent and not tested here | Real 3B QLoRA training, Ollama evaluation, Ollama/GGUF export |
| Mocked/stubbed | `change_aware` extraction is denser uniform sampling; transcript suggestions are keyword rules; smoke adapters contain no model weights |
| Incomplete/untested | No video playback waveform/timeline; real-codec matrix and full browser workflow are not tested here; GUI `uniform`/`scene_change` extraction choices do not match backend strategy names and currently error |
| Deferred | Automatic Twitch/YouTube scraping, required cloud transcription, live-client capture/control, and automatic model promotion |

The Studio entry points are `python scripts/start_studio.py` and
`.\start_playmind_studio.bat`. Do not substitute `start_playmind.bat`: that
opens the separate owned-game lab.

## 1. Install

Python 3.10+ and both `ffmpeg` and `ffprobe` are required for real video import.

### Windows

```powershell
winget install Gyan.FFmpeg
git clone https://github.com/THESWLLC/playmind.git
cd playmind
git pull origin main
.\setup_playmind_studio.ps1
ffmpeg -version
ffprobe -version
```

Open a new terminal after `winget` if the tools are not found. For real planner
training, also run:

```powershell
.\setup_windows.ps1 -WithTraining
.\.venv\Scripts\python.exe -m pip install trl bitsandbytes
```

Install a CUDA-compatible PyTorch build for the local NVIDIA driver; the
generic dependency command cannot choose that build safely.

### Debian/Ubuntu/WSL

```bash
sudo apt update && sudo apt install -y ffmpeg python3-venv
git clone https://github.com/THESWLLC/playmind.git
cd playmind
git pull origin main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-playmind.txt
ffmpeg -version
ffprobe -version
```

For real planner training:

```bash
python -m pip install -r requirements-playmind-ml.txt
python -m pip install trl bitsandbytes
```

## 2. Run this command first

Validate startup without opening a port:

```powershell
.\.venv\Scripts\python.exe scripts\start_studio.py --dry-run
```

Then start the Studio:

```powershell
.\start_playmind_studio.bat
```

It opens `http://127.0.0.1:8787/`. Optional Windows launcher parameters are
`-Port 8788`, `-NoBrowser`, and `-Config path\to\studio.json`.

On Linux/macOS/WSL:

```bash
python scripts/start_studio.py --dry-run
python scripts/start_studio.py
```

Verified CLI flags are `--host`, `--port` (default `8787`), `--config`,
`--no-browser`, and `--dry-run`. The server binds to loopback by default; do
not expose it to a network.

## 3. Import one video

Use only a recording you may possess and use for training. Supported
extensions are `.mp4`, `.mkv`, `.mov`, `.avi`, and `.webm`.

In **Projects/Import**, enter the local path and project name, choose `copy` or
`reference`, confirm both rights/local-ML-use boxes, and click **Import**.
The UI is a path form, not a file picker.

Equivalent backend command:

```bash
python - <<'PY'
from playmind.studio.app import StudioApp
from playmind.studio.provenance import ProvenanceRecord

app = StudioApp()
project = app.import_video(
    "recordings/my-session.mp4",
    provenance=ProvenanceRecord(
        source_type="user_owned_recording",
        source_id="my-session-2026-07-29",
        rights_confirmed=True,
        training_use_allowed=True,
        private_use_only=True,
        notes="Recorded by me; local research only.",
    ),
    mode="copy",
    name="My reviewed session",
)
print(project["project_id"])
print(app.extract_frames("overview", interval_seconds=10))
print(app.analyze(do_ocr=False))
PY
```

`copy` stores a copy under `data/playmind/studio/projects/<project-id>/source/`.
Use `mode="reference"` only if the original path will remain stable. See
[offline video import](docs/OFFLINE_VIDEO_IMPORT.md) and
[data provenance](docs/DATA_PROVENANCE_AND_PERMISSION.md).

## 4. Annotate and review

In **Analysis**, use `overview`, extract frames, optionally enable OCR, then
analyze. The GUI's `uniform` and `scene_change` choices currently do not match
implemented backend names; do not use them. In **Annotation timeline**, enter
start/end seconds, choose type/category, and add. Select a row, then accept or
reject it. F7/F8 work only while that review panel is focused and `pynput` is
available. This is a time-range table, not a playable video timeline.

Equivalent backend command:

```bash
python - <<'PY'
from playmind.studio.annotations import AnnotationStore, TimelineSegment

PROJECT_ID = "<project-id>"
store = AnnotationStore(PROJECT_ID)
segment = store.add(TimelineSegment(
    start=12.5,
    end=18.0,
    segment_type="skill",
    category="recover_health",
    label="Player stops and recovers",
    notes="Visible health recovery; independently reviewed.",
))
store.review(segment.segment_id)
print([item.to_dict() for item in store.list()])
PY
```

Suggestions are never training data until a person marks them `reviewed`.
Label uncertain material `unknown` and corrupt/irrelevant material `unusable`.
See [offline annotation](docs/OFFLINE_ANNOTATION.md).

## 5. Export reviewed data

In **Datasets**, click **Export reviewed project**, then inspect counts,
rejected projects, and leakage in the response/artifacts.

```bash
python - <<'PY'
import json
from playmind.studio.app import StudioApp

PROJECT_ID = "<project-id>"
app = StudioApp()
app.select_project(PROJECT_ID)
result = app.export_datasets()
print(json.dumps(result, indent=2))
print(json.dumps(app.readiness(leakage=result["leakage"]), indent=2))
PY
```

The Studio bridge writes SFT splits under
`data/playmind/planner/sft/studio/` and preference splits under
`data/playmind/planner/preferences/studio/`. Check counts, rejected projects,
and leakage before training.

## 6. Build a held-out real benchmark

Do not train on benchmark projects. Build independently reviewed scenarios,
then freeze a new immutable version under
`data/playmind/planner/evaluation/`. Follow
[Real Benchmark Builder](docs/REAL_BENCHMARK_BUILDER.md). The evaluator accepts
JSONL, while the builder stores a versioned JSON envelope; that guide includes
the exact conversion step. The **Benchmark Builder** tab accepts a JSON list,
benchmark ID, and freezes it; it does not automatically derive scenarios from
annotations or require the full recommended category set unless supplied
through the API.

## 7. Run the smoke train

In **Training**, **Start smoke train** launches the exact synthetic smoke path
and labels it **SMOKE / NO REAL WEIGHTS TRAINED**. CLI equivalent:

```bash
python scripts/train_planner_sft.py \
  --preset cpu_tiny_smoke \
  --smoke \
  --max-steps 2 \
  --run-id studio-smoke
```

This ignores real datasets and writes
`models/playmind/runs/studio-smoke/adapter/smoke_artifact.json`. It is
**SMOKE / NO REAL WEIGHTS**, cannot be used as a model, and cannot be promoted.
A successful run proves only CLI, artifact, manifest, and registry plumbing.

## 8. Train on an RTX 4070 Ti

First verify the base model license, exported split counts, no leakage, a
frozen held-out benchmark, CUDA, and free disk. Then:

```bash
python scripts/train_planner_sft.py \
  --base-model /path/to/licensed-3b-hf-model \
  --preset rtx_4070_ti_3b_qlora \
  --train-file data/playmind/planner/sft/studio/train.jsonl \
  --eval-file data/playmind/planner/sft/studio/val.jsonl \
  --run-id studio-3b-sft \
  --no-register
```

The 12 GB preset uses sequence length 1024, microbatch 1, accumulation 16,
gradient checkpointing, LoRA rank 16/alpha 32, and 4-bit NF4 when CUDA
`bitsandbytes` is working. Stop if it warns that QLoRA is unavailable and
falls back to full LoRA. Real training has not been validated on this CI host.
See [first model training](docs/FIRST_MMO_MODEL_TRAINING.md).

## 9. Evaluate

The frozen benchmark envelope must first be converted to JSONL as documented
in the benchmark guide. The evaluator cannot load a LoRA adapter path directly,
so create an Ollama tag for offline evaluation:

```bash
python scripts/export_planner_ollama.py \
  --adapter-path models/playmind/runs/studio-3b-sft/adapter \
  --base-model /path/to/licensed-3b-hf-model \
  --model-name playmind-studio-3b \
  --output models/playmind/Modelfile.studio-3b \
  --create
```

Register that exact tag as a restricted evaluation candidate:

```bash
python - <<'PY'
from playmind.planner_v2.model_registry import ModelRegistry
print(ModelRegistry().register(
    "studio-3b-sft-eval",
    display_name="playmind-studio-3b",
    base_model="/path/to/licensed-3b-hf-model",
    adapter_path="models/playmind/runs/studio-3b-sft/adapter",
    status="candidate",
    live_use_prohibited=True,
    source_game_profile="retail_wow_offline_only",
    allowed_uses=["offline_evaluation"],
    reason="restricted Offline Studio artifact for actuator-free evaluation",
))
PY
```

Then run:

```bash
python scripts/evaluate_planner.py \
  --suite data/playmind/planner/evaluation/studio_real_benchmark_v1.jsonl \
  --candidate-id studio-3b-sft-eval \
  --generic-model llama3.2 \
  --output-dir data/playmind/planner/evaluation
```

The evaluator is offline and actuator-free. It now writes both timestamped
reports and `runs/<run-id>/report.json`, plus `index.json`. It resolves
registry fields in `gguf_path`, `merged_path`, `display_name`, then
`base_model` order; verify `display_name` is the exact Ollama tag.
The Studio **Learning Proof → Evaluate offline** button currently starts the
default built-in synthetic suite with no candidate selection; use the CLI
above for a real suite/candidate claim.

## 10. Export

The Ollama package used in step 9 is already one export. Keep it local unless
the base-model and source-data terms permit redistribution.

For GGUF, merge the adapter into the licensed Hugging Face base model first;
an adapter alone cannot be converted:

```bash
python scripts/export_planner_gguf.py \
  --model-path /path/to/merged-hf-model \
  --output models/playmind/playmind-studio-3b-f16.gguf \
  --llama-cpp-dir third_party/llama.cpp \
  --outtype f16
```

Export success proves artifact creation, not model quality or permission to
redistribute the base model, adapter, recordings, frames, or derived data.

## 11. Interpret learning proof

Evidence becomes stronger in this order:

1. Smoke succeeds: plumbing works; no learning occurred.
2. Real training loss falls while validation loss remains credible: optimizer
   fit occurred; useful behavior is not yet proved.
3. Candidate beats scripted and generic baselines on a held-out real suite:
   offline task evidence exists, subject to label quality.
4. Gains repeat on a second untouched project/source split with low illegal
   skill and parse-failure rates: evidence is less likely to be leakage.
5. A human reviews failures and judges plans useful and safe: practical
   offline evidence exists.

No step proves live gameplay improvement, and Studio artifacts from the
`retail_wow_offline_only` profile are marked for offline use only.

## 12. Personal judgment before accepting a result

- Confirm each source's owner, consent/license, allowed purpose, retention, and
  redistribution terms yourself.
- Watch held-out clips without seeing candidate output; write the expected plan
  and acceptable alternatives first.
- Review candidate, baseline, and generic outputs blind where practical.
- Inspect every illegal skill, JSON failure, death/recovery error, and repeated
  plan—not only the aggregate score.
- Reject a result if train and benchmark share a project, source hash, near-
  duplicate clip, or annotator answer copied into training.
- Decide whether the measured gain matters to a person and generalizes beyond
  the benchmark wording.
- Record dissent and limitations. Do not promote automatically; smoke and
  live-use-prohibited records are hard-blocked from promotion.

Continue with [Studio Quickstart](docs/PLAYMIND_STUDIO_QUICKSTART.md),
[Learning Proof Dashboard](docs/LEARNING_PROOF_DASHBOARD.md), and
[Account Safety Architecture](docs/ACCOUNT_SAFETY_ARCHITECTURE.md).
