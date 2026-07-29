# MMO LLM troubleshooting

Run diagnostics first:

```bash
python3 scripts/doctor.py
python3 scripts/start_all.py --dry-run
```

On Windows use `.\.venv\Scripts\python.exe` instead of `python3`.
Top-level Doctor `ok=true` does not mean every optional dependency/service is
available; inspect each section.

## Ollama is down or a model is missing

Symptoms: `ollama.reachable=false`, connection refused, timeout, or backend
errors in evaluation.

```bash
ollama serve
ollama list
ollama pull llama3.2
```

Confirm `planner_v2.host` (default `http://127.0.0.1:11434`) and configured
model name. Planner runtime catches transport/validation errors and uses a
scripted fallback; evaluation records failed backend rows. This keeps the
system safe but is not evidence that the LLM candidate worked.

## `pynput` missing or physical capture unavailable

```bash
python -m pip install -r requirements-playmind.txt
python scripts/doctor.py
```

Restart the GUI after installation. On Linux/WSL, global desktop listeners may
lack a display/session; record on the Windows game host. On Windows, an
elevated game may require the recorder at the same integrity level. If no
physical events appear, do not export the session as human data.

## CUDA out of memory

- Use `rtx_4070_ti_3b_qlora`, not experimental 7B.
- Confirm `bitsandbytes` is installed and the manifest says `4bit-nf4`.
- Keep microbatch 1 and close other GPU applications.
- Reduce preset sequence length in code; there is no CLI length override.
- Remove stale trainer processes and check `nvidia-smi`.
- Resume from a saved checkpoint only after correcting memory settings.

If the trainer warns that QLoRA is unavailable and falls back to full LoRA,
stop: the fallback consumes substantially more memory. See
[PLANNER_TRAINING.md](./PLANNER_TRAINING.md).

## Plan validation fails

The validator requires schema version 1, a non-empty goal, complete skill-step
objects, 1–120 second integer timeouts, valid replan events, confidence in
`[0,1]`, and known/available skills. It also rejects death-recovery plus combat
in one plan.

Inspect **Planner** validation/error fields or benchmark CSV output. Malformed
or low-confidence runtime output falls back to a scripted plan. Do not relax
the validator to accommodate model output; correct the prompt/data/model.

## Focus loss or unexpected input

Click **Emergency stop**, then **Stop**. Leave `enable_keyboard=false` and live
keyboard unchecked until the cause is known. Match the exact game window title
and avoid Alt-Tab during capture. When capture reports “unfocused,” the owned
loop sets a soft emergency stop and forces `wait`.

Physical demonstration focus metadata has a separate limitation: the current
GUI does not wire a game-focus provider into `PhysicalInputCapture`, so it may
label Alt-Tabbed events focused. Stop recording before switching applications
and mark contaminated sessions `Bad`.

## Dataset export is empty

Check source rows for:

- `input_source: "playmind_generated"`
- `training_eligible: false`
- no `inferred_skill`, `skill`, or plan
- no `meta.jsonl` under a session

Generated and ineligible rows are intentionally excluded. Do not solve this by
using `--include-ineligible` for training; record and label clean human data.

## Training exits immediately

Real SFT requires an existing train JSONL and explicit `--base-model` for GPU
presets. Install `torch transformers datasets peft trl accelerate` and
`bitsandbytes` for CUDA QLoRA. `requirements-playmind-ml.txt` currently omits
the last two trainer-specific packages (`trl`, `bitsandbytes`).

Run `--smoke` to isolate CLI/registry plumbing. Remember that smoke ignores the
real dataset and writes a synthetic placeholder.

## Evaluation cannot pass promotion gates

The built-in suite has 18 scenarios; the default gate requires 100. Supply a
larger frozen held-out `--suite`. Also verify the candidate registry backend is
the actual deployed model: the evaluator does not load `adapter_path`
directly.

## Studio does not start

Run the startup-only validation:

```bash
python scripts/start_studio.py --dry-run
```

Verified launcher flags are `--host`, `--port`, `--config`, `--no-browser`,
and `--dry-run`; the default is `127.0.0.1:8787`. On Windows:

```powershell
.\setup_playmind_studio.ps1
.\start_playmind_studio.bat -Port 8788 -NoBrowser
```

Use `-Config path\to\studio.json` for a custom Studio config. If port 8787 is
busy, select another port. Do not run `start_playmind.bat` as a substitute: it
opens the separate owned-game lab and is not permitted for retail WoW.

## FFmpeg or ffprobe is missing

Symptoms include `MediaToolUnavailableError`, “ffmpeg was not found,” or
“ffprobe was not found.”

```powershell
winget install Gyan.FFmpeg
ffmpeg -version
ffprobe -version
```

```bash
# Debian/Ubuntu/WSL
sudo apt update && sudo apt install -y ffmpeg
ffmpeg -version
ffprobe -version
```

Open a new terminal after installation so `PATH` refreshes. Both executables
are required: `ffprobe` inspects/hashes metadata before project creation and
`ffmpeg` extracts frames.

## Video import or extraction fails

- Supported extensions are `.mp4`, `.mkv`, `.mov`, `.avi`, and `.webm`.
- Renaming an unsupported/corrupt file does not convert it; inspect or remux it
  with FFmpeg.
- “media has no video stream” means ffprobe found no video stream.
- Reference-mode projects fail after the original is moved/deleted; restore it
  or import again.
- `change_aware` is currently a denser uniform-sampling stub, not a visual
  change detector.
- Keyframe manifests currently have `timestamp: null`; use overview/manual
  extraction when exact timestamps are needed.

Run the media status check:

```bash
python - <<'PY'
from playmind.studio.media_probe import media_tools_status
print(media_tools_status())
PY
```

## No latest planner evaluation appears

The evaluator now writes timestamped JSON/CSV/Markdown, a canonical
`data/playmind/planner/evaluation/runs/<run-id>/report.json`, and a normalized
`index.json`. Rebuild discovery with:

```bash
python - <<'PY'
import json
from playmind.studio.eval_index import write_index
print(json.dumps(write_index(), indent=2))
PY
```

Both the Studio and owned-game dashboards now use the canonical discovery
adapter. If a dashboard still shows no report, open `index.json` or generated
Markdown directly and restart/refresh the GUI. Malformed JSON is skipped.

The real benchmark builder writes a JSON envelope, while
`evaluate_planner.py --suite` currently expects one scenario object per JSONL
line. Follow the conversion step in
[REAL_BENCHMARK_BUILDER](./REAL_BENCHMARK_BUILDER.md); passing the pretty
`*_v1.json` envelope directly causes a JSON parsing/schema failure.

## A smoke run looks like a trained model

Planner SFT/DPO `--smoke` and `--dry-run` ignore real training files and write
synthetic placeholders. Confirm:

```text
training_manifest.json: "smoke": true
adapter/smoke_artifact.json
registry: smoke=true, allowed_uses=["smoke_validation"]
```

Label the result **SMOKE / NO REAL WEIGHTS**. It proves command, artifact,
manifest, and registry plumbing only. It cannot be used or promoted; registry
promotion rejects smoke artifacts even with manual override. Use a different
run ID for real training.

## Port 8777 is busy

Windows:

```powershell
.\start_playmind.bat -Port 8778
```

Other platforms:

```bash
python3 scripts/start_all.py --port 8778
```

## Safe shutdown did not complete

Stop recording, click **Stop**, and wait for `idle` before closing the GUI.
Emergency stop clears planner execution. After a forced close, ensure no
PlayMind Python process remains before enabling authorized input again.
