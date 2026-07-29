# MMO LLM planner quickstart

PlayMind is for games you own or are authorized to automate. Do not use it with
official MMO clients whose terms prohibit automation. Start in `observe` or
`shadow`; useful learned play requires real demonstrations and held-out
evaluation.

Status terms used here:

- **Implemented** — code exists.
- **Tested** — covered by the repository's automated tests or a command noted
  below.
- **Mocked** — synthetic data/backend plumbing only; not evidence of gameplay.
- **Deferred** — requires local data, hardware, or external tooling.

## 1. First-time Windows setup

From PowerShell in the repository:

```powershell
Copy-Item config\owned_game.example.json config\owned_game.json
notepad config\owned_game.json
.\setup_windows.ps1
.\start_playmind.bat
```

Set the owned game's real window title and ROIs in `config/owned_game.json`.
Leave these safe defaults unchanged initially:

```json
{
  "i_own_this_game": false,
  "enable_keyboard": false,
  "mode": "shadow"
}
```

`start_playmind.bat` uses the repo-relative `.venv`, runs System Doctor, opens
the GUI at <http://127.0.0.1:8777/>, and starts safely even when Ollama is down.
See [WINDOWS_SETUP.md](./WINDOWS_SETUP.md) for details. On Linux/macOS, use
`python3 scripts/doctor.py` and `python3 scripts/start_all.py`.

## 2. Observe before recording

1. Open the GUI's **Dashboard** tab.
2. Select `observe` to capture/status-check without calling Planner V2, or
   `shadow` to generate and validate plans without sending input.
3. Keep **live keyboard** unchecked, enter a directive, and click **Start**.
4. Inspect **Live Perception**, **Planner**, **Alerts**, and **System Doctor**.

The dry-run actuator remains active while live keyboard is unchecked.
`observe`, `shadow`, and `replay` never authorize Planner V2 input regardless of
ownership flags.

## 3. Record a real demonstration

Keep the owned loop running so samples receive synchronized status snapshots.
Play the game yourself using physical keyboard and mouse input:

1. Open **Demonstrations**, provide a session and goal, then click
   **Start recording**. `F9` is the global start/stop toggle when `pynput`
   works.
2. Focus only the game and demonstrate one coherent task. Avoid chat, menus,
   desktop shortcuts, and PlayMind-generated live input.
3. Click **Success**, **Failure**, or **Bad**, then **Stop**. Mark contamination
   `Bad`.

Rows are written to
`data/playmind/demonstrations/<session>/meta.jsonl`; session state is in
`session.json`. Physical and generated events are source-labelled separately.
The GUI currently does not supply a real game-focus provider to the physical
listener, so stop recording before Alt-Tabbing; do not rely on `focused=true`
as proof. See [HUMAN_DEMONSTRATIONS.md](./HUMAN_DEMONSTRATIONS.md).

## 4. Review and export

Use **Replay**, **Demonstrations**, and **Dataset** in the GUI, then inspect the
source rows before exporting:

```powershell
Get-Content data\playmind\demonstrations\<session>\session.json
Get-Content data\playmind\demonstrations\<session>\meta.jsonl
.\.venv\Scripts\python.exe scripts\export_planner_sft.py
```

The exporter excludes `playmind_generated`, ineligible, and empty-plan rows by
default and writes episode-safe splits:

```text
data/playmind/planner/sft/{train,val,test}.jsonl
data/playmind/planner/manifests/sft.manifest.json
```

Do not use `--include-ineligible` for normal training. Details:
[PLANNER_DATASET.md](./PLANNER_DATASET.md).

## 5. Tiny smoke train

```powershell
.\.venv\Scripts\python.exe scripts\train_planner_sft.py --smoke --max-steps 2
```

This tested path needs no ML packages, writes a synthetic placeholder artifact
under `models/playmind/runs/`, and registers it as `candidate`. It does **not**
load the exported dataset, train model weights, or prove learning.

## 6. Real RTX 4070 Ti QLoRA

WSL2 is recommended. Install the ML stack plus the trainer's additional runtime
dependencies:

```bash
source .venv-wsl/bin/activate
python -m pip install -r requirements-playmind-ml.txt
python -m pip install trl bitsandbytes
python scripts/train_planner_sft.py \
  --base-model /path/to/licensed-3b-hf-model \
  --preset rtx_4070_ti_3b_qlora
```

Review the base model's license. The preset is implemented for a 12 GB 4070 Ti:
sequence length 1024, microbatch 1, gradient accumulation 16, gradient
checkpointing, LoRA rank 16/alpha 32, and NF4 4-bit loading. Real GPU training
has not been run by repository CI. See [PLANNER_TRAINING.md](./PLANNER_TRAINING.md)
and [WSL2_TRAINING.md](./WSL2_TRAINING.md).

## 7. Evaluate

Start Ollama and ensure the generic and exported candidate tags exist, then:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_planner.py `
  --generic-model llama3.2 `
  --candidate-id <registry-model-id>
```

Reports are written under `data/playmind/planner/evaluation/`. The command
compares scripted, generic Ollama, and any resolvable registry production or
candidate backend. A newly registered adapter is not directly loaded by this
evaluator; export/deploy it as an Ollama-resolvable model first and confirm the
backend identity. The built-in frozen suite has 18 mocked scenarios, below the
default 100-scenario promotion gate. See
[PLANNER_EVALUATION.md](./PLANNER_EVALUATION.md).

## 8. Shadow, promote, reject, and roll back

Run the candidate in `shadow`, leave live keyboard off, and review plans,
validation, alerts, and comparative reports. Training and evaluation never
auto-promote.

In **Models**, use **Promote** only after all gates pass; use **Reject** with a
reason for a failed candidate. The GUI does not expose a gate override. Manual
overrides are available only through the registry API and are warning-audited.
Use **Rollback** to restore the previous production model. Full gate and audit
details: [MODEL_PROMOTION.md](./MODEL_PROMOTION.md).

## 9. Export to Ollama

For a compatible adapter and a local base-model directory:

```powershell
.\.venv\Scripts\python.exe scripts\export_planner_ollama.py `
  --adapter-path models\playmind\runs\<run-id>\adapter `
  --base-model C:\models\<licensed-base-model> `
  --model-name playmind-planner-candidate `
  --output models\playmind\Modelfile.candidate
ollama create playmind-planner-candidate -f models\playmind\Modelfile.candidate
```

For an already merged local model, replace the adapter/base options with
`--merged-path <directory>`. Add `--create` to let the exporter invoke
`ollama create`. GGUF conversion is separate and deferred unless a llama.cpp
checkout and merged model are available.

## 10. Stop safely

1. Stop and label any active demonstration.
2. Click **Stop** and wait for the Dashboard to show `idle`.
3. If input or focus behaves unexpectedly, click **Emergency stop** first.
4. Close the GUI terminal with `Ctrl+C` only after the loop is idle.

The emergency stop clears queued planner execution and blocks a restart until
cleared. If the process was killed abruptly, verify no Python process remains
before re-enabling keyboard input.
