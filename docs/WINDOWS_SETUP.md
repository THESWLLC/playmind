# Windows setup

## Requirements

- Windows 10/11 and PowerShell
- Python 3.10+ from python.org with the launcher or `python` on `PATH`
- An owned/authorized game for local capture
- Optional: Tesseract OCR and Ollama
- Optional training: NVIDIA GPU; WSL2 is recommended

Do not use PlayMind with an official MMO client whose terms prohibit
automation.

## Install

Open PowerShell in the repository:

```powershell
Copy-Item config\owned_game.example.json config\owned_game.json
notepad config\owned_game.json
.\setup_windows.ps1
```

The script verifies Python 3.10+, creates `.venv`, upgrades pip, installs
`requirements.txt` when present, then installs `requirements-playmind.txt`.
Use `.\setup_windows.ps1 -WithTraining` to also install
`requirements-playmind-ml.txt`; real planner training still needs `trl` and
`bitsandbytes` explicitly, and WSL2 is the preferred GPU environment.

At minimum edit:

```json
{
  "i_own_this_game": false,
  "enable_keyboard": false,
  "mode": "shadow",
  "capture": {"window_title": "Exact Owned Game Window Title"}
}
```

Calibrate ROIs and keymap before considering live input. Keep ownership and
keyboard false during setup and demonstration review.

## Start

Double-click `start_playmind.bat` or run:

```powershell
.\start_playmind.bat
.\start_playmind.bat -Port 8778 -NoBrowser
```

The batch file invokes `start_playmind.ps1` relative to the repository. If
`.venv` is missing, setup runs automatically. Startup activates the venv, runs
System Doctor, opens the local URL, and starts `scripts/start_all.py`.

Direct diagnostics:

```powershell
.\.venv\Scripts\python.exe scripts\doctor.py
.\.venv\Scripts\python.exe scripts\doctor.py --json
.\.venv\Scripts\python.exe scripts\start_all.py --dry-run
```

Doctor checks Python, config validity, capture/training modules, Ollama, CUDA,
RAM, disk, permissions, and llama.cpp availability. Its top-level `ok` means
only supported Python plus structurally valid config; inspect individual
sections for optional failures.

## Troubleshooting basics

- **PowerShell blocks scripts:** the `.bat` launcher already uses
  `-ExecutionPolicy Bypass`. For direct setup use
  `powershell.exe -ExecutionPolicy Bypass -File .\setup_windows.ps1`.
- **Python not found:** reinstall Python 3.10+ and enable “Add Python to PATH,”
  then reopen PowerShell.
- **Port in use:** start with `-Port 8778`.
- **Window not found/focus loss:** match `capture.window_title` exactly; keep
  shadow and live keyboard off while calibrating.
- **No physical events/F9:** rerun setup and confirm `pynput` in Doctor. Some
  elevated games require PlayMind to run at the same integrity level.
- **OCR unavailable:** install Tesseract separately and ensure its executable
  is on `PATH`; installing `pytesseract` alone is not the OCR engine.
- **Ollama unreachable:** start Ollama, run `ollama list`, and pull/create the
  configured model. Planner V2 falls back to scripted planning.
- **Training packages fail or CUDA is absent:** use WSL2 and
  [WSL2_TRAINING.md](./WSL2_TRAINING.md).

For runtime and model-specific failures see
[TROUBLESHOOTING.md](./TROUBLESHOOTING.md).
