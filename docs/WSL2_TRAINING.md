# WSL2 planner training

Use Windows for game capture and WSL2 for local NVIDIA training. Training reads
exported JSONL and does not need the game running.

## Prerequisites

1. Install WSL2 and a recent Ubuntu distribution.
2. Install a Windows NVIDIA driver with WSL CUDA support.
3. In WSL, confirm `nvidia-smi` sees the RTX GPU.
4. Do not install a separate Linux NVIDIA kernel driver inside WSL.
5. Export and review planner datasets from Windows first.

## Repository bootstrap

From Windows PowerShell in the repository:

```powershell
.\setup_wsl_training.ps1
```

This prints, but does not run, a command that converts the current repository
path to WSL form, creates `.venv-wsl`, installs both PlayMind requirements
files, and runs Doctor. Review it, then:

```powershell
.\setup_wsl_training.ps1 -Run
```

The script does not install `trl` or `bitsandbytes`, which real planner
training requires. In WSL:

```bash
cd /mnt/c/path/to/repo
source .venv-wsl/bin/activate
python -m pip install trl bitsandbytes
python scripts/doctor.py
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

If the repository is on another drive, use `/mnt/d/...`, etc. Never activate
the Windows `.venv` from WSL; use `.venv-wsl`.

## Train from the `/mnt/c` repository

```bash
cd /mnt/c/path/to/repo
source .venv-wsl/bin/activate

python scripts/export_planner_sft.py
python scripts/train_planner_sft.py \
  --base-model /home/<user>/models/<licensed-3b-model> \
  --preset rtx_4070_ti_3b_qlora
```

Dataset paths under the repository work directly. Large model caches,
checkpoints, and repeated small-file operations can be slower on `/mnt/c`.
Prefer a Linux-filesystem base-model/cache path (for example
`/home/<user>/models` and `HF_HOME=/home/<user>/.cache/huggingface`). If moving
the whole repository into WSL for speed, copy exported data deliberately and
avoid training from a stale checkout.

## GPU passthrough checks

Both checks must pass:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Doctor's `cuda.driver_visible` and `torch.cuda_available` are separate. A
visible `nvidia-smi` with false PyTorch CUDA usually means a CPU-only or
incompatible PyTorch build. Reinstall PyTorch using the command for the
supported CUDA build from pytorch.org, then retest before installing more
packages.

For the 4070 Ti preset, confirm the run manifest reports
`quantization: "4bit-nf4"`. A warning that bitsandbytes QLoRA is unavailable
means the trainer fell back to much larger full LoRA; stop and fix the
environment instead of risking an OOM.

## WSL operational notes

- Allocate enough WSL RAM/swap, but GPU VRAM remains the primary QLoRA limit.
- Close Windows applications using significant GPU memory.
- Use `nvidia-smi` from another terminal to observe memory.
- Keep base model, adapter, dataset manifest, preset, seed, and metrics
  together for reproducibility.
- Windows Ollama and WSL networking may not share the same localhost behavior
  on every setup. Evaluation can run on Windows after training/export.
- The setup script is bootstrap convenience, not proof that CUDA QLoRA works.

See [PLANNER_TRAINING.md](./PLANNER_TRAINING.md) for presets and OOM handling.
