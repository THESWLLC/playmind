[CmdletBinding()]
param(
    [switch]$Run
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$WslRoot = (wsl.exe wslpath -a $Root).Trim()
$Command = "cd '$WslRoot' && python3 -m venv .venv-wsl && source .venv-wsl/bin/activate && python -m pip install --upgrade pip && python -m pip install -r requirements-playmind.txt -r requirements-playmind-ml.txt && python scripts/doctor.py"

Write-Host "WSL training setup command:"
Write-Host $Command -ForegroundColor Cyan
if ($Run) {
    wsl.exe bash -lc $Command
} else {
    Write-Host ""
    Write-Host "Review it, then run: .\setup_wsl_training.ps1 -Run"
}
