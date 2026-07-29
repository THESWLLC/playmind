[CmdletBinding()]
param(
    [string]$Python = "py"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Venv = Join-Path $Root ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating PlayMind Studio virtual environment..." -ForegroundColor Cyan
    & $Python -3 -m venv $Venv
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $Root "requirements-playmind.txt")

Write-Host ""
Write-Host "Studio Python setup complete." -ForegroundColor Green
Write-Host "FFmpeg is also required for video import and frame extraction." -ForegroundColor Yellow
Write-Host "Install it separately (for example: winget install Gyan.FFmpeg), then ensure ffmpeg.exe and ffprobe.exe are on PATH."
Write-Host "Start Studio with: .\start_playmind_studio.bat"
