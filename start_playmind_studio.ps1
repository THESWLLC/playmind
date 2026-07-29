[CmdletBinding()]
param(
    [int]$Port = 8787,
    [switch]$NoBrowser,
    [string]$Config
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Studio virtual environment missing; running setup..." -ForegroundColor Yellow
    & (Join-Path $Root "setup_playmind_studio.ps1")
}

Write-Host "Running offline Studio doctor..." -ForegroundColor Cyan
$DoctorArgs = @((Join-Path $Root "scripts\start_studio.py"), "--dry-run", "--port", $Port)
if ($Config) { $DoctorArgs += @("--config", $Config) }
& $Python @DoctorArgs
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Studio doctor failed. Review the output before importing recordings."
}

$Url = "http://127.0.0.1:$Port/"
if (-not $NoBrowser) {
    Start-Process $Url
}

$StartArgs = @((Join-Path $Root "scripts\start_studio.py"), "--host", "127.0.0.1", "--port", $Port, "--no-browser")
if ($Config) { $StartArgs += @("--config", $Config) }
& $Python @StartArgs
