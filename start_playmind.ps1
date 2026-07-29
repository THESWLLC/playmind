[CmdletBinding()]
param(
    [int]$Port = 8777,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Virtual environment missing; running setup_windows.ps1..." -ForegroundColor Yellow
    & (Join-Path $Root "setup_windows.ps1")
}

. (Join-Path $Root ".venv\Scripts\Activate.ps1")
& $Python (Join-Path $Root "scripts\doctor.py")
if ($LASTEXITCODE -ne 0) {
    Write-Warning "System Doctor reported a configuration issue. The GUI will still start in safe shadow mode."
}

$Url = "http://127.0.0.1:$Port/"
if (-not $NoBrowser) {
    Start-Process $Url
}
& $Python (Join-Path $Root "scripts\start_all.py") --host 127.0.0.1 --port $Port --no-browser
