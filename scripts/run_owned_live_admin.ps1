# Elevate and run the owned Ascension live loop.
$ErrorActionPreference = "Stop"
$root = "c:\Users\Shawn\playmind"
Set-Location $root
$env:PYTHONPATH = $root
$env:PYTHONUNBUFFERED = "1"

$python = "C:\Users\Shawn\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$logDir = Join-Path $root "data\playmind"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "admin_run.log"
Remove-Item $log -ErrorAction SilentlyContinue

function Log([string]$msg) {
  $line = "$(Get-Date -Format o) $msg"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $msg
}

try {
  $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  Log "Admin=$isAdmin cwd=$(Get-Location) python=$python"
  if (-not (Test-Path $python)) { throw "Python not found: $python" }

  Log "Starting live farm loop (120 ticks) — hands off Ascension"
  Start-Sleep -Seconds 2
  & $python -u "$root\scripts\run_owned_loop.py" --config "$root\config\owned_game.json" --live --directive farm --max-ticks 120 2>&1 | ForEach-Object { Log "$_" }
  Log "done exit=$LASTEXITCODE"
  exit $LASTEXITCODE
}
catch {
  Log "ERROR: $_"
  exit 1
}
