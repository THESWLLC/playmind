[CmdletBinding()]
param(
    [switch]$WithTraining
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Get-Command py -ErrorAction SilentlyContinue
if ($Python) {
    $PythonExe = "py"
    $PythonArgs = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonExe = "python"
    $PythonArgs = @()
} else {
    throw "Python 3.10+ was not found. Install it from https://python.org and enable 'Add Python to PATH'."
}

& $PythonExe @PythonArgs -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'; print(sys.version)"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $PythonExe @PythonArgs -m venv .venv
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
if (Test-Path "requirements.txt") {
    & $VenvPython -m pip install -r requirements.txt
}
& $VenvPython -m pip install -r requirements-playmind.txt
if ($WithTraining) {
    & $VenvPython -m pip install -r requirements-playmind-ml.txt
}

Write-Host ""
Write-Host "PlayMind setup complete." -ForegroundColor Green
Write-Host "Run .\start_playmind.bat"
