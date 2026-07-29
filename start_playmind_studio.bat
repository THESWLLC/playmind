@echo off
rem PlayMind Studio: offline recording import, review, training, and evaluation.
setlocal
set "ROOT=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%start_playmind_studio.ps1" %*
exit /b %ERRORLEVEL%
