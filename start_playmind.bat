@echo off
setlocal
set "ROOT=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%start_playmind.ps1" %*
exit /b %ERRORLEVEL%
