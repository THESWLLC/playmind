@echo off
rem OWNED-GAME LAB launcher (port 8777). This is NOT PlayMind Studio.
rem For offline Studio use start_playmind_studio.bat (port 8787).
setlocal
set "ROOT=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%start_playmind.ps1" %*
exit /b %ERRORLEVEL%
