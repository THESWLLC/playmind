@echo off
:: Elevated Ascension play + live Brain GUI (LLM thinking log in browser).
:: Runs until you click Stop in the GUI (ticks=0).
net session >nul 2>&1
if %errorLevel% NEQ 0 (
  echo Requesting Administrator so keys reach Ascension...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

cd /d c:\Users\Shawn\playmind
set PYTHONPATH=c:\Users\Shawn\playmind
set PYTHONUNBUFFERED=1
set PY=C:\Users\Shawn\AppData\Local\Python\pythoncore-3.14-64\python.exe
set OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe

echo Ensuring vision model qwen2.5vl:7b is available...
"%OLLAMA%" pull qwen2.5vl:7b
echo.
echo Opening PlayMind Brain GUI — Start live, ticks=0 means run until Stop.
echo Vision model: qwen2.5vl:7b
echo http://127.0.0.1:8777/
"%PY%" -u scripts\run_owned_gui.py
echo.
echo Done. Exit code %ERRORLEVEL%
pause
