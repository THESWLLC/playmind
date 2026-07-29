@echo off
:: Elevated Ascension play: vision LLM reads the screen, acts, and learns.
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
echo PlayMind SCREEN-LLM starting in 3s — hands off Ascension
timeout /t 3 /nobreak >nul
"%PY%" -u scripts\run_owned_loop.py --config config\owned_game.json --live --ollama --vision-model qwen2.5vl:7b --ollama-model llama3.2 --learn --directive farm --max-ticks 80
echo.
echo Tip: for a live LLM thinking window use scripts\PLAY_WITH_GUI.bat
echo Done. Exit code %ERRORLEVEL%
pause
