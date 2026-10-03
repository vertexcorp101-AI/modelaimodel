@echo off
setlocal
title LightBrain - launch (browser test)
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found on this PC.
  echo Install Python 3.11 or newer from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during setup.
  pause
  exit /b 1
)

echo --- Checking package (numpy)...
python -c "import numpy" >nul 2>nul
if errorlevel 1 (
  echo --- First run: installing numpy...
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet numpy
)
REM torch is only needed to TRAIN the tiny fallback model, not to chat.

echo --- Checking the web server is not already running...
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:7860/api/health' -TimeoutSec 1).StatusCode | Out-Null; exit 0 } catch { exit 1 }" >nul 2>nul
if not errorlevel 1 goto launched

echo --- Starting the LightBrain server (brain: SmolLM2 via llamafile)...
set LOG=runs\server.log
if exist "%LOG%" del "%LOG%"
start "LightBrain Server - close this window to stop" /min cmd /c "python server.py > runs\server.log 2>&1"

echo --- Waiting for the brain to be ready (first load can take up to a minute)...
for /l %%i in (1,1,120) do (
  powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:7860/api/health' -TimeoutSec 1).StatusCode | Out-Null; exit 0 } catch { exit 1 }" >nul 2>nul
  if not errorlevel 1 goto launched
  ping -n 2 127.0.0.1 >nul
)

echo [ERROR] The server did not become ready in time.
echo --- Tail of the server log ---
powershell -NoProfile -Command "Get-Content -LiteralPath 'runs\server.log' -Tail 25 -ErrorAction SilentlyContinue"
echo.
echo Hint: drop .txt / .md files into the knowledge folder and ask about them.
pause
exit /b 1

:launched
echo --- Opening your browser at http://localhost:7860
start "" http://localhost:7860
echo.
echo Running! Keep the "LightBrain Server" window open; close it to stop.
echo Re-run this file anytime to chat again.
timeout /t 3 >nul
exit /b 0