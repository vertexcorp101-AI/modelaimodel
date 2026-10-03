@echo off
setlocal
title LightBrain - OpenAI-compatible provider (local test)
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found on this PC.
  pause
  exit /b 1
)

set KEY=%1
if "%KEY%"=="" (
  echo --- Generating an API key for this run (set one with: run_provider.bat my-key)...
  for /f %%k in ('python provider.py --gen-key') do set "KEY=%%k"
)
echo --- API key for this run:
echo     %KEY%
echo --- Use it as:  Authorization: Bearer %KEY%
echo --- Note: for Belmo deploys, set API_KEY in the dashboard env instead.

echo --- Starting provider on 127.0.0.1:3000 (model downloads on first run)...
cmd /c "set API_KEY=%KEY%&& python provider.py --host 127.0.0.1 --port 3000"