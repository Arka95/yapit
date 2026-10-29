@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Set up the Python environment first. See README.md.
  pause
  exit /b 1
)
start "Yapit Voice Enrollment" ".venv\Scripts\pythonw.exe" -m yapit.desktop
