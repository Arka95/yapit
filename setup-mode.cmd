@echo off
setlocal
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0scripts\setup-mode.py" %*
) else (
  py -3.12 "%~dp0scripts\setup-mode.py" %*
)
exit /b %errorlevel%
