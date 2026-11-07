@echo off
setlocal
if exist "%~dp0..\.venv\Scripts\python.exe" (
  "%~dp0..\.venv\Scripts\python.exe" "%~dp0run-mode.py" %*
) else (
  py -3.12 "%~dp0run-mode.py" %*
)
exit /b %errorlevel%
