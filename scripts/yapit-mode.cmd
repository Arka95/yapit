@echo off
setlocal
if exist "%~dp0..\.venv\Scripts\yapit.exe" (
  "%~dp0..\.venv\Scripts\python.exe" "%~dp0run-mode.py" %*
) else if exist "%~dp0..\.venv-cpu\Scripts\yapit.exe" (
  "%~dp0..\.venv-cpu\Scripts\python.exe" "%~dp0run-mode.py" %*
) else if exist "%~dp0..\.venv-gpu\Scripts\yapit.exe" (
  "%~dp0..\.venv-gpu\Scripts\python.exe" "%~dp0run-mode.py" %*
) else (
  py -3.12 "%~dp0run-mode.py" %*
)
exit /b %errorlevel%
