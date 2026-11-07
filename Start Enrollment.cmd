@echo off
setlocal
set "mode=cpu"
if not "%~1"=="" set "mode=%~1"
call "%~dp0yapit-mode.cmd" %mode% enroll-app
exit /b %errorlevel%
