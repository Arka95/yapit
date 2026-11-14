@echo off
setlocal
set "mode=default"
if not "%~1"=="" set "mode=%~1"
call "%~dp0yapit-mode.cmd" %mode% enroll-voice
exit /b %errorlevel%
