@echo off
setlocal
call "%~dp0scripts\yapit-mode.cmd" default %*
exit /b %errorlevel%
