@echo off
call "%~dp0START.bat" --no-virtual-camera %*
exit /b %errorlevel%
