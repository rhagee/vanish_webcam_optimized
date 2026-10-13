@echo off
call "%~dp0START.bat" --virtual-camera %*
exit /b %errorlevel%
