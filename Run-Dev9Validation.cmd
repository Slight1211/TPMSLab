@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Run-Dev9Validation.ps1" %*
pause
