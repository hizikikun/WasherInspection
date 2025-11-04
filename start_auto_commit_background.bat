@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM Start auto-commit script in background (minimized window)
start /min powershell -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0test_auto_commit.ps1"

REM Optional: Wait a moment to check if it started successfully
timeout /t 2 /nobreak >nul

echo Auto-commit script started in background.
echo Check auto-commit.log for status.
pause

