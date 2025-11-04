@echo off
chcp 65001 >nul
echo Starting GitHub Auto-Commit System...
echo Log file: auto-commit.log
echo Press Ctrl+C to stop
echo.

cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; & '.\auto-commit.ps1'"

pause
