@echo off
chcp 65001 >nul
echo ========================================
echo リモートデスクトップ状態確認
echo ========================================
echo.

PowerShell -ExecutionPolicy Bypass -File "%~dp0check_remote_desktop_status.ps1"

pause







