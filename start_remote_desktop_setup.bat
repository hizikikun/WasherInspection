@echo off
chcp 65001 >nul
echo ========================================
echo リモートデスクトップ設定
echo ========================================
echo.
echo このスクリプトは、Windowsリモートデスクトップ（RDP）を有効化します。
echo.
echo 注意: 管理者権限が必要です。
echo.
pause

PowerShell -ExecutionPolicy Bypass -File "%~dp0setup_remote_desktop.ps1"

pause







