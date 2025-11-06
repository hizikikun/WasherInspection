@echo off
chcp 65001 > nul
echo ========================================
echo WasherInspection リモートサーバー起動
echo ========================================
echo.

cd /d "%~dp0"

python scripts\start_remote_server.py

pause





