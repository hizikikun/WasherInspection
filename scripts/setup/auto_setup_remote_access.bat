@echo off
chcp 65001 > nul
echo ========================================
echo リモートアクセス自動セットアップ
echo ========================================
echo.

cd /d "%~dp0"

python scripts\auto_setup_remote_access.py

echo.
pause





