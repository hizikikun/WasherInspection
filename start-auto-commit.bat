@echo off
echo GitHub自動転送システムを開始します...
echo ログファイル: auto-commit.log
echo 停止するには Ctrl+C を押してください
echo.

cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "auto-commit.ps1"

pause
