@echo off
chcp 65001 > nul
echo ========================================
echo インターネット経由アクセストンネル起動
echo ========================================
echo.

cd /d "%~dp0"

python scripts\remote_server_tunnel.py --start

pause





