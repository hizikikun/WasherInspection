@echo off
chcp 65001 > nul
echo ========================================
echo リモートサーバー環境セットアップ
echo ========================================
echo.

cd /d "%~dp0"

echo 必要なパッケージをインストール中...
pip install flask flask-cors

echo.
echo ========================================
echo セットアップ完了！
echo ========================================
echo.
echo サーバーを起動するには:
echo   start_remote_server.bat
echo.
echo または:
echo   python scripts\start_remote_server.py
echo.
pause





