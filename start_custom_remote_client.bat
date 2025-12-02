@echo off
chcp 65001 >nul
echo ========================================
echo カスタムリモートデスクトップ - クライアント
echo ノートパソコン側で実行してください
echo ========================================
echo.

set /p SERVER_IP="サーバーIPアドレスを入力してください (デフォルト: 192.168.1.100): "
if "%SERVER_IP%"=="" set SERVER_IP=192.168.1.100

python custom_remote_desktop.py client %SERVER_IP%

pause






