@echo off
chcp 65001 > nul
echo ========================================
echo リモートサーバー起動（バックグラウンド）
echo ========================================
echo.
echo 学習中でも安全に実行できます。
echo.

cd /d "%~dp0"

echo リモートサーバーを起動中...
start /B python scripts\remote_server.py

timeout /t 2 /nobreak > nul

echo.
echo リモートサーバーが起動しました。
echo.
echo ローカルアクセス: http://localhost:5000
echo.
echo IPアドレスを確認中...
python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(f'リモートアクセス: http://{s.getsockname()[0]}:5000'); s.close()" 2>nul || echo IPアドレスの取得に失敗しました

echo.
echo このウィンドウは閉じてもサーバーは動作し続けます。
echo サーバーを停止するには、タスクマネージャーでPythonプロセスを終了してください。
echo.
pause





