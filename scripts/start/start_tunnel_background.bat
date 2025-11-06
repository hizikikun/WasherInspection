@echo off
chcp 65001 > nul
echo ========================================
echo インターネット経由アクセストンネル起動
echo ========================================
echo.
echo 学習中でも安全に実行できます。
echo.

cd /d "%~dp0"

echo トンネルを起動中...
echo 数秒後にURLが表示されます...
echo.

start /B python scripts\remote_server_tunnel.py --start

timeout /t 5 /nobreak > nul

echo.
echo トンネルが起動しました。
echo.
echo アクセスURLを確認中...
echo.
echo 注意: ngrokのURLは以下の方法で確認できます:
echo   1. ブラウザで http://localhost:4040 にアクセス（ngrokローカルAPI）
echo   2. または、数秒待ってから上記URLにアクセスしてください
echo.
echo このウィンドウは閉じてもトンネルは動作し続けます。
echo トンネルを停止するには、タスクマネージャーでPythonプロセスを終了してください。
echo.
pause





