@echo off
chcp 65001 > nul
echo ========================================
echo 学習中でも実行可能なリモートアクセスセットアップ
echo ========================================
echo.
echo このスクリプトは学習中でも安全に実行できます。
echo 学習プロセスには影響しません。
echo.

cd /d "%~dp0"

echo [1/2] リモートアクセスの自動セットアップを実行中...
python scripts\auto_setup_remote_access.py

echo.
echo [2/2] リモートサーバーを起動中...
echo.
echo 注意: リモートサーバーはバックグラウンドで実行されます。
echo 学習中でも安全に使用できます。
echo.

start /B python scripts\remote_server.py

timeout /t 3 /nobreak > nul

echo.
echo ========================================
echo セットアップ完了！
echo ========================================
echo.
echo リモートサーバーが起動しました。
echo.
echo ローカルアクセス: http://localhost:5000
echo.
echo インターネット経由アクセスを有効にするには:
echo   別のコマンドプロンプトで以下を実行:
echo   python scripts\remote_server_tunnel.py --start
echo.
echo または、統合アプリの「🌍 インターネット経由アクセス」ボタンを使用
echo.
pause





