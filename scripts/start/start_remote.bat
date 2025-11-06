@echo off
REM このスクリプトはどこからでも実行可能
REM プロジェクトディレクトリに自動的に移動します

chcp 65001 > nul

REM このバッチファイルの場所を取得
set "BATCH_DIR=%~dp0"
cd /d "%BATCH_DIR%"

echo ========================================
echo リモートアクセス起動
echo ========================================
echo.
echo プロジェクトディレクトリ: %CD%
echo.

REM セットアップが完了しているか確認
if not exist "config\remote_tunnel_config.json" (
    echo [INFO] リモートアクセスのセットアップを実行します...
    echo.
    python scripts\auto_setup_remote_access.py
    echo.
    timeout /t 2 /nobreak > nul
)

REM リモートサーバーが起動しているか確認
echo [INFO] リモートサーバーの状態を確認中...
netstat -ano | findstr :5000 >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] リモートサーバーは既に起動しています。
    echo.
) else (
    echo [INFO] リモートサーバーを起動中...
    start /B python scripts\remote_server.py
    timeout /t 2 /nobreak > nul
    echo [OK] リモートサーバーを起動しました。
    echo.
)

echo ========================================
echo アクセス情報
echo ========================================
echo.
echo ローカルアクセス: http://localhost:5000
echo.

REM IPアドレスを取得
python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print('リモートアクセス: http://' + s.getsockname()[0] + ':5000'); s.close()" 2>nul
if %errorlevel% neq 0 (
    echo リモートアクセス: IPアドレスの取得に失敗しました
)
echo.

echo ========================================
echo インターネット経由アクセスを有効にするには:
echo ========================================
echo 別のコマンドプロンプトで以下を実行:
echo   cd /d "%CD%"
echo   python scripts\remote_server_tunnel.py --start
echo.
echo または、統合アプリの「🌍 インターネット経由アクセス」ボタンを使用
echo.
echo このウィンドウは閉じてもサーバーは動作し続けます。
echo.
pause





