@echo off
chcp 65001 > nul
echo ========================================
echo リモートアクセス起動（どこからでも実行可能）
echo ========================================
echo.

REM このスクリプトの場所を取得
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo プロジェクトディレクトリに移動しました: %CD%
echo.

REM セットアップが完了しているか確認
if not exist "config\remote_tunnel_config.json" (
    echo [INFO] リモートアクセスのセットアップがまだ完了していません。
    echo [INFO] 自動セットアップを実行します...
    echo.
    call scripts\auto_setup_remote_access.py
    echo.
)

REM リモートサーバーが起動しているか確認
echo [INFO] リモートサーバーの状態を確認中...
netstat -ano | findstr :5000 >nul 2>&1
if %errorlevel% == 0 (
    echo [INFO] リモートサーバーは既に起動しています。
    echo.
    echo アクセスURL:
    echo   ローカル: http://localhost:5000
    echo.
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(f'   リモート: http://{s.getsockname()[0]}:5000'); s.close()" 2>nul || echo   リモート: IPアドレスの取得に失敗しました
    echo.
) else (
    echo [INFO] リモートサーバーを起動中...
    start /B python scripts\remote_server.py
    timeout /t 2 /nobreak > nul
    echo [OK] リモートサーバーを起動しました。
    echo.
    echo アクセスURL:
    echo   ローカル: http://localhost:5000
    echo.
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(f'   リモート: http://{s.getsockname()[0]}:5000'); s.close()" 2>nul || echo   リモート: IPアドレスの取得に失敗しました
    echo.
)

echo ========================================
echo インターネット経由アクセスを有効にするには:
echo ========================================
echo   別のコマンドプロンプトで以下を実行:
echo   cd /d "%CD%"
echo   python scripts\remote_server_tunnel.py --start
echo.
echo または、統合アプリの「🌍 インターネット経由アクセス」ボタンを使用
echo.
echo このウィンドウは閉じてもサーバーは動作し続けます。
echo.
pause





