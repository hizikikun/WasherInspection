@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"

echo ========================================
echo ワッシャー検査アプリケーション起動中...
echo ========================================
echo.

python dashboard/integrated_washer_app.py > app_startup.log 2>&1
type app_startup.log

if errorlevel 1 (
    echo.
    echo ========================================
    echo エラーが発生しました
    echo ========================================
    echo.
    echo エラーログを確認してください: app_error_log.txt
    echo 起動ログを確認してください: app_startup.log
    echo.
    pause
)


