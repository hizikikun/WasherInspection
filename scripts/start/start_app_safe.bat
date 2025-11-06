@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"

echo アプリケーションを起動しています...
echo.

REM コンソールウィンドウを表示するバージョンで起動（エラー確認用）
python dashboard\integrated_washer_app_visible.py

if errorlevel 1 (
    echo.
    echo エラーが発生しました。
    echo エラーログを確認してください: app_error_log.txt
    echo.
    pause
)







