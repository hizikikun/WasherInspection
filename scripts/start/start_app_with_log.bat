@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"

echo Starting application...
echo Python: python
echo Script: dashboard/integrated_washer_app.py

python dashboard/integrated_washer_app.py > app_startup.log 2>&1

if errorlevel 1 (
    echo.
    echo ========================================
    echo エラーが発生しました
    echo ========================================
    echo ログファイルを確認してください: app_startup.log
    type app_startup.log
    echo.
    echo 何かキーを押して終了してください...
    pause >nul
)












