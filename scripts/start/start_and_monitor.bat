@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"

echo ========================================
echo アプリ起動と監視を開始します
echo ========================================
echo.

REM アプリを起動
start "ワッシャー検査アプリ" python dashboard/integrated_washer_app.py

REM 少し待機
timeout /t 3 /nobreak >nul

REM 監視スクリプトを起動
echo 監視を開始します...
python monitor_app.py











