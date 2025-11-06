@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"
python dashboard/integrated_washer_app.py
if errorlevel 1 (
    echo.
    echo エラーが発生しました。何かキーを押して終了してください...
    pause >nul
)

