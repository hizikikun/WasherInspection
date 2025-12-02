@echo off
chcp 65001 >nul
echo ========================================
echo リモートデスクトップ管理アプリ
echo ========================================
echo.

python remote_desktop_manager.py

if errorlevel 1 (
    echo.
    echo エラーが発生しました。
    echo Pythonがインストールされているか確認してください。
    pause
)







