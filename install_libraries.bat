@echo off
chcp 65001 > nul
echo ========================================
echo Chrome風リモートデスクトップ用ライブラリのインストール
echo ========================================
echo.
echo Python 3.13.9 が確認されました
echo.
echo 必要なライブラリをインストールします...
echo.
pause

echo.
echo [1] pipをアップグレード中...
python -m pip install --upgrade pip

echo.
echo [2] 必要なライブラリをインストール中...
python -m pip install mss pillow pyautogui pyperclip cryptography

echo.
echo ========================================
echo インストール完了
echo ========================================
echo.
echo インストールされたライブラリを確認します...
echo.
python -m pip list | findstr "mss pillow pyautogui pyperclip cryptography"

echo.
echo ========================================
echo セットアップ確認を実行します
echo ========================================
echo.
python check_chrome_like_setup.py

echo.
pause



