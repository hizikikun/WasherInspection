@echo off
chcp 65001 > nul
echo ========================================
echo 同じPythonバージョンをダウンロードする支援スクリプト
echo ========================================
echo.
echo このスクリプトは、デスクトップ側と同じPythonバージョンを
echo ダウンロードするためのURLを表示します。
echo.
echo デスクトップ側のPythonバージョンを確認してください。
echo.
pause

echo.
echo ========================================
echo デスクトップ側のPythonバージョンを確認
echo ========================================
echo.
echo デスクトップ側で以下を実行してください:
echo   python --version
echo.
echo 表示されたバージョン番号をメモしてください。
echo.
pause

echo.
echo ========================================
echo ダウンロード方法
echo ========================================
echo.
echo 方法1: Python公式サイトからダウンロード（推奨）
echo.
echo 1. ブラウザで以下にアクセス:
echo    https://www.python.org/downloads/release/
echo.
echo 2. デスクトップ側と同じバージョンを探す
echo    例: Python 3.13.9 の場合
echo       https://www.python.org/downloads/release/python-3139/
echo.
echo 3. 「Windows installer (64-bit)」をクリック
echo.
echo 方法2: 直接URLでダウンロード
echo.
echo デスクトップ側が Python 3.13.9 の場合:
echo   https://www.python.org/ftp/python/3.13.9/python-3.13.9-amd64.exe
echo.
echo デスクトップ側が Python 3.12.7 の場合:
echo   https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe
echo.
echo デスクトップ側が Python 3.11.10 の場合:
echo   https://www.python.org/ftp/python/3.11.10/python-3.11.10-amd64.exe
echo.
echo ========================================
echo インストール時の注意
echo ========================================
echo.
echo ⚠️ 重要: インストール時に以下にチェックを入れる:
echo   ✅ Add Python to PATH
echo   ✅ Install launcher for all users
echo.
echo ========================================
pause



