@echo off
chcp 65001 > nul
echo ========================================
echo ノートパソコン側 - Python確認スクリプト
echo ========================================
echo.

echo [1] python --version の結果を確認...
python --version 2>&1
echo.

echo [2] python -V で確認...
python -V 2>&1
echo.

echo [3] py ランチャーで確認...
py --version 2>&1
echo.

echo [4] Pythonのパスを確認...
where python 2>nul
if %errorlevel% neq 0 (
    echo ❌ python コマンドのパスが見つかりません
) else (
    echo ✅ python コマンドのパス:
    where python
)
echo.

echo [5] Pythonが実際に動作するか確認...
python -c "import sys; print('Python', sys.version)" 2>&1
echo.

echo [6] pipの確認...
pip --version 2>nul
if %errorlevel% neq 0 (
    echo ❌ pip コマンドが見つかりません
    echo.
    echo python -m pip で確認...
    python -m pip --version 2>&1
) else (
    echo ✅ pipが利用できます
    pip --version
)
echo.

echo ========================================
echo 結果サマリー
echo ========================================
echo.

python -c "import sys; print(sys.version)" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Pythonは動作しています
    echo    次のステップ: ライブラリをインストールしてください
    echo    python -m pip install mss pillow pyautogui pyperclip cryptography
) else (
    echo ❌ Pythonが正しく動作していない可能性があります
    echo    次のステップ: Pythonを再インストールしてください
    echo    1. https://www.python.org/downloads/ からダウンロード
    echo    2. インストール時に「Add Python to PATH」にチェックを入れる
)

echo.
echo ========================================
pause



