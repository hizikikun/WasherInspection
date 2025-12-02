@echo off
chcp 65001 > nul
echo ========================================
echo Python インストール確認スクリプト
echo ========================================
echo.

echo [1] Pythonコマンドの確認...
python --version 2>nul
if %errorlevel% neq 0 (
    echo ❌ python コマンドが見つかりません
    echo.
    echo [1-2] py ランチャーの確認...
    py --version 2>nul
    if %errorlevel% neq 0 (
        echo ❌ py コマンドも見つかりません
        echo.
        echo 結論: Pythonがインストールされていない可能性が高いです
    ) else (
        echo ✅ py ランチャーが見つかりました
        py --version
        echo.
        echo Pythonはインストールされていますが、pythonコマンドが使えない状態です
        echo py コマンドを使用するか、PATHを設定してください
    )
) else (
    echo ✅ Pythonがインストールされています
    python --version
)

echo.
echo ========================================
echo [2] Pythonのインストール場所を確認
echo ========================================
echo.

where python 2>nul
if %errorlevel% neq 0 (
    echo ❌ python コマンドのパスが見つかりません
) else (
    echo ✅ python コマンドのパス:
    where python
)

echo.
where py 2>nul
if %errorlevel% neq 0 (
    echo ❌ py ランチャーのパスが見つかりません
) else (
    echo ✅ py ランチャーのパス:
    where py
)

echo.
echo ========================================
echo [3] 一般的なPythonインストール場所を確認
echo ========================================
echo.

if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python" (
    echo ✅ Pythonフォルダが見つかりました:
    dir "C:\Users\%USERNAME%\AppData\Local\Programs\Python" /b
) else (
    echo ❌ C:\Users\%USERNAME%\AppData\Local\Programs\Python が見つかりません
)

if exist "C:\Python3*" (
    echo ✅ C:\Python3* フォルダが見つかりました:
    dir "C:\Python3*" /b /ad
) else (
    echo ❌ C:\Python3* フォルダが見つかりません
)

echo.
echo ========================================
echo [4] pipの確認
echo ========================================
echo.

pip --version 2>nul
if %errorlevel% neq 0 (
    echo ❌ pip コマンドが見つかりません
    echo.
    python -m pip --version 2>nul
    if %errorlevel% neq 0 (
        echo ❌ python -m pip も使えません
    ) else (
        echo ✅ python -m pip は使えます
        python -m pip --version
    )
) else (
    echo ✅ pipが利用できます
    pip --version
)

echo.
echo ========================================
echo 結果サマリー
echo ========================================
echo.

python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Pythonはインストールされています
    echo    次のステップ: 必要なライブラリをインストールしてください
    echo    python -m pip install mss pillow pyautogui pyperclip cryptography
) else (
    py --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo ⚠️  py ランチャーは使えますが、python コマンドが使えません
        echo    次のステップ: py コマンドを使用するか、Pythonを再インストールしてください
    ) else (
        echo ❌ Pythonがインストールされていないようです
        echo    次のステップ: Pythonをインストールしてください
        echo    1. https://www.python.org/downloads/ からダウンロード
        echo    2. インストール時に「Add Python to PATH」にチェックを入れる
    )
)

echo.
echo ========================================
pause



