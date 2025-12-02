@echo off
chcp 65001 > nul
echo ========================================
echo Python 再インストール支援スクリプト
echo ========================================
echo.
echo このスクリプトは、Pythonの再インストールを支援します。
echo.
echo 注意: このスクリプトは情報を表示するだけです。
echo 実際のアンインストールとインストールは手動で行ってください。
echo.
pause

echo.
echo ========================================
echo 現在のPythonの状態を確認します
echo ========================================
echo.

echo [1] Pythonのバージョンを確認...
python --version 2>nul
if %errorlevel% neq 0 (
    echo ❌ Pythonがインストールされていないか、PATHに追加されていません
) else (
    echo ✅ Pythonがインストールされています
)

echo.
echo [2] pipのバージョンを確認...
pip --version 2>nul
if %errorlevel% neq 0 (
    echo ❌ pipが利用できません
) else (
    echo ✅ pipが利用できます
)

echo.
echo [3] Pythonのインストール場所を確認...
where python 2>nul
if %errorlevel% neq 0 (
    echo ❌ Pythonのパスが見つかりません
) else (
    echo ✅ Pythonのパスが見つかりました
)

echo.
echo ========================================
echo 次のステップ
echo ========================================
echo.
echo 1. Pythonをアンインストールする場合:
echo    - コントロールパネル → プログラムと機能
echo    - Python関連のプログラムをすべてアンインストール
echo.
echo 2. Pythonを再インストールする場合:
echo    - https://www.python.org/downloads/ からダウンロード
echo    - インストール時に「Add Python to PATH」にチェックを入れる
echo.
echo 3. 詳細な手順は「PYTHON_REINSTALL_GUIDE.md」を参照してください
echo.
echo ========================================
pause




