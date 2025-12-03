@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   安全なマージ解決
echo ========================================
echo.

echo 【方法1: リモートの状態に合わせる（推奨）】
echo ローカルの変更は失われますが、確実にマージできます
echo.
echo 【方法2: 手動で競合を解決】
echo 時間がかかりますが、変更を保持できます
echo.
echo どちらを選択しますか？
echo 1 = リモートの状態に合わせる（簡単）
echo 2 = 手動で解決（時間がかかる）
set /p CHOICE=

if "%CHOICE%"=="1" (
    echo.
    echo リモートの状態に合わせます...
    git fetch origin
    git reset --hard origin/master
    echo.
    echo Pushを実行します...
    git push origin master
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ========================================
        echo   成功！
        echo ========================================
    )
) else if "%CHOICE%"=="2" (
    echo.
    echo 手動解決の手順:
    echo.
    echo 1. 問題のあるファイルを削除または移動
    echo 2. git pull origin master
    echo 3. 競合を解決
    echo 4. git push origin master
    echo.
    echo 詳細は「手動解決手順.txt」を参照してください
) else (
    echo 無効な選択です
)

echo.
pause


