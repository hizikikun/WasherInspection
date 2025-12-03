@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   最終マージ解決（安全版）
echo ========================================
echo.

echo 【問題点】
echo リモートに存在するがローカルに存在しないファイルが原因でマージが失敗しています
echo.

echo ステップ1: マージ状態をクリーンアップ
echo.
git merge --abort 2>nul
git reset --hard HEAD
echo.

echo ステップ2: リモートの最新版を取得
echo.
git fetch origin
echo.

echo ステップ3: リモートの変更を強制的にマージ
echo ローカルの変更は破棄されます
echo.
echo 注意: ローカルの変更は失われます
echo 続行しますか？ Y または N を入力してください
set /p CONFIRM=
if /i not "%CONFIRM%"=="Y" (
    echo 中止しました
    pause
    exit /b 0
)

echo.
echo リモートの状態に合わせます...
git reset --hard origin/master
echo.

echo ステップ4: Pushを実行
echo.
git push origin master

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo   成功！
    echo ========================================
) else (
    echo.
    echo ========================================
    echo   プッシュに失敗しました
    echo ========================================
    echo.
    echo 手動で実行する場合:
    echo   git push origin master
)

echo.
pause

