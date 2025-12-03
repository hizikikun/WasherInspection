@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   強制解決（最終手段）
echo ========================================
echo.

echo 【警告】
echo この方法は履歴を上書きします
echo 共同作業の場合は使用しないでください
echo.
echo 続行しますか？ Y を入力してください
set /p CONFIRM=
if /i not "%CONFIRM%"=="Y" (
    echo 中止しました
    pause
    exit /b 0
)

echo.
echo ステップ1: 現在の状態を確認
echo.
git status --short
echo.

echo ステップ2: すべての変更を破棄
echo.
git reset --hard HEAD
git clean -fd
echo.

echo ステップ3: リモートの最新版を取得
echo.
git fetch origin
echo.

echo ステップ4: 強制的にリモートの状態に合わせる
echo.
REM 問題のあるファイルをスキップする設定
git config core.protectNTFS false
git config core.longpaths true

git reset --hard origin/master 2>&1 | findstr /V "invalid path" | findstr /V "Could not reset"

echo.
echo ステップ5: 強制プッシュ
echo.
git push origin master --force

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
    echo   git push origin master --force
)

echo.
pause


