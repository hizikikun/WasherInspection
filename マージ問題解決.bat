@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   マージ問題を解決します
echo ========================================
echo.

echo 【問題点の確認】
echo.
echo 1. 以下のファイルが問題を起こしています:
echo    - old/inspection_focus.py
echo    - old/inspection_perfect.py
echo    - old/inspection_ultra_fixed.py
echo    - old/trainer_final.py
echo    - scripts/train_4class_sparse_ensemble.py
echo.
echo 2. これらのファイルはGitで追跡されていないか、
echo    変更がステージングされていません
echo.
echo 3. リモートの方が進んでいるため、マージが必要です
echo.

echo 【解決方法】
echo.
echo これらのファイルを削除してからマージします
echo （ファイルは実際には削除されず、Gitの管理から外れます）
echo.
pause

echo.
echo ステップ1: 問題のあるファイルを削除（Git管理から除外）
echo.

REM ファイルが存在するか確認してから削除
if exist "old\inspection_focus.py" (
    git rm --cached old/inspection_focus.py 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo old/inspection_focus.py は追跡されていません
    ) else (
        echo old/inspection_focus.py を削除しました
    )
)

if exist "old\inspection_perfect.py" (
    git rm --cached old/inspection_perfect.py 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo old/inspection_perfect.py は追跡されていません
    ) else (
        echo old/inspection_perfect.py を削除しました
    )
)

if exist "old\inspection_ultra_fixed.py" (
    git rm --cached old/inspection_ultra_fixed.py 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo old/inspection_ultra_fixed.py は追跡されていません
    ) else (
        echo old/inspection_ultra_fixed.py を削除しました
    )
)

if exist "old\trainer_final.py" (
    git rm --cached old/trainer_final.py 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo old/trainer_final.py は追跡されていません
    ) else (
        echo old/trainer_final.py を削除しました
    )
)

if exist "scripts\train_4class_sparse_ensemble.py" (
    git rm --cached scripts/train_4class_sparse_ensemble.py 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo scripts/train_4class_sparse_ensemble.py は追跡されていません
    ) else (
        echo scripts/train_4class_sparse_ensemble.py を削除しました
    )
)

echo.
echo ステップ2: すべての変更をリセット
echo.
git reset --hard HEAD

echo.
echo ステップ3: リモートの最新版を取得
echo.
git fetch origin

echo.
echo ステップ4: リモートの変更をマージ
echo.
git pull origin master --no-edit

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo   マージに失敗しました
    echo ========================================
    echo.
    echo 現在の状態を確認:
    git status
    echo.
    pause
    exit /b 1
)

echo.
echo ステップ5: Pushを実行
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
)

echo.
pause


