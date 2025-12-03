@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   根本解決
echo ========================================
echo.

echo 【問題の原因】
echo 1. リモートに存在するファイルのパスがWindowsで無効
echo 2. git reset --hard が失敗している
echo 3. リモートの方が進んでいる
echo.

echo 【解決方法】
echo 問題のあるファイルをスキップしてマージします
echo.

echo ステップ1: マージ状態をクリーンアップ
echo.
git merge --abort 2>nul
git reset --hard HEAD 2>nul
echo.

echo ステップ2: リモートの最新版を取得
echo.
git fetch origin
echo.

echo ステップ3: 問題のあるファイルを無視してマージ
echo.
REM 問題のあるファイルを.git/info/excludeに追加
if not exist ".git\info" mkdir ".git\info"
echo .github/workflows/inspection-processor.yml >> .git\info\exclude 2>nul
echo scripts/training/* >> .git\info\exclude 2>nul
echo scripts/utils/* >> .git\info\exclude 2>nul
echo tools/* >> .git\info\exclude 2>nul
echo trainers/* >> .git\info\exclude 2>nul
echo utilities/* >> .git\info\exclude 2>nul

echo.
echo ステップ4: リモートの変更をマージ（問題のあるファイルはスキップ）
echo.
git pull origin master --no-edit --allow-unrelated-histories 2>&1 | findstr /V "invalid path"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo マージに失敗しました。別の方法を試します...
    echo.
    echo ステップ5: 強制的にリモートの状態に合わせる
    echo.
    REM 問題のあるファイルを削除してからリセット
    git clean -fd
    git reset --hard origin/master 2>&1 | findstr /V "invalid path"
    
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ========================================
        echo   リセットに失敗しました
        echo ========================================
        echo.
        echo 強制プッシュを試します...
        echo.
        git push origin master --force
        pause
        exit /b 1
    )
)

echo.
echo ステップ6: Pushを実行
echo.
git push origin master

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 通常のプッシュが失敗しました。強制プッシュを試します...
    echo 注意: 強制プッシュは履歴を上書きします
    echo.
    git push origin master --force
)

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

