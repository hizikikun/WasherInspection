@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   マージ状態を確認して修正
echo ========================================
echo.

echo 現在の状態を確認中...
git status
echo.

echo マージが進行中の場合は中止します...
git merge --abort 2>nul
if %ERRORLEVEL% EQU 0 (
    echo マージを中止しました。
    echo.
)

echo 作業ディレクトリをクリーンアップ中...
git reset --hard HEAD
echo.

echo 注意: git cleanは実行しません（バッチファイルを保護するため）
echo 不要なファイルがある場合は手動で削除してください
echo.

echo リモートの最新版を取得中...
git fetch origin
echo.

echo 現在のブランチを確認中...
git branch --show-current > temp_branch.txt
set /p CURRENT_BRANCH=<temp_branch.txt
del temp_branch.txt

echo 現在のブランチ: %CURRENT_BRANCH%
echo.

echo リモートとローカルの差分を確認中...
git log HEAD..origin/%CURRENT_BRANCH% --oneline >nul 2>&1
set HAS_REMOTE_CHANGES=%ERRORLEVEL%

if %HAS_REMOTE_CHANGES% EQU 0 (
    echo リモートに新しい変更があります。マージします...
    echo.
    git pull origin %CURRENT_BRANCH% --no-edit
    set PULL_RESULT=%ERRORLEVEL%
    
    REM マージの状態を確認
    git status | findstr /C:"Unmerged paths" >nul
    set HAS_CONFLICTS=%ERRORLEVEL%
    
    git status | findstr /C:"All conflicts fixed" >nul
    set CONFLICTS_FIXED=%ERRORLEVEL%
    
    if %PULL_RESULT% NEQ 0 (
        REM プルが失敗した
        echo.
        echo ========================================
        echo   マージに失敗しました。
        echo ========================================
        echo.
        echo 現在の状態:
        git status
        echo.
        echo マージを中止します...
        git merge --abort 2>nul
        echo.
        echo 手動で解決する場合:
        echo   1. git status で競合ファイルを確認
        echo   2. 競合を解決
        echo   3. git add .
        echo   4. git commit -m "マージ: 競合を解決"
        echo   5. このスクリプトを再度実行
        pause
        exit /b 1
    )
    
    if %HAS_CONFLICTS% EQU 0 (
        REM 競合がある
        echo.
        echo ========================================
        echo   競合が検出されました。
        echo   手動で解決してください。
        echo ========================================
        echo.
        echo 競合ファイル:
        git status
        echo.
        echo 手動で解決する場合:
        echo   1. 競合ファイルを編集して解決
        echo   2. git add .
        echo   3. git commit -m "マージ: 競合を解決"
        echo   4. このスクリプトを再度実行
        pause
        exit /b 1
    )
    
    if %CONFLICTS_FIXED% EQU 0 (
        REM 競合が解決されたが、まだコミットされていない
        echo 競合が解決されました。コミットを完了します...
        git commit --no-edit
    )
    
    echo マージ完了
    echo.
) else (
    echo リモートに新しい変更はありません。
    echo.
)

echo プッシュを実行します...
git push origin %CURRENT_BRANCH%
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo   プッシュ成功！
    echo ========================================
) else (
    echo.
    echo ========================================
    echo   プッシュ失敗
    echo ========================================
    echo.
    echo 手動で実行する場合:
    echo   git push origin %CURRENT_BRANCH%
    echo.
    echo または強制プッシュ（注意が必要）:
    echo   git push origin %CURRENT_BRANCH% --force
)

echo.
pause
