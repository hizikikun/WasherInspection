@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   GitHubへのPush
echo ========================================
echo.

echo 現在のブランチを確認中...
git branch --show-current > temp_branch.txt
set /p CURRENT_BRANCH=<temp_branch.txt
del temp_branch.txt

echo 現在のブランチ: %CURRENT_BRANCH%
echo.

echo リモートブランチを確認中...
git branch -r
echo.

echo プッシュを実行します...
echo.

REM 初めてpushする場合は -u オプションを使用
git push -u origin %CURRENT_BRANCH%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo   プッシュ成功！
    echo ========================================
) else (
    echo.
    echo ========================================
    echo   プッシュ失敗。別のブランチ名を試します...
    echo ========================================
    echo.
    
    REM masterブランチの場合、mainを試す
    if "%CURRENT_BRANCH%"=="master" (
        echo mainブランチで試します...
        git push -u origin main
    ) else (
        REM mainブランチの場合、masterを試す
        if "%CURRENT_BRANCH%"=="main" (
            echo masterブランチで試します...
            git push -u origin master
        )
    )
)

echo.
pause

