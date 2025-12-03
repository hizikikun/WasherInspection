@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   簡単解決
echo ========================================
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
) else (
    echo.
    echo ========================================
    echo   プッシュに失敗しました
    echo ========================================
)

echo.
pause

