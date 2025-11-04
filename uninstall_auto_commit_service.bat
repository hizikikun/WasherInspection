@echo off
chcp 65001 >nul
echo ========================================
echo Auto-Commit Service Uninstaller
echo ========================================
echo.

set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT=%STARTUP%\Auto-Commit WasherInspection.lnk"

if exist "%SHORTCUT%" (
    del "%SHORTCUT%"
    echo Auto-commit startup shortcut removed.
    echo.
    echo Note: If the script is currently running,
    echo you need to close it manually from Task Manager.
) else (
    echo Shortcut not found. It may have been already removed.
)

pause

