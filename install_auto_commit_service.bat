@echo off
chcp 65001 >nul
echo ========================================
echo Auto-Commit Service Installer
echo ========================================
echo.
echo This will add auto-commit to Windows startup.
echo The script will run automatically when Windows starts.
echo.
pause

REM Get the script directory
set "SCRIPT_DIR=%~dp0"
set "BATCH_FILE=%SCRIPT_DIR%start_auto_commit_background.bat"

REM Get current user's startup folder
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"

REM Create shortcut in startup folder
echo Creating startup shortcut...
powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%STARTUP%\Auto-Commit WasherInspection.lnk'); $Shortcut.TargetPath = '%BATCH_FILE%'; $Shortcut.WorkingDirectory = '%SCRIPT_DIR%'; $Shortcut.WindowStyle = 1; $Shortcut.Save()"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Installation completed successfully!
    echo ========================================
    echo.
    echo Auto-commit will start automatically on next Windows startup.
    echo To start it now, run: start_auto_commit_background.bat
    echo.
    echo To uninstall, delete the shortcut from:
    echo %STARTUP%
) else (
    echo.
    echo ========================================
    echo Installation failed!
    echo ========================================
    echo Please run as administrator.
)

pause

