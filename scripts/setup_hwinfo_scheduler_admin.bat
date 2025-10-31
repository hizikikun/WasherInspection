@echo off
REM HWiNFO自動再起動タスクをタスクスケジューラに登録（管理者権限版）

echo ========================================
echo HWiNFO Auto Restart Task Setup
echo ========================================
echo.
echo This script requires administrator privileges.
echo Please run as administrator.
echo.
pause

REM Check for administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Administrator privileges required.
    echo Please right-click and select "Run as administrator"
    pause
    exit /b 1
)

set "TASK_NAME=HWiNFO_AutoRestart"
set "PYTHON_PATH=C:\Users\tomoh\AppData\Local\Programs\Python\Python310\python.exe"
set "SCRIPT_PATH=%~dp0hwinfo_auto_restart.py"
set "WORK_DIR=%~dp0"

REM Delete existing task if exists
schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1

REM Create new task (logon trigger, will need manual repetition setup)
schtasks /Create /TN "%TASK_NAME%" ^
    /TR "\"%PYTHON_PATH%\" \"%SCRIPT_PATH%\" restart" ^
    /SC ONLOGON ^
    /RL HIGHEST ^
    /RU "%USERNAME%" ^
    /F /IT

if %ERRORLEVEL% EQU 0 (
    echo [OK] Task created successfully
    echo.
    echo IMPORTANT: Please configure repetition interval manually:
    echo 1. Open Task Scheduler
    echo 2. Find task: %TASK_NAME%
    echo 3. Go to Triggers tab
    echo 4. Edit the trigger
    echo 5. Check "Repeat task every" and set to 12 hours
    echo 6. Set Duration to "Indefinitely"
    echo 7. Click OK
    echo.
) else (
    echo [ERROR] Failed to create task
    pause
    exit /b 1
)

pause

