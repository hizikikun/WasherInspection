@echo off
REM HWiNFO自動再起動タスクをタスクスケジューラに登録するバッチファイル

echo ========================================
echo HWiNFO自動再起動タスクの登録
echo ========================================
echo.

REM 現在のディレクトリを取得
set "SCRIPT_DIR=%~dp0"
set "PYTHON_PATH=C:\Users\tomoh\AppData\Local\Programs\Python\Python310\python.exe"
set "RESTART_SCRIPT=%SCRIPT_DIR%hwinfo_auto_restart.py"

REM タスク名
set "TASK_NAME=HWiNFO_AutoRestart"

REM 既存のタスクを削除（存在する場合）
schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1

REM 新しいタスクを作成（12時間ごとに実行）
REM ログオン時に開始し、12時間間隔で繰り返し実行
schtasks /Create /TN "%TASK_NAME%" ^
    /TR "\"%PYTHON_PATH%\" \"%RESTART_SCRIPT%\" restart" ^
    /SC ONLOGON ^
    /RL HIGHEST ^
    /RU "%USERNAME%" ^
    /F /IT

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] タスクスケジューラへの登録に失敗しました
    echo.
    echo 以下のコマンドを手動で実行してください:
    echo schtasks /Create /TN "%TASK_NAME%" ^
    echo     /TR "\"%PYTHON_PATH%\" \"%RESTART_SCRIPT%\" restart" ^
    echo     /SC ONLOGON ^
    echo     /RL HIGHEST ^
    echo     /RU "%USERNAME%" ^
    echo     /F /IT
    echo.
    echo 12時間間隔での繰り返し設定は、タスクスケジューラのGUIから:
    echo 1. タスクスケジューラを開く
    echo 2. "%TASK_NAME%" タスクを選択
    echo 3. 「トリガー」タブで「編集」をクリック
    echo 4. 「繰り返し間隔」を「12時間」に設定
    echo 5. 「継続時間」を「無期限」に設定
    echo.
    exit /b 1
) else (
    echo [OK] タスクスケジューラへの登録が完了しました
    echo.
    echo 注意: 12時間間隔での繰り返し設定は、タスクスケジューラのGUIから設定してください:
    echo 1. タスクスケジューラを開く
    echo 2. "%TASK_NAME%" タスクを選択
    echo 3. 「トリガー」タブで「編集」をクリック
    echo 4. 「繰り返し間隔」を「12時間」に設定
    echo 5. 「継続時間」を「無期限」に設定
    echo.
)

pause

