@echo off
REM HWiNFO64を再起動するバッチファイル
REM 12時間制限を回避するために使用

REM HWiNFO64のプロセスを終了
for /F "usebackq tokens=2" %%a in (
    `tasklist /fi "IMAGENAME eq HWiNFO64.EXE" 2^>nul ^| findstr "[0-9]"`
) do (
    taskkill /f /PID %%a 2>nul
)

REM 少し待機（プロセス終了を確実にする）
timeout /t 2 /nobreak >nul

REM HWiNFO64のパス（複数の可能性をチェック）
set "HWINFO_PATH="

if exist "C:\Program Files\HWiNFO64\HWiNFO64.EXE" (
    set "HWINFO_PATH=C:\Program Files\HWiNFO64\HWiNFO64.EXE"
) else if exist "C:\Program Files (x86)\HWiNFO64\HWiNFO64.EXE" (
    set "HWINFO_PATH=C:\Program Files (x86)\HWiNFO64\HWiNFO64.EXE"
) else (
    REM 環境変数またはレジストリから取得を試みる（見つからない場合は手動設定が必要）
    echo HWiNFO64.exe が見つかりません。パスを手動で設定してください。
    exit /b 1
)

REM HWiNFO64を起動
start "" "%HWINFO_PATH%"

exit /b 0

