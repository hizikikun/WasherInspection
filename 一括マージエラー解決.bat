@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   一括マージエラー解決
echo ========================================
echo.

echo 【問題点】
echo 大量のファイルがマージで上書きされる可能性があります
echo これらのファイルを一時的に退避してからマージします
echo.
pause

echo.
echo ステップ1: すべての変更をリセット
echo.
git reset --hard HEAD

echo.
echo ステップ2: 追跡されていないファイルを一時的に退避
echo.
REM 一時ディレクトリを作成
if not exist "temp_backup" mkdir temp_backup

REM 問題のあるファイルを一時的に移動
for %%f in (
    ".github\workflows\inspection-processor.yml"
    "docs\CODE_TRAINING_SYNC_GUIDE.md"
    "docs\FILE_NAMING.md"
    "docs\GITHUB_TOOLS.md"
    "docs\PROJECT_STRUCTURE.md"
    "docs\README.md"
    "github_tools\auto_sync.py"
    "github_tools\cursor_integration.py"
    "github_tools\github_autocommit.py"
    "github_tools\github_sync.py"
    "github_tools\token_setup.py"
    "old\camera2_inspection.py"
    "old\camera_simple.py"
    "old\github_unused\cursor_extension.py"
    "old\github_unused\github_integration.py"
    "old\github_unused\sync.py"
    "old\inspection_fixed.py"
    "old\inspection_focus.py"
    "old\inspection_perfect.py"
    "old\inspection_ultra_fixed.py"
    "old\trainer_final.py"
    "scripts\train_4class_sparse_ensemble.py"
) do (
    if exist "%%f" (
        echo 移動中: %%f
        move "%%f" "temp_backup\" >nul 2>&1
    )
)

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
    echo 退避したファイルを復元します...
    if exist "temp_backup" (
        xcopy /E /I /Y "temp_backup\*" . >nul 2>&1
    )
    pause
    exit /b 1
)

echo.
echo ステップ5: 退避したファイルを復元
echo.
if exist "temp_backup" (
    xcopy /E /I /Y "temp_backup\*" . >nul 2>&1
    echo ファイルを復元しました
)

echo.
echo ステップ6: Pushを実行
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
echo 一時バックアップフォルダを削除しますか？ (Y/N)
set /p DELETE_BACKUP=
if /i "%DELETE_BACKUP%"=="Y" (
    if exist "temp_backup" rmdir /S /Q "temp_backup"
    echo 削除しました
)

echo.
pause

