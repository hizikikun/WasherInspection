@echo off
REM ポータブル化設定の確認スクリプト

echo ========================================
echo ポータブル化設定の確認
echo ========================================
echo.

REM プロジェクトルートを取得
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo プロジェクトパス: %PROJECT_ROOT%
echo.

REM 設定ファイルを確認
if exist "config\project_path.json" (
    echo [OK] 設定ファイルが見つかりました: config\project_path.json
    type config\project_path.json
) else (
    echo [WARN] 設定ファイルが見つかりません
    echo パス更新スクリプトを実行してください: update_paths.ps1
)

echo.
echo ========================================
echo 確認完了
echo ========================================
pause





