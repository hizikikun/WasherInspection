@echo off
chcp 65001 >nul
cls
echo.
echo ========================================
echo   GitHubへのPush手順（手動実行）
echo ========================================
echo.
echo 【このウィンドウで実行するコマンド】
echo.
echo 1. 現在の状態を確認
echo    git status
echo.
echo 2. 現在のブランチを確認
echo    git branch
echo.
echo 3. リモートの情報を取得
echo    git fetch origin
echo.
echo 4. リモートの変更をマージ
echo    git pull origin master
echo    （ブランチがmainの場合は git pull origin main）
echo.
echo 5. Pushを実行
echo    git push origin master
echo    （ブランチがmainの場合は git push origin main）
echo.
echo ========================================
echo.
echo 【エラーが出た場合】
echo.
echo マージが失敗した場合:
echo   git merge --abort
echo   その後、競合を解決してから再度 git pull
echo.
echo プッシュが拒否された場合:
echo   再度 git pull を実行してから git push
echo.
echo ========================================
echo.
echo 上記のコマンドを順番に実行してください。
echo 各コマンドをコピー&ペーストしてEnterを押します。
echo.
pause


