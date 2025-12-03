@echo off
chcp 65001 >nul
cls
echo.
echo ========================================
echo   手動実行用コマンド（1つずつ実行）
echo ========================================
echo.
echo 【ステップ1: 問題のあるファイルをGit管理から除外】
echo 以下のコマンドを1つずつコピー&ペーストしてEnterを押してください。
echo.
echo git rm --cached old/inspection_focus.py
echo.
echo git rm --cached old/inspection_perfect.py
echo.
echo git rm --cached old/inspection_ultra_fixed.py
echo.
echo git rm --cached old/trainer_final.py
echo.
echo git rm --cached scripts/train_4class_sparse_ensemble.py
echo.
echo ========================================
echo.
echo 【ステップ2: すべての変更をリセット】
echo.
echo git reset --hard HEAD
echo.
echo ========================================
echo.
echo 【ステップ3: リモートの最新版を取得】
echo.
echo git fetch origin
echo.
echo ========================================
echo.
echo 【ステップ4: リモートの変更をマージ】
echo.
echo git pull origin master
echo.
echo ========================================
echo.
echo 【ステップ5: Pushを実行】
echo.
echo git push origin master
echo.
echo ========================================
echo.
echo 【使い方】
echo 1. このウィンドウからコマンドを1つずつコピー
echo 2. コマンドプロンプトに貼り付け
echo 3. Enterを押す
echo 4. 次のコマンドに進む
echo.
echo ========================================
echo.
pause


