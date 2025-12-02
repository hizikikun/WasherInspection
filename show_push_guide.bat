@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   GitHubへのPush手順ガイド
echo ========================================
echo.

echo 【基本的なPush手順】
echo.
echo ステップ1: 現在の状態を確認
echo   git status
echo.
echo ステップ2: 変更をステージング（追加）
echo   git add .
echo   または
echo   git add ファイル名
echo.
echo ステップ3: コミット（変更を記録）
echo   git commit -m "変更内容の説明"
echo.
echo ステップ4: GitHubにPush（アップロード）
echo   git push origin main
echo   または
echo   git push origin master
echo.
echo ========================================
echo.
echo 【コマンド一覧（まとめ）】
echo.
echo   git status
echo   git add .
echo   git commit -m "変更内容"
echo   git push origin main
echo.
echo 【すべて一度に実行】
echo   git add . ^&^& git commit -m "更新" ^&^& git push origin main
echo.
echo ========================================
echo.
echo 【よくあるエラーと解決方法】
echo.
echo エラー: "Updates were rejected"
echo   解決: git pull origin main を実行してから再度push
echo.
echo エラー: "Authentication failed"
echo   解決: Personal Access Tokenを使用して認証
echo.
echo エラー: "Branch not found"
echo   解決: git branch でブランチ名を確認してからpush
echo.
echo ========================================
echo.
echo 【簡単な方法】
echo   GUIアプリを使用: python change_history_viewer.py
echo   その後「コミット^&プッシュ」ボタンをクリック
echo.
echo ========================================
echo.
echo 詳細は GITHUB_PUSH_GUIDE.html をブラウザで開いてください
echo.
pause

