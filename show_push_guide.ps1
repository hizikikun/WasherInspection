# UTF-8 encoding setup
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GitHubへのPush手順ガイド" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "【基本的なPush手順】" -ForegroundColor Yellow
Write-Host ""
Write-Host "ステップ1: 現在の状態を確認" -ForegroundColor Green
Write-Host "  git status"
Write-Host ""
Write-Host "ステップ2: 変更をステージング（追加）" -ForegroundColor Green
Write-Host "  git add ."
Write-Host "  または"
Write-Host "  git add ファイル名"
Write-Host ""
Write-Host "ステップ3: コミット（変更を記録）" -ForegroundColor Green
Write-Host "  git commit -m `"変更内容の説明`""
Write-Host ""
Write-Host "ステップ4: GitHubにPush（アップロード）" -ForegroundColor Green
Write-Host "  git push origin main"
Write-Host "  または"
Write-Host "  git push origin master"
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "【コマンド一覧（まとめ）】" -ForegroundColor Yellow
Write-Host ""
Write-Host "  git status"
Write-Host "  git add ."
Write-Host "  git commit -m `"変更内容`""
Write-Host "  git push origin main"
Write-Host ""
Write-Host "【すべて一度に実行】" -ForegroundColor Yellow
Write-Host "  git add . && git commit -m `"更新`" && git push origin main"
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "【よくあるエラーと解決方法】" -ForegroundColor Yellow
Write-Host ""
Write-Host "エラー: `"Updates were rejected`"" -ForegroundColor Red
Write-Host "  解決: git pull origin main を実行してから再度push"
Write-Host ""
Write-Host "エラー: `"Authentication failed`"" -ForegroundColor Red
Write-Host "  解決: Personal Access Tokenを使用して認証"
Write-Host ""
Write-Host "エラー: `"Branch not found`"" -ForegroundColor Red
Write-Host "  解決: git branch でブランチ名を確認してからpush"
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "【簡単な方法】" -ForegroundColor Yellow
Write-Host "  GUIアプリを使用: python change_history_viewer.py"
Write-Host "  その後「コミット&プッシュ」ボタンをクリック"
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "詳細は GITHUB_PUSH_GUIDE.html をブラウザで開いてください" -ForegroundColor Cyan
Write-Host ""

# HTMLファイルを開くかどうか確認
$openHtml = Read-Host "HTMLファイルをブラウザで開きますか？ (Y/N)"
if ($openHtml -eq "Y" -or $openHtml -eq "y") {
    if (Test-Path "GITHUB_PUSH_GUIDE.html") {
        Start-Process "GITHUB_PUSH_GUIDE.html"
        Write-Host "ブラウザで開きました" -ForegroundColor Green
    } else {
        Write-Host "HTMLファイルが見つかりません" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "何かキーを押して終了してください..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

