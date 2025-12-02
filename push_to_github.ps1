# UTF-8 encoding setup
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GitHubへのPush" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "現在のブランチを確認中..." -ForegroundColor Yellow
$currentBranch = git branch --show-current
if (-not $currentBranch) {
    $currentBranch = "master"
}

Write-Host "現在のブランチ: $currentBranch" -ForegroundColor Green
Write-Host ""

Write-Host "リモートブランチを確認中..." -ForegroundColor Yellow
git branch -r
Write-Host ""

Write-Host "プッシュを実行します..." -ForegroundColor Yellow
Write-Host ""

# 初めてpushする場合は -u オプションを使用
Write-Host "git push -u origin $currentBranch" -ForegroundColor Cyan
git push -u origin $currentBranch

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  プッシュ成功！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  プッシュ失敗。別のブランチ名を試します..." -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    
    # masterブランチの場合、mainを試す
    if ($currentBranch -eq "master") {
        Write-Host "mainブランチで試します..." -ForegroundColor Yellow
        git push -u origin main
    } elseif ($currentBranch -eq "main") {
        # mainブランチの場合、masterを試す
        Write-Host "masterブランチで試します..." -ForegroundColor Yellow
        git push -u origin master
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "プッシュ成功！" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "プッシュに失敗しました。" -ForegroundColor Red
        Write-Host ""
        Write-Host "手動で実行する場合:" -ForegroundColor Yellow
        Write-Host "  git push -u origin $currentBranch" -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host "何かキーを押して終了してください..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

