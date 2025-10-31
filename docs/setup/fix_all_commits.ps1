# PowerShell script to fix all commit messages
# This script uses git rebase to fix commit messages

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Green
Write-Host "コミットメッセージ一括修正スクリプト" -ForegroundColor Green  
Write-Host "============================================================" -ForegroundColor Green

# Git filter-branch用のPythonスクリプトを作成
$filterScript = @"
import sys
import codecs

if sys.platform.startswith('win'):
    sys.stdin = codecs.getreader('utf-8')(sys.stdin.buffer)
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

msg = sys.stdin.read()
fixes = {
    '情ｮ報｣: PROJECT_STRUCTURE.mdのｮ荳ｻ隕√ヵを｡を､スｫ隱ｬ譏弱そをｯをｷスｧスｳを呈峩がｰ': '修正: PROJECT_STRUCTURE.mdの主要ファイル説明セクションを更新',
    '譖ｴがｰ: docs/PROJECT_STRUCTURE.mdのｮGitHubステテスｫ荳隕ｧをゆｿｮ報｣': '更新: docs/PROJECT_STRUCTURE.mdのGitHubドキュメントを更新',
    '譖ｴがｰ: ス峨くス･ス｡スｳス亥テのｮ蜿､のテヵを｡を､スｫ実行ｒ情ｮ報｣': '更新: プロジェクトファイル整理とリネーム',
    '譖ｴがｰ: .gitignoreのｫス舌ャをｯを｢ステテのｨ謨ｴ逅テせをｯスｪス励ヨを定ｿｽ出': '更新: .gitignoreにディレクトリとignoreファイルを追加',
    'fix: safe_git_commit.pyを竪itPythonを剃ｽｿ逕ｨの吶ｋを医≧のｫ情ｮ報｣': 'fix: safe_git_commit.pyをGitPythonを使用するように修正',
    'chore: Git UTF-8險ｭ螳壹ヤスｼスｫのｨをｳス溘ャス医Γステそスｼをｸ情ｮ報｣ステテスｫを定ｿｽ出': 'chore: Git UTF-8設定ツールとコミットメッセージ修正ツールを追加',
}

msg_stripped = msg.strip()
if msg_stripped in fixes:
    sys.stdout.write(fixes[msg_stripped] + '\n')
else:
    sys.stdout.write(msg)
"@

# 一時ファイルに書き込み
$tempScript = "$env:TEMP\git_msg_filter_$$.py"
$filterScript | Out-File -FilePath $tempScript -Encoding UTF8

try {
    Write-Host "`nフィルタースクリプトを作成しました: $tempScript" -ForegroundColor Yellow
    
    # Git filter-branchを実行
    $env:FILTER_BRANCH_SQUELCH_WARNING = "1"
    $env:PYTHONIOENCODING = "utf-8"
    
    Write-Host "`ngit filter-branchを実行していまム..." -ForegroundColor Yellow
    git filter-branch -f --msg-filter "python $tempScript" -- --all
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ コミットメッセージの修正が完了しました！" -ForegroundColor Green
        Write-Host "`n次のステップ:" -ForegroundColor Yellow
        Write-Host "1. git log で確認" -ForegroundColor Cyan
        Write-Host "2. git push --force-with-lease origin main" -ForegroundColor Cyan
    } else {
        Write-Host "`n✗ エラーが発生しました" -ForegroundColor Red
    }
} finally {
    # 一時ファイルを削除
    if (Test-Path $tempScript) {
        Remove-Item $tempScript -Force
    }
}


