# 日本語文字エンコーディングテスト
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "=== 文字エンコーディングテスト ==="
Write-Host "現在の時刻: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "日本語文字のテスト: こんにちは、世界！"
Write-Host "GitHub自動転送システムのテスト"
Write-Host "================================"

# ログファイルに書き込みテスト
$logFile = "japanese-test.log"
$testMsg = "テストメッセージ: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - 日本語文字テスト"
Write-Host $testMsg
$testMsg | Out-File -FilePath $logFile -Append -Encoding UTF8

Write-Host "ログファイルに書き込みました: $logFile"



