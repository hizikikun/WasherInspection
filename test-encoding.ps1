# Encoding test script
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "=== Encoding Test ==="
Write-Host "Current time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Japanese text test: Hello World!"
Write-Host "GitHub auto-commit system test"
Write-Host "================================"

# Log file write test
$logFile = "encoding-test.log"
$testMsg = "Test message: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Japanese text test"
Write-Host $testMsg
$testMsg | Out-File -FilePath $logFile -Append -Encoding UTF8

Write-Host "Written to log file: $logFile"