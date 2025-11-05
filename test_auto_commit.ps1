# UTF-8 encoding setup
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Move to script directory
$here = $PSScriptRoot
if (-not $here) { $here = Split-Path -Parent $PSCommandPath }
if (-not $here) { $here = Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location -Path $here

Write-Host "Auto-commit script started. Watching for changes..."
Write-Host "Press Ctrl+C to stop"
Write-Host ""

$QuietSeconds = 60
$lastCheck = Get-Date

while ($true) {
    Start-Sleep -Seconds 10
    
    $gitStatus = git status --porcelain 2>&1
    if ($gitStatus) {
        $now = Get-Date
        $timeSinceLastCheck = ($now - $lastCheck).TotalSeconds
        
        if ($timeSinceLastCheck -ge $QuietSeconds) {
            $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            Write-Host "$timestamp - Changes detected, committing..."
            
            git add -A
            $commitMsg = "Auto-commit: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
            git commit -m $commitMsg
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "$timestamp - Commit successful, pushing..."
                git push origin master
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "$timestamp - Push successful!"
                } else {
                    Write-Host "$timestamp - Push failed"
                }
            } else {
                Write-Host "$timestamp - Commit failed"
            }
            
            $lastCheck = Get-Date
        } else {
            $remaining = $QuietSeconds - $timeSinceLastCheck
            Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Waiting... ($([math]::Round($remaining))s remaining)"
        }
    }
}
