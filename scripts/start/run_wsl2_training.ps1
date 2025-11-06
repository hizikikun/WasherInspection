# WSL2環境で学習スクリプトを実行（GPU使用）
# PowerShellスクリプト

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "WSL2 GPU学習実行" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$scriptPath = Join-Path $PSScriptRoot "run_wsl2_training.sh"
$scriptPathWsl = $scriptPath.Replace('\', '/').Replace('C:', '/mnt/c').Replace(':', '')

Write-Host "WSL2環境で学習を実行しています..." -ForegroundColor Yellow
Write-Host ""

# WSL2でスクリプトを実行
wsl bash "$scriptPathWsl" $args

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "完了" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan













