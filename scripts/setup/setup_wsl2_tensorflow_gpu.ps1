# TensorFlow GPU完全サポート設定スクリプト（WSL2使用）
# Windows環境でWSL2を使ってTensorFlowの完全なCUDAサポートを有効化

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TensorFlow GPU完全サポート設定 (WSL2)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# WSL2がインストールされているか確認
try {
    $wslStatus = wsl --status 2>&1
    Write-Host "[OK] WSL2がインストールされています" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] WSL2がインストールされていません。" -ForegroundColor Red
    Write-Host "以下のコマンドでWSL2をインストールしてください:" -ForegroundColor Yellow
    Write-Host "  wsl --install" -ForegroundColor White
    exit 1
}

# スクリプトファイルのパスを取得
$scriptPath = Join-Path $PSScriptRoot "setup_wsl2_tensorflow_gpu.sh"
$scriptPathWsl = $scriptPath.Replace('\', '/').Replace('C:', '/mnt/c').Replace(':', '')

Write-Host "WSL2内でセットアップを実行しています..." -ForegroundColor Yellow
Write-Host "（時間がかかる場合があります）" -ForegroundColor Yellow
Write-Host ""

# WSL2でスクリプトを実行
wsl bash "$scriptPathWsl"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "セットアップ完了" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "WSL2環境でTensorFlow GPUサポートが有効になりました。" -ForegroundColor Green
Write-Host ""
Write-Host "使用方法:" -ForegroundColor Yellow
Write-Host "  wsl" -ForegroundColor White
Write-Host "  cd /mnt/c/Users/tomoh/WasherInspection" -ForegroundColor White
Write-Host "  source venv_wsl2/bin/activate" -ForegroundColor White
Write-Host "  python scripts/train_4class_sparse_ensemble.py" -ForegroundColor White
