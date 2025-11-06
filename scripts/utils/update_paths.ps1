# プロジェクト内のハードコードされたパスを更新するスクリプト

param(
    [string]$ProjectPath = $PSScriptRoot
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "プロジェクトパスを更新" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# プロジェクトパスを取得
if (-not $ProjectPath) {
    $ProjectPath = $PSScriptRoot
}

if (-not (Test-Path $ProjectPath)) {
    Write-Host "エラー: プロジェクトパスが見つかりません: $ProjectPath" -ForegroundColor Red
    exit 1
}

Write-Host "プロジェクトパス: $ProjectPath" -ForegroundColor Yellow

# WSLパスを計算（ドライブレターを自動検出）
$driveLetter = $ProjectPath.Substring(0, 1)
$wslPath = $ProjectPath.Replace('\', '/').Replace("$driveLetter`:", "/mnt/$($driveLetter.ToLower())")

Write-Host "WSLパス: $wslPath" -ForegroundColor Yellow
Write-Host ""

# 更新するファイルのリスト
$filesToUpdate = @(
    "dashboard\integrated_washer_app.py",
    "setup_wsl2_tensorflow_gpu.sh",
    "run_wsl2_training.sh",
    "check_wsl2_gpu.sh",
    "test_wsl2_gpu_detection.py"
)

$oldPath = "C:\\Users\\tomoh\\WasherInspection"
$oldPathWsl = "/mnt/c/Users/tomoh/WasherInspection"
$oldPathAlt = 'C:\Users\tomoh\WasherInspection'

$newPath = $ProjectPath
$newPathWsl = $wslPath
$newPathEscaped = $ProjectPath.Replace('\', '\\')

$updatedCount = 0

foreach ($file in $filesToUpdate) {
    $filePath = Join-Path $ProjectPath $file
    if (Test-Path $filePath) {
        Write-Host "更新中: $file" -ForegroundColor Yellow
        
        $content = Get-Content $filePath -Raw -Encoding UTF8
        $originalContent = $content
        
        # Windowsパスの置換
        $content = $content -replace [regex]::Escape($oldPath), $newPathEscaped
        $content = $content -replace [regex]::Escape($oldPathAlt), $newPath
        $content = $content -replace [regex]::Escape($oldPath.Replace('\\', '\')), $newPath
        
        # WSLパスの置換
        $content = $content -replace [regex]::Escape($oldPathWsl), $newPathWsl
        
        if ($content -ne $originalContent) {
            Set-Content -Path $filePath -Value $content -Encoding UTF8 -NoNewline
            Write-Host "  更新完了" -ForegroundColor Green
            $updatedCount++
        } else {
            Write-Host "  変更なし" -ForegroundColor Gray
        }
    }
}

# venv_wsl2内のパスも更新（再作成が必要な場合は警告）
$venvPath = Join-Path $ProjectPath "venv_wsl2"
if (Test-Path $venvPath) {
    Write-Host ""
    Write-Host "警告: venv_wsl2が存在します。" -ForegroundColor Yellow
    Write-Host "仮想環境にはハードコードされたパスが含まれている可能性があります。" -ForegroundColor Yellow
    Write-Host "WSL2環境で再作成することを推奨します:" -ForegroundColor Yellow
    Write-Host "  cd $wslPath" -ForegroundColor White
    Write-Host "  bash setup_wsl2_tensorflow_gpu.sh" -ForegroundColor White
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "更新完了: $updatedCount ファイルを更新しました" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""





