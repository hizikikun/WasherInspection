# F drive migration script (Auto mode - no confirmation)
# Move project to F drive for notebook PC compatibility

param(
    [string]$TargetDrive = "F:",
    [string]$TargetFolder = "WasherInspection"
)

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Moving project to F drive (Auto mode)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get current project path
$CurrentPath = $PSScriptRoot
if (-not $CurrentPath) {
    $CurrentPath = Get-Location
}

Write-Host "Current project path: $CurrentPath" -ForegroundColor Yellow

# Build target path
$TargetPath = Join-Path $TargetDrive $TargetFolder

Write-Host "Target path: $TargetPath" -ForegroundColor Yellow
Write-Host ""

# Check if F drive exists
if (-not (Test-Path $TargetDrive)) {
    Write-Host "Error: F drive not found." -ForegroundColor Red
    exit 1
}

# Check if target folder already exists and remove it
if (Test-Path $TargetPath) {
    Write-Host "Warning: $TargetPath already exists. Removing..." -ForegroundColor Yellow
    try {
        Remove-Item -Path $TargetPath -Recurse -Force -ErrorAction Stop
        Start-Sleep -Seconds 2
        Write-Host "Removed existing folder." -ForegroundColor Green
    } catch {
        Write-Host "Warning: Could not remove existing folder. Trying to copy anyway..." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Copying project to F drive..." -ForegroundColor Green
Write-Host "This may take several minutes..." -ForegroundColor Yellow

# Copy project using Copy-Item
try {
    Copy-Item -Path $CurrentPath -Destination $TargetPath -Recurse -Force -ErrorAction Stop
    Write-Host "Copy completed!" -ForegroundColor Green
} catch {
    Write-Host "Error: Copy failed. Error: $_" -ForegroundColor Red
    exit 1
}

# Create path configuration file
$configPath = Join-Path $TargetPath "config" "project_path.json"
$configDir = Split-Path $configPath -Parent
if (-not (Test-Path $configDir)) {
    New-Item -ItemType Directory -Path $configDir -Force | Out-Null
}

$wslPath = $TargetPath.Replace('\', '/')
if ($wslPath -match '^([A-Z]):') {
    $driveLetter = $matches[1].ToLower()
    $wslPath = $wslPath.Replace("$($matches[1]):", "/mnt/$driveLetter")
}

$config = @{
    project_root = $TargetPath
    project_root_wsl = $wslPath
    original_path = $CurrentPath
    moved_date = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
} | ConvertTo-Json -Depth 10

$config | Out-File -FilePath $configPath -Encoding UTF8

Write-Host ""
Write-Host "Path configuration file created: $configPath" -ForegroundColor Green

# Run path update script
Write-Host ""
Write-Host "Updating paths..." -ForegroundColor Green
$updateScript = Join-Path $TargetPath "update_paths.ps1"
if (Test-Path $updateScript) {
    Push-Location $TargetPath
    try {
        & $updateScript -ProjectPath $TargetPath
    } catch {
        Write-Host "Warning: Path update script failed: $_" -ForegroundColor Yellow
    }
    Pop-Location
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Move completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "New project path: $TargetPath" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Navigate to $TargetPath and start the app" -ForegroundColor White
Write-Host "2. If using WSL2, run setup_wsl2_tensorflow_gpu.sh again" -ForegroundColor White
Write-Host ""
