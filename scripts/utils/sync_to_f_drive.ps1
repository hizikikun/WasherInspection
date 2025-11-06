# Sync project files to F drive (incremental update)
# Update only changed files to F drive

param(
    [string]$TargetDrive = "F:",
    [string]$TargetFolder = "WasherInspection"
)

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Syncing project to F drive" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get current project path
$CurrentPath = $PSScriptRoot
if (-not $CurrentPath) {
    $CurrentPath = Get-Location
}

# Build target path
$TargetPath = Join-Path $TargetDrive $TargetFolder

Write-Host "Source: $CurrentPath" -ForegroundColor Yellow
Write-Host "Target: $TargetPath" -ForegroundColor Yellow
Write-Host ""

# Check if F drive exists
if (-not (Test-Path $TargetDrive)) {
    Write-Host "Error: F drive not found." -ForegroundColor Red
    exit 1
}

# Create target if it doesn't exist
if (-not (Test-Path $TargetPath)) {
    Write-Host "Creating target folder..." -ForegroundColor Green
    New-Item -ItemType Directory -Path $TargetPath -Force | Out-Null
}

Write-Host "Syncing files (this may take a while)..." -ForegroundColor Green

# Use robocopy to sync files (only copy newer files)
$robocopyArgs = @(
    $CurrentPath,
    $TargetPath,
    "/E",           # Include subdirectories
    "/XO",          # Exclude older files
    "/XN",          # Exclude newer files (keep newer at destination)
    "/NP",          # No progress
    "/NDL",         # No directory list
    "/NFL",         # No file list
    "/NJH",         # No job header
    "/NJS"          # No job summary
)

$robocopyResult = Start-Process -FilePath "robocopy" -ArgumentList $robocopyArgs -Wait -PassThru -NoNewWindow

# Robocopy exit codes: 0-7 = success, 8+ = errors
if ($robocopyResult.ExitCode -le 7) {
    Write-Host "Sync completed!" -ForegroundColor Green
} else {
    Write-Host "Warning: Some files may not have been copied. Exit code: $($robocopyResult.ExitCode)" -ForegroundColor Yellow
}

# Create/update path configuration file
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
    last_sync = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
} | ConvertTo-Json -Depth 10

$config | Out-File -FilePath $configPath -Encoding UTF8

Write-Host ""
Write-Host "Path configuration updated: $configPath" -ForegroundColor Green

# Update paths in F drive folder
Write-Host ""
Write-Host "Updating paths in F drive folder..." -ForegroundColor Green
$updateScript = Join-Path $TargetPath "update_paths.ps1"
if (Test-Path $updateScript) {
    Push-Location $TargetPath
    try {
        & $updateScript -ProjectPath $TargetPath
    } catch {
        Write-Host "Warning: Path update script failed: $_" -ForegroundColor Yellow
    }
    Pop-Location
} else {
    Write-Host "Warning: update_paths.ps1 not found in target folder" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Sync completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project is now available at: $TargetPath" -ForegroundColor Yellow
Write-Host ""





