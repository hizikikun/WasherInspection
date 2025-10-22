param(
  [int]$QuietSeconds = 120,
  [int]$LargeDiffLines = 200,
  [int]$LargeDiffFiles = 5
)

# スクリプトの場所へ移動（どこから起動しても .git が見つかるように）
$here = $PSScriptRoot
if (-not $here) { $here = Split-Path -Parent $PSCommandPath }
if (-not $here) { $here = Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location -Path $here


$ErrorActionPreference = "Stop"
$root = (Get-Location).Path
Write-Host "Watching $root ..."

$fsw = New-Object System.IO.FileSystemWatcher $root, "*"
$fsw.IncludeSubdirectories = $true
$fsw.EnableRaisingEvents = $true

$changed = $false
$last = Get-Date

$onChange = {
  $script:changed = $true
  $script:last = Get-Date
}
$handlers = @()
$handlers += Register-ObjectEvent $fsw Changed -Action $onChange
$handlers += Register-ObjectEvent $fsw Created -Action $onChange
$handlers += Register-ObjectEvent $fsw Deleted -Action $onChange
$handlers += Register-ObjectEvent $fsw Renamed -Action $onChange

function Get-DiffStats {
  git add -A | Out-Null
  $numstat = git diff --cached --numstat
  $files = @()
  $added = 0; $deleted = 0
  foreach ($line in $numstat) {
    $parts = $line -split "`t"
    if ($parts.Length -ge 3) {
      if ($parts[0] -match "^\d+$") { $added += [int]$parts[0] }
      if ($parts[1] -match "^\d+$") { $deleted += [int]$parts[1] }
      $files += $parts[2]
    }
  }
  [pscustomobject]@{
    Files = $files
    NumFiles = $files.Count
    LinesChanged = $added + $deleted
  }
}

function New-CommitMessage([pscustomobject]$stats, [bool]$isLarge) {
  $type = if ($isLarge) { "feat" } else { "chore" }
  $scope = ""
  if ($stats.NumFiles -gt 0) {
    $top = ($stats.Files | Select-Object -First 1)
    $scope = "(" + ($top -replace "[\\/].*$","") + ")"
  }
  $summary = ("{0}{1}: auto update - {2} files, {3} lines" -f $type, $scope, $stats.NumFiles, $stats.LinesChanged)
  $body = @(
    "Changed files:",
    ($stats.Files | ForEach-Object { "- $_" })
  ) -join "`n"
  if ($isLarge) {
    $body += "`n`nBREAKING CHANGE: Large update auto-PR"
  }
  @"
$summary

$body
"@
}

while ($true) {
  Start-Sleep -Seconds 5
  if (-not $changed) { continue }
  if ((Get-Date) -lt $last.AddSeconds($QuietSeconds)) { continue }

  $changed = $false
  if (-not (git status --porcelain)) { continue }

  $stats = Get-DiffStats
  if ($stats.NumFiles -eq 0) { git reset | Out-Null; continue }

  $isLarge = ($stats.LinesChanged -ge $LargeDiffLines) -or ($stats.NumFiles -ge $LargeDiffFiles)
  $msg = New-CommitMessage $stats $isLarge

  git commit -m $msg | Out-Null

  if ($isLarge) {
    $branch = "auto/" + (Get-Date -Format "yyyyMMdd-HHmmss")
    git switch -c $branch | Out-Null
    git push -u origin $branch
    try {
      $title = ("{0}: large auto update" -f $branch)
      gh pr create --fill --title $title --body "Auto-generated PR for large change."
    } catch { Write-Warning $_ }
  } else {
    git push
  }

  if ($isLarge) {
    git switch - | Out-Null
  }
}
