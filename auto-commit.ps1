param(
  [int]$QuietSeconds = 60,
  [int]$LargeDiffLines = 100,
  [int]$LargeDiffFiles = 3
)

# スクリプトの場所へ移動（どこから起動しても .git が見つかるように）
$here = $PSScriptRoot
if (-not $here) { $here = Split-Path -Parent $PSCommandPath }
if (-not $here) { $here = Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location -Path $here

# ログファイルの設定
$logFile = "auto-commit.log"
$ErrorActionPreference = "Continue"
$root = (Get-Location).Path
Write-Host "Watching $root ..." | Tee-Object -FilePath $logFile -Append

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
  try {
    git add -A 2>&1 | Out-Null
    $numstat = git diff --cached --numstat 2>&1
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
  } catch {
    Write-Warning "Git操作でエラーが発生しました: $_" | Tee-Object -FilePath $logFile -Append
    return [pscustomobject]@{
      Files = @()
      NumFiles = 0
      LinesChanged = 0
    }
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
  Start-Sleep -Seconds 10
  if (-not $changed) { continue }
  if ((Get-Date) -lt $last.AddSeconds($QuietSeconds)) { continue }

  $changed = $false
  $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  
  try {
    $gitStatus = git status --porcelain 2>&1
    if (-not $gitStatus) { 
      Write-Host "$timestamp - 変更なし" | Tee-Object -FilePath $logFile -Append
      continue 
    }

    Write-Host "$timestamp - 変更を検出しました" | Tee-Object -FilePath $logFile -Append
    $stats = Get-DiffStats
    if ($stats.NumFiles -eq 0) { 
      git reset 2>&1 | Out-Null
      Write-Host "$timestamp - ステージングをリセットしました" | Tee-Object -FilePath $logFile -Append
      continue 
    }

    $isLarge = ($stats.LinesChanged -ge $LargeDiffLines) -or ($stats.NumFiles -ge $LargeDiffFiles)
    $msg = New-CommitMessage $stats $isLarge

    Write-Host "$timestamp - コミット実行中: $($stats.NumFiles)ファイル, $($stats.LinesChanged)行" | Tee-Object -FilePath $logFile -Append
    $commitResult = git commit -m $msg 2>&1
    if ($LASTEXITCODE -eq 0) {
      Write-Host "$timestamp - コミット成功" | Tee-Object -FilePath $logFile -Append
    } else {
      Write-Warning "$timestamp - コミット失敗: $commitResult" | Tee-Object -FilePath $logFile -Append
      continue
    }

    if ($isLarge) {
      $branch = "auto/" + (Get-Date -Format "yyyyMMdd-HHmmss")
      Write-Host "$timestamp - 大きな変更を検出、ブランチ作成: $branch" | Tee-Object -FilePath $logFile -Append
      git switch -c $branch 2>&1 | Out-Null
      $pushResult = git push -u origin $branch 2>&1
      if ($LASTEXITCODE -eq 0) {
        Write-Host "$timestamp - ブランチプッシュ成功" | Tee-Object -FilePath $logFile -Append
        try {
          $title = ("{0}: large auto update" -f $branch)
          $prResult = gh pr create --fill --title $title --body "Auto-generated PR for large change." 2>&1
          if ($LASTEXITCODE -eq 0) {
            Write-Host "$timestamp - PR作成成功" | Tee-Object -FilePath $logFile -Append
          } else {
            Write-Warning "$timestamp - PR作成失敗: $prResult" | Tee-Object -FilePath $logFile -Append
          }
        } catch { 
          Write-Warning "$timestamp - PR作成エラー: $_" | Tee-Object -FilePath $logFile -Append
        }
      } else {
        Write-Warning "$timestamp - ブランチプッシュ失敗: $pushResult" | Tee-Object -FilePath $logFile -Append
      }
      git switch - 2>&1 | Out-Null
    } else {
      $pushResult = git push 2>&1
      if ($LASTEXITCODE -eq 0) {
        Write-Host "$timestamp - プッシュ成功" | Tee-Object -FilePath $logFile -Append
      } else {
        Write-Warning "$timestamp - プッシュ失敗: $pushResult" | Tee-Object -FilePath $logFile -Append
      }
    }
  } catch {
    Write-Warning "$timestamp - エラーが発生しました: $_" | Tee-Object -FilePath $logFile -Append
  }
}
