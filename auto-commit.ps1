# UTF-8 encoding setup
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Parameters
$QuietSeconds = 60
$LargeDiffLines = 100
$LargeDiffFiles = 3

# Move to script directory
$here = $PSScriptRoot
if (-not $here) { $here = Split-Path -Parent $PSCommandPath }
if (-not $here) { $here = Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location -Path $here

# Log file setup
$logFile = "auto-commit.log"
$ErrorActionPreference = "Continue"
$root = (Get-Location).Path
Write-Host "Watching $root ..." | Out-File -FilePath $logFile -Append -Encoding UTF8

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
    $errorMsg = "Git operation error: $_"
    Write-Warning $errorMsg
    $errorMsg | Out-File -FilePath $logFile -Append -Encoding UTF8
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
      $msg = "$timestamp - No changes"
      Write-Host $msg
      $msg | Out-File -FilePath $logFile -Append -Encoding UTF8
      continue 
    }

    $msg = "$timestamp - Changes detected"
    Write-Host $msg
    $msg | Out-File -FilePath $logFile -Append -Encoding UTF8
    
    $stats = Get-DiffStats
    if ($stats.NumFiles -eq 0) { 
      git reset 2>&1 | Out-Null
      $msg = "$timestamp - Staging reset"
      Write-Host $msg
      $msg | Out-File -FilePath $logFile -Append -Encoding UTF8
      continue 
    }

    $isLarge = ($stats.LinesChanged -ge $LargeDiffLines) -or ($stats.NumFiles -ge $LargeDiffFiles)
    $msg = New-CommitMessage $stats $isLarge

    $commitMsg = "$timestamp - Committing: $($stats.NumFiles) files, $($stats.LinesChanged) lines"
    Write-Host $commitMsg
    $commitMsg | Out-File -FilePath $logFile -Append -Encoding UTF8
    
    $commitResult = git commit -m $msg 2>&1
    if ($LASTEXITCODE -eq 0) {
      $successMsg = "$timestamp - Commit successful"
      Write-Host $successMsg
      $successMsg | Out-File -FilePath $logFile -Append -Encoding UTF8
    } else {
      $errorMsg = "$timestamp - Commit failed: $commitResult"
      Write-Warning $errorMsg
      $errorMsg | Out-File -FilePath $logFile -Append -Encoding UTF8
      continue
    }

    if ($isLarge) {
      $branch = "auto/" + (Get-Date -Format "yyyyMMdd-HHmmss")
      $branchMsg = "$timestamp - Large change detected, creating branch: $branch"
      Write-Host $branchMsg
      $branchMsg | Out-File -FilePath $logFile -Append -Encoding UTF8
      
      git switch -c $branch 2>&1 | Out-Null
      $pushResult = git push -u origin $branch 2>&1
      if ($LASTEXITCODE -eq 0) {
        $pushSuccessMsg = "$timestamp - Branch push successful"
        Write-Host $pushSuccessMsg
        $pushSuccessMsg | Out-File -FilePath $logFile -Append -Encoding UTF8
        
        try {
          $title = ("{0}: large auto update" -f $branch)
          $prResult = gh pr create --fill --title $title --body "Auto-generated PR for large change." 2>&1
          if ($LASTEXITCODE -eq 0) {
            $prSuccessMsg = "$timestamp - PR created successfully"
            Write-Host $prSuccessMsg
            $prSuccessMsg | Out-File -FilePath $logFile -Append -Encoding UTF8
          } else {
            $prErrorMsg = "$timestamp - PR creation failed: $prResult"
            Write-Warning $prErrorMsg
            $prErrorMsg | Out-File -FilePath $logFile -Append -Encoding UTF8
          }
        } catch { 
          $prExceptionMsg = "$timestamp - PR creation error: $_"
          Write-Warning $prExceptionMsg
          $prExceptionMsg | Out-File -FilePath $logFile -Append -Encoding UTF8
        }
      } else {
        $pushErrorMsg = "$timestamp - Branch push failed: $pushResult"
        Write-Warning $pushErrorMsg
        $pushErrorMsg | Out-File -FilePath $logFile -Append -Encoding UTF8
      }
      git switch - 2>&1 | Out-Null
    } else {
      $pushResult = git push 2>&1
      if ($LASTEXITCODE -eq 0) {
        $pushSuccessMsg = "$timestamp - Push successful"
        Write-Host $pushSuccessMsg
        $pushSuccessMsg | Out-File -FilePath $logFile -Append -Encoding UTF8
      } else {
        $pushErrorMsg = "$timestamp - Push failed: $pushResult"
        Write-Warning $pushErrorMsg
        $pushErrorMsg | Out-File -FilePath $logFile -Append -Encoding UTF8
      }
    }
  } catch {
    $exceptionMsg = "$timestamp - Error occurred: $_"
    Write-Warning $exceptionMsg
    $exceptionMsg | Out-File -FilePath $logFile -Append -Encoding UTF8
  }
}
