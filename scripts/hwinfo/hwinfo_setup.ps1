# HWiNFO Auto Restart Task Scheduler Setup (PowerShell)

$TaskName = "HWiNFO_AutoRestart"
$PythonPath = "C:\Users\tomoh\AppData\Local\Programs\Python\Python310\python.exe"
$ScriptPath = "$PSScriptRoot\hwinfo_auto_restart.py"
$Arguments = "`"$ScriptPath`" restart"

# Remove existing task if present
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($ExistingTask) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "[INFO] Existing task removed"
}

# Task action
$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument $Arguments -WorkingDirectory $PSScriptRoot

# Task trigger (at logon, repeat every 12 hours)
$Trigger = New-ScheduledTaskTrigger -AtLogOn
$Trigger.Repetition = @{
    Interval = "PT12H"
    Duration = "PT0S"
}

# Task settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable:$false

# Task principal
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest

# Register task
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "Auto restart HWiNFO64 every 12 hours to avoid Shared Memory limitation"
    
    Write-Host "[OK] Task registered successfully" -ForegroundColor Green
    Write-Host ""
    Write-Host "Task Name: $TaskName" -ForegroundColor Cyan
    Write-Host "Interval: 12 hours" -ForegroundColor Cyan
    Write-Host "Start: At logon" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Check in Task Scheduler:" -ForegroundColor Yellow
    Write-Host "  Start Menu > Task Scheduler"
    Write-Host "  Task Scheduler Library > $TaskName"
}
catch {
    Write-Host "[ERROR] Failed to register task: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please refer to: docs/HWiNFO_AUTO_RESTART_SETUP.md" -ForegroundColor Yellow
    exit 1
}

