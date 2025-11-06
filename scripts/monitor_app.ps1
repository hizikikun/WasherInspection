# アプリ監視スクリプト
# アプリの起動状態、エラーログ、プロセスを監視します

$appName = "統合ワッシャー検査・学習システム"
$logFile = "app_startup.log"
$errorLogFile = "app_error_log.txt"
$monitorInterval = 5  # 5秒ごとにチェック

Write-Host "========================================"
Write-Host "アプリ監視を開始します"
Write-Host "========================================"
Write-Host "アプリ名: $appName"
Write-Host "監視間隔: $monitorInterval 秒"
Write-Host "Ctrl+C で終了"
Write-Host "========================================"
Write-Host ""

$checkCount = 0

while ($true) {
    $checkCount++
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    Write-Host "[$timestamp] 監視チェック #$checkCount" -ForegroundColor Cyan
    
    # 1. Pythonプロセスをチェック
    $pythonProcesses = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $_.MainWindowTitle -like "*ワッシャー*" -or 
        $_.CommandLine -like "*integrated_washer_app*"
    }
    
    if ($pythonProcesses) {
        Write-Host "  [OK] Pythonプロセス実行中: $($pythonProcesses.Count)個" -ForegroundColor Green
        foreach ($proc in $pythonProcesses) {
            Write-Host "    - PID: $($proc.Id), メモリ: $([math]::Round($proc.WorkingSet64/1MB, 2)) MB" -ForegroundColor Gray
        }
    } else {
        Write-Host "  [警告] Pythonプロセスが見つかりません" -ForegroundColor Yellow
        
        # すべてのPythonプロセスを確認
        $allPython = Get-Process python -ErrorAction SilentlyContinue
        if ($allPython) {
            Write-Host "    実行中のPythonプロセス: $($allPython.Count)個" -ForegroundColor Gray
        }
    }
    
    # 2. エラーログをチェック
    if (Test-Path $errorLogFile) {
        $errorLog = Get-Content $errorLogFile -Tail 10 -Encoding UTF8 -ErrorAction SilentlyContinue
        if ($errorLog) {
            $latestError = $errorLog | Select-Object -Last 1
            if ($latestError -match "エラー" -or $latestError -match "ERROR" -or $latestError -match "Traceback") {
                Write-Host "  [エラー] エラーログに新しいエラーが検出されました" -ForegroundColor Red
                Write-Host "    最新のエラー:" -ForegroundColor Yellow
                $errorLog | Select-Object -Last 3 | ForEach-Object {
                    Write-Host "    $_" -ForegroundColor Red
                }
            } else {
                Write-Host "  [OK] エラーログ: 問題なし" -ForegroundColor Green
            }
        }
    } else {
        Write-Host "  [情報] エラーログファイルが存在しません" -ForegroundColor Gray
    }
    
    # 3. 起動ログをチェック
    if (Test-Path $logFile) {
        $logContent = Get-Content $logFile -Tail 5 -Encoding UTF8 -ErrorAction SilentlyContinue
        if ($logContent) {
            Write-Host "  [情報] 最新のログ:" -ForegroundColor Gray
            $logContent | ForEach-Object {
                Write-Host "    $_" -ForegroundColor DarkGray
            }
        }
    }
    
    # 4. ウィンドウの状態をチェック
    $windows = Get-Process | Where-Object { $_.MainWindowTitle -like "*ワッシャー*" -or $_.MainWindowTitle -like "*統合*" }
    if ($windows) {
        Write-Host "  [OK] アプリウィンドウが検出されました" -ForegroundColor Green
        foreach ($win in $windows) {
            Write-Host "    - $($win.MainWindowTitle) (PID: $($win.Id))" -ForegroundColor Gray
        }
    } else {
        Write-Host "  [情報] アプリウィンドウは非表示の可能性があります" -ForegroundColor Gray
    }
    
    Write-Host ""
    
    # 次のチェックまで待機
    Start-Sleep -Seconds $monitorInterval
}

