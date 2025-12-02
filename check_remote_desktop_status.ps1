# リモートデスクトップの状態を確認するスクリプト

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "リモートデスクトップ状態確認" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# リモートデスクトップの有効/無効状態
Write-Host "[1] リモートデスクトップの状態:" -ForegroundColor Yellow
try {
    $rdpEnabled = (Get-ItemProperty -Path 'HKLM:\System\CurrentControlSet\Control\Terminal Server' -Name "fDenyTSConnections" -ErrorAction Stop).fDenyTSConnections
    
    if ($rdpEnabled -eq 0) {
        Write-Host "  ✅ 有効" -ForegroundColor Green
    } else {
        Write-Host "  ❌ 無効" -ForegroundColor Red
    }
} catch {
    Write-Host "  ⚠️  状態を取得できませんでした" -ForegroundColor Yellow
}

Write-Host ""

# ファイアウォールルールの状態
Write-Host "[2] ファイアウォールルールの状態:" -ForegroundColor Yellow
try {
    $firewallRules = Get-NetFirewallRule -DisplayGroup "リモートデスクトップ" -ErrorAction SilentlyContinue
    
    if ($firewallRules) {
        $enabledRules = $firewallRules | Where-Object { $_.Enabled -eq $true }
        if ($enabledRules) {
            Write-Host "  ✅ 有効なルールが存在します" -ForegroundColor Green
            foreach ($rule in $enabledRules) {
                Write-Host "    - $($rule.DisplayName)" -ForegroundColor White
            }
        } else {
            Write-Host "  ❌ ルールは存在しますが、無効です" -ForegroundColor Red
        }
    } else {
        Write-Host "  ❌ ルールが見つかりませんでした" -ForegroundColor Red
    }
} catch {
    Write-Host "  ⚠️  状態を取得できませんでした" -ForegroundColor Yellow
}

Write-Host ""

# ネットワークレベル認証の状態
Write-Host "[3] ネットワークレベル認証の状態:" -ForegroundColor Yellow
try {
    $nlaEnabled = (Get-ItemProperty -Path 'HKLM:\System\CurrentControlSet\Control\Terminal Server\WinStations\RDP-Tcp' -Name "UserAuthentication" -ErrorAction Stop).UserAuthentication
    
    if ($nlaEnabled -eq 1) {
        Write-Host "  ✅ 有効（推奨）" -ForegroundColor Green
    } else {
        Write-Host "  ❌ 無効" -ForegroundColor Red
        Write-Host "     ⚠️  セキュリティのため、有効化することをお勧めします" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ⚠️  状態を取得できませんでした" -ForegroundColor Yellow
}

Write-Host ""

# ネットワーク情報
Write-Host "[4] ネットワーク情報:" -ForegroundColor Yellow
$computerName = $env:COMPUTERNAME
Write-Host "  コンピューター名: $computerName" -ForegroundColor White

try {
    $ipAddresses = Get-NetIPAddress -AddressFamily IPv4 | Where-Object { 
        $_.IPAddress -notlike "127.*" -and 
        $_.IPAddress -notlike "169.254.*" 
    } | Select-Object -ExpandProperty IPAddress
    
    if ($ipAddresses) {
        Write-Host "  IPアドレス:" -ForegroundColor White
        foreach ($ip in $ipAddresses) {
            Write-Host "    - $ip" -ForegroundColor Green
        }
    } else {
        Write-Host "  ⚠️  IPアドレスを取得できませんでした" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ⚠️  IPアドレスを取得できませんでした" -ForegroundColor Yellow
}

Write-Host ""

# ユーザーアカウント情報
Write-Host "[5] ユーザーアカウント情報:" -ForegroundColor Yellow
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
Write-Host "  現在のユーザー: $currentUser" -ForegroundColor White

try {
    $user = Get-LocalUser -Name $env:USERNAME -ErrorAction Stop
    if ($user.PasswordExpires) {
        Write-Host "  ✅ パスワードが設定されています" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  パスワードの状態を確認できませんでした" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ⚠️  ユーザー情報を取得できませんでした" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "確認完了" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 接続方法の案内
Write-Host "接続方法:" -ForegroundColor Yellow
Write-Host "1. リモート接続するPCで 'mstsc' コマンドを実行" -ForegroundColor White
Write-Host "2. 上記のIPアドレスまたはコンピューター名を入力" -ForegroundColor White
Write-Host "3. このPCのユーザー名とパスワードでログイン" -ForegroundColor White
Write-Host ""

Write-Host "⚠️  注意:" -ForegroundColor Red
Write-Host "- 大学のWiFiから接続する場合、ポート3389がブロックされている可能性があります" -ForegroundColor Yellow
Write-Host "- その場合は、TeamViewerやAnyDeskなどのソフトウェアを使用することをお勧めします" -ForegroundColor Yellow
Write-Host ""

pause







