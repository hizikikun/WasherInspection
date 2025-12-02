# リモートデスクトップ設定スクリプト
# このスクリプトは、Windowsリモートデスクトップ（RDP）を有効化します

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "リモートデスクトップ設定スクリプト" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 管理者権限の確認
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "エラー: このスクリプトは管理者権限で実行する必要があります。" -ForegroundColor Red
    Write-Host "PowerShellを管理者として実行してください。" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "続行するには、管理者権限でPowerShellを開き、このスクリプトを再実行してください。" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "[1/5] リモートデスクトップの状態を確認中..." -ForegroundColor Yellow

# リモートデスクトップの現在の状態を確認
$rdpEnabled = (Get-ItemProperty -Path 'HKLM:\System\CurrentControlSet\Control\Terminal Server' -Name "fDenyTSConnections").fDenyTSConnections

if ($rdpEnabled -eq 0) {
    Write-Host "リモートデスクトップは既に有効です。" -ForegroundColor Green
} else {
    Write-Host "[2/5] リモートデスクトップを有効化中..." -ForegroundColor Yellow
    
    # リモートデスクトップを有効化
    Set-ItemProperty -Path 'HKLM:\System\CurrentControlSet\Control\Terminal Server' -Name "fDenyTSConnections" -Value 0
    
    Write-Host "リモートデスクトップを有効化しました。" -ForegroundColor Green
}

Write-Host "[3/5] ファイアウォールルールを確認中..." -ForegroundColor Yellow

# ファイアウォールルールを確認・作成
$firewallRule = Get-NetFirewallRule -Name "RemoteDesktop*" -ErrorAction SilentlyContinue

if ($firewallRule) {
    Write-Host "ファイアウォールルールは既に存在します。" -ForegroundColor Green
} else {
    Write-Host "ファイアウォールルールを作成中..." -ForegroundColor Yellow
    
    # ファイアウォールルールを有効化
    Enable-NetFirewallRule -DisplayGroup "リモートデスクトップ"
    
    Write-Host "ファイアウォールルールを有効化しました。" -ForegroundColor Green
}

Write-Host "[4/5] ネットワークレベル認証の設定を確認中..." -ForegroundColor Yellow

# ネットワークレベル認証の設定（セキュリティ向上）
$nlaEnabled = (Get-ItemProperty -Path 'HKLM:\System\CurrentControlSet\Control\Terminal Server\WinStations\RDP-Tcp' -Name "UserAuthentication" -ErrorAction SilentlyContinue).UserAuthentication

if ($nlaEnabled -eq 1) {
    Write-Host "ネットワークレベル認証は既に有効です。" -ForegroundColor Green
} else {
    Write-Host "ネットワークレベル認証を有効化中..." -ForegroundColor Yellow
    
    Set-ItemProperty -Path 'HKLM:\System\CurrentControlSet\Control\Terminal Server\WinStations\RDP-Tcp' -Name "UserAuthentication" -Value 1
    
    Write-Host "ネットワークレベル認証を有効化しました。" -ForegroundColor Green
}

Write-Host "[5/5] ネットワーク情報を取得中..." -ForegroundColor Yellow

# IPアドレス情報を取得
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "接続情報" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# コンピューター名
$computerName = $env:COMPUTERNAME
Write-Host "コンピューター名: $computerName" -ForegroundColor White

# IPアドレス
$ipAddresses = Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } | Select-Object -ExpandProperty IPAddress

if ($ipAddresses) {
    Write-Host "IPアドレス:" -ForegroundColor White
    foreach ($ip in $ipAddresses) {
        Write-Host "  - $ip" -ForegroundColor Green
    }
} else {
    Write-Host "IPアドレス: 取得できませんでした" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "設定完了" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "リモートデスクトップが有効化されました。" -ForegroundColor Green
Write-Host ""
Write-Host "接続方法:" -ForegroundColor Yellow
Write-Host "1. リモート接続するPCで 'mstsc' コマンドを実行" -ForegroundColor White
Write-Host "2. 上記のIPアドレスまたはコンピューター名を入力" -ForegroundColor White
Write-Host "3. このPCのユーザー名とパスワードでログイン" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  注意事項:" -ForegroundColor Red
Write-Host "- このPCのユーザーアカウントにパスワードが設定されている必要があります" -ForegroundColor Yellow
Write-Host "- 大学のWiFiから接続する場合、ポート3389がブロックされている可能性があります" -ForegroundColor Yellow
Write-Host "- その場合は、TeamViewerやAnyDeskなどのソフトウェアを使用することをお勧めします" -ForegroundColor Yellow
Write-Host ""
Write-Host "詳細は REMOTE_DESKTOP_SETUP_GUIDE.md を参照してください。" -ForegroundColor Cyan
Write-Host ""

pause







