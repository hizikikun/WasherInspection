# リモート操作機能

このPCを別のPCから遠隔操作できるようにする機能です。

## 機能

- ✅ 学習の開始・停止
- ✅ 学習進捗のリアルタイム監視
- ✅ ログの表示
- ✅ 精度・損失の確認
- ✅ Webインターフェース（ブラウザから操作）

## セットアップ

### 1. 必要なパッケージのインストール

```bash
pip install flask flask-cors
```

### 2. サーバーの起動

**方法1: バッチファイルから起動**
```
start_remote_server.bat
```

**方法2: Pythonスクリプトから起動**
```bash
python scripts/start_remote_server.py
```

**方法3: 直接起動**
```bash
python scripts/remote_server.py
```

### 3. アクセス方法

**同じPCから:**
```
http://localhost:5000
```

**別PCから:**
```
http://<このPCのIPアドレス>:5000
```

例: `http://192.168.1.100:5000`

## IPアドレスの確認方法

### Windows
```cmd
ipconfig
```
「IPv4 アドレス」を確認

### Linux/Mac
```bash
ifconfig
```
または
```bash
hostname -I
```

## ファイアウォール設定

Windowsファイアウォールでポート5000を開放する必要があります。

### Windowsでの設定手順

1. 「Windows Defender ファイアウォール」を開く
2. 「詳細設定」をクリック
3. 「受信の規則」→「新しい規則」
4. 「ポート」を選択
5. TCPポート5000を指定
6. 「接続を許可する」を選択
7. すべてのプロファイルに適用
8. 名前を付けて保存

または、PowerShellで実行:
```powershell
New-NetFirewallRule -DisplayName "WasherInspection Remote Server" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow
```

## 使用方法

1. サーバーを起動
2. ブラウザで `http://<IPアドレス>:5000` にアクセス
3. 学習開始ボタンをクリックして学習を開始
4. 進捗がリアルタイムで更新されます
5. 学習停止ボタンで学習を停止できます

## セキュリティ

現在は認証なしでアクセス可能です。本番環境では以下を推奨：

1. **パスワード認証の有効化**
   - `config/remote_server_config.json` で設定

2. **特定IPアドレスのみ許可**
   - `allowed_ips` に許可するIPアドレスを追加

3. **HTTPSの使用**
   - リバースプロキシ（nginxなど）を使用

4. **VPN経由でのアクセス**
   - より安全な接続方法

## トラブルシューティング

### 接続できない

1. **ファイアウォール設定を確認**
   - ポート5000が開放されているか確認

2. **IPアドレスを確認**
   - サーバー起動時に表示されるIPアドレスを使用

3. **ネットワーク接続を確認**
   - 同じネットワークに接続されているか確認

4. **サーバーが起動しているか確認**
   - エラーメッセージを確認

### 学習が開始できない

1. **学習スクリプトのパスを確認**
   - `scripts/train_4class_sparse_ensemble.py` が存在するか確認

2. **Python環境を確認**
   - 必要なパッケージがインストールされているか確認

3. **ログを確認**
   - サーバーのコンソール出力を確認

## カスタマイズ

### ポート番号の変更

```bash
python scripts/remote_server.py --port 8080
```

### 特定のホストにバインド

```bash
python scripts/remote_server.py --host 192.168.1.100
```

### デバッグモード

```bash
python scripts/remote_server.py --debug
```

## API エンドポイント

### GET /api/status
学習状態を取得

### POST /api/start
学習を開始

### POST /api/stop
学習を停止

### GET /api/logs
ログファイルを取得




