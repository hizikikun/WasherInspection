# 学習中のリモートアクセス設定

学習中でも安全にリモートアクセスを設定・起動できます。

## 学習中でも実行可能

リモートサーバーやトンネルは、学習プロセスとは独立して動作するため、**学習中でも安全に起動できます**。

## 方法1: バッチファイルから実行（推奨）

### リモートサーバーのみ起動
```
start_remote_server_background.bat
```

### インターネット経由アクセストンネル起動
```
start_tunnel_background.bat
```

### 完全セットアップ（学習中でも実行可能）
```
setup_remote_access_while_training.bat
```

## 方法2: 統合アプリから

学習タブの以下ボタンから起動できます：
- **🌐 リモートサーバー起動** - ローカルネットワーク経由
- **🌍 インターネット経由アクセス** - インターネット経由

学習中でも安全に実行できます。

## 方法3: コマンドラインから

### リモートサーバー起動
```bash
python scripts\remote_server.py
```

### トンネル起動
```bash
python scripts\remote_server_tunnel.py --start
```

## 注意事項

1. **学習への影響なし**
   - リモートサーバーとトンネルは独立したプロセスで動作
   - 学習プロセスのCPU/GPU使用率には影響しません
   - ネットワーク帯域のみを使用

2. **バックグラウンド実行**
   - バッチファイルは `start /B` でバックグラウンド実行
   - ウィンドウを閉じてもサーバーは動作し続けます

3. **停止方法**
   - タスクマネージャーでPythonプロセスを終了
   - または、統合アプリから停止

## セットアップ済みか確認

以下のファイルが存在すれば、セットアップは完了しています：
- `config/remote_tunnel_config.json`
- `config/remote_server_config.json`

存在しない場合は、以下を実行：
```bash
python scripts\auto_setup_remote_access.py
```

## トラブルシューティング

### ポート5000が使用中

別のプロセスがポート5000を使用している可能性があります：
```bash
netstat -ano | findstr :5000
```

### ngrokが起動しない

1. ngrokがインストールされているか確認：
   ```bash
   ngrok version
   ```

2. インストールされていない場合：
   ```bash
   python scripts\auto_setup_remote_access.py
   ```

### アクセスできない

1. ファイアウォール設定を確認
2. リモートサーバーが起動しているか確認
3. ブラウザで `http://localhost:5000` にアクセスして確認





