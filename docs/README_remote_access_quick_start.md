# リモートアクセス クイックスタート

## 問題: バッチファイルが見つからない

コマンドプロンプトで `setup_remote_access_while_training.bat` を実行しても「認識されていません」というエラーが出る場合、**プロジェクトのディレクトリに移動する必要があります**。

## 解決方法

### 方法1: プロジェクトディレクトリに移動してから実行（推奨）

```cmd
cd C:\Users\tomoh\WasherInspection
setup_remote_access_while_training.bat
```

または、`quick_start_remote_access.bat` を使用（どこからでも実行可能）：
```cmd
cd C:\Users\tomoh\WasherInspection
quick_start_remote_access.bat
```

### 方法2: フルパスで実行

```cmd
C:\Users\tomoh\WasherInspection\setup_remote_access_while_training.bat
```

### 方法3: エクスプローラーから実行

1. エクスプローラーで `C:\Users\tomoh\WasherInspection` を開く
2. `setup_remote_access_while_training.bat` をダブルクリック

## クイックスタート手順

### ステップ1: プロジェクトディレクトリに移動

```cmd
cd C:\Users\tomoh\WasherInspection
```

### ステップ2: リモートアクセスを起動

**オプションA: 完全セットアップ（初回のみ）**
```cmd
setup_remote_access_while_training.bat
```

**オプションB: クイックスタート（推奨）**
```cmd
quick_start_remote_access.bat
```

**オプションC: リモートサーバーのみ**
```cmd
start_remote_server_background.bat
```

**オプションD: インターネット経由アクセス**
```cmd
start_tunnel_background.bat
```

### ステップ3: アクセス

- **ローカル**: http://localhost:5000
- **リモート**: http://<このPCのIPアドレス>:5000

## 学習中でも実行可能

すべてのバッチファイルは学習中でも安全に実行できます。

## トラブルシューティング

### エラー: "認識されていません"

**原因**: プロジェクトディレクトリにいない

**解決方法**:
```cmd
cd C:\Users\tomoh\WasherInspection
```

### ポート5000が使用中

別のプロセスが使用している可能性があります。確認：
```cmd
netstat -ano | findstr :5000
```

### ngrokが見つからない

自動セットアップを実行：
```cmd
cd C:\Users\tomoh\WasherInspection
python scripts\auto_setup_remote_access.py
```





