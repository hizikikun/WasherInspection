# インターネット経由アクセス機能

外出先の異なるネットワークからもアクセスできるようにする機能です。

## 対応方法

### 1. ngrok（推奨・簡単）

**特徴:**
- 無料で使用可能
- 設定が簡単
- 動的にURLが生成される

**セットアップ:**

1. ngrokをインストール
   - https://ngrok.com/download からダウンロード
   - 解凍してPATHに追加

2. 設定
   ```bash
   python scripts/configure_tunnel.py
   ```
   または手動で `config/remote_tunnel_config.json` を編集

3. トンネルを起動
   ```bash
   python scripts/remote_server_tunnel.py --start
   ```
   または
   ```bash
   start_tunnel.bat
   ```

**アクセス:**
- 起動時に表示されるURL（例: `https://xxxx-xxxx-xxxx.ngrok.io`）を使用
- このURLはインターネット上のどこからでもアクセス可能

### 2. Cloudflare Tunnel

**特徴:**
- 無料で高速
- より安定した接続
- 独自ドメインを使用可能

**セットアップ:**

1. Cloudflareアカウントを作成
   - https://one.dash.cloudflare.com/ にアクセス

2. Tunnelを作成してトークンを取得

3. cloudflaredをインストール
   - https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/

4. 設定
   ```bash
   python scripts/configure_tunnel.py
   ```
   Tunnel Tokenを入力

5. トンネルを起動
   ```bash
   python scripts/remote_server_tunnel.py --start
   ```

### 3. カスタムトンネル

独自のトンネルサービスを使用する場合:

1. 設定
   ```bash
   python scripts/configure_tunnel.py
   ```
   起動コマンドを入力

2. トンネルを起動
   ```bash
   python scripts/remote_server_tunnel.py --start
   ```

## 使用方法

### 統合アプリから

1. 学習タブを開く
2. 「🌍 インターネット経由アクセス」ボタンをクリック
3. URLが表示されたら、それをコピー
4. 別のPCやスマートフォンからそのURLにアクセス

### コマンドラインから

```bash
# トンネルを開始
python scripts/remote_server_tunnel.py --start

# トンネルを停止
python scripts/remote_server_tunnel.py --stop

# 状態を確認
python scripts/remote_server_tunnel.py --status
```

## セキュリティ注意事項

⚠️ **重要: インターネット経由でのアクセス**

1. **認証の追加**
   - 現在は認証なしでアクセス可能
   - 本番環境では必ず認証を有効化してください

2. **HTTPSの使用**
   - ngrokやCloudflare Tunnelは自動的にHTTPSを使用します

3. **一時的なアクセスのみ**
   - トンネルを停止するとURLは無効になります
   - 使用後は必ずトンネルを停止してください

4. **ファイアウォール設定**
   - ローカルファイアウォールでポート5000を開放する必要があります

## トラブルシューティング

### ngrokが起動しない

1. ngrokがインストールされているか確認
   ```bash
   ngrok version
   ```

2. 認証トークンが設定されているか確認
   - 無料プランでも使用可能ですが、認証トークンがあるとより安定します

### URLが取得できない

1. 数秒待ってから再試行
2. コンソール出力を確認
3. 設定ファイルを確認

### 接続できない

1. ローカルサーバーが起動しているか確認
   ```bash
   python scripts/remote_server.py
   ```

2. ファイアウォール設定を確認
3. トンネルが正常に起動しているか確認

## 無料プランの制限

### ngrok無料プラン
- セッションが8時間で切れる
- URLは起動のたびに変わる
- 帯域制限あり

### Cloudflare Tunnel無料プラン
- 無料で使用可能
- より安定した接続
- 独自ドメイン使用可能

## 推奨設定

外出先からのアクセスには、以下の設定を推奨します：

1. **ngrokを使用**（最も簡単）
2. **認証トークンを設定**（より安定）
3. **使用後は必ず停止**（セキュリティ）





