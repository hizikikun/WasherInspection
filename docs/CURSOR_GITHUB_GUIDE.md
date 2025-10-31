# Cursor GitHub統合システム セットアップガイド

## 概要
Cursor IDE上から直接GitHubとやり取りできるシステムです。ブラウザを開かずに、ワンクリックで送受信が可能です。

## 機能

### 🚀 ワンクリック操作
- **📤 Send to GitHub**: 現在の変更をワンクリックでGitHubに送信
- **📥 Receive from GitHub**: GitHubの最新変更をワンクリックで受信
- **🔄 Auto-Sync**: 自動的にバックグラウンドで同期

### 🔄 自動同期
- **バックグラウンド監視**: ファイル変更を自動検出
- **閾値設定**: 指定したファイル数で自動送信
- **時間間隔**: 設定した間隔で自動チェック
- **双方向同期**: 送信と受信を自動実行

### 🎯 Cursor統合
- **GUI拡張**: Cursor内でGUIインターフェース
- **ステータス表示**: 接続状態と同期状況を表示
- **ログ表示**: すべての操作をログで確認
- **通知機能**: 操作完了時に通知

## セットアップ手順

### 1. GitHub認証の設定

#### 方法1: 自動設定（推奨）
```bash
python cursor_github_integration.py
```
- 自動的にブラウザが開いてトークン作成ページに移動
- トークンを作成後、コマンドラインで設定

#### 方法2: 手動設定
1. GitHub → Settings → Developer settings → Personal access tokens
2. "Generate new token" → "Generate new token (classic)"
3. `repo`権限を選択
4. トークンをコピー
5. 設定ファイルに貼り付け

### 2. 設定ファイルの編集
`cursor_github_config.json`を編集：
```json
{
  "github_token": "ghp_your_token_here",
  "github_owner": "your-username",
  "github_repo": "washer-inspection-code",
  "auto_sync_enabled": true,
  "sync_interval": 60,
  "auto_commit_threshold": 3,
  "one_click_enabled": true,
  "notifications_enabled": true
}
```

### 3. システムの起動

#### GUI拡張（推奨）
```bash
# バッチファイル
start_cursor_gui.bat

# 直接実行
python cursor_github_extension.py
```

#### コマンドライン版
```bash
# バッチファイル
start_cursor_github.bat

# 直接実行
python cursor_github_integration.py
```

#### 自動同期モード
```bash
# バッチファイル
start_auto_sync.bat

# 直接実行
python cursor_github_integration.py --auto-sync
```

## 使用方法

### GUI拡張の操作

#### メイン画面
- **📤 Send to GitHub**: 現在の変更を送信
- **📥 Receive from GitHub**: 最新変更を受信
- **🔄 Start/Stop Auto-Sync**: 自動同期の開始/停止
- **⚙️ Settings**: 設定の変更
- **Activity Log**: 操作履歴の確認

#### ステータス表示
- **🔴 Not Connected**: GitHubに接続されていない
- **🟡 Connected**: GitHubに接続済み
- **🟢 Auto-Sync Running**: 自動同期が動作中

### コマンドライン操作

#### インタラクティブモード
```bash
python cursor_github_integration.py
```

#### ワンクリック送信
```bash
python cursor_github_integration.py --send
```

#### ワンクリック受信
```bash
python cursor_github_integration.py --receive
```

#### トークン設定
```bash
python cursor_github_integration.py --token YOUR_TOKEN
```

## 自動同期の設定

### 同期間隔
```json
{
  "sync_interval": 60  // 60秒間隔でチェック
}
```

### 自動送信閾値
```json
{
  "auto_commit_threshold": 3  // 3ファイル変更で自動送信
}
```

### 監視ファイル
```json
{
  "include_files": [".py", ".json", ".md", ".yml", ".yaml", ".txt"],
  "exclude_dirs": ["__pycache__", ".git", "node_modules", ".vscode"]
}
```

## 通知機能

### システム通知
- **Windows**: MessageBoxで通知
- **macOS**: システム通知で表示
- **Linux**: notify-sendで通知

### ログ通知
- すべての操作がログに記録
- タイムスタンプ付きで表示
- GUI内でリアルタイム確認

## トラブルシューティング

### GitHub接続エラー
1. トークンが正しく設定されているか確認
2. リポジトリ名とオーナー名が正しいか確認
3. インターネット接続を確認
4. トークンの権限（repo）を確認

### 自動同期が動作しない
1. `auto_sync_enabled`が`true`になっているか確認
2. `sync_interval`の値を確認
3. `auto_commit_threshold`の値を確認
4. ファイルが監視対象に含まれているか確認

### GUIが表示されない
1. tkinterがインストールされているか確認
2. Pythonのバージョンを確認（3.6以上推奨）
3. エラーメッセージを確認

## カスタマイズ

### 通知の無効化
```json
{
  "notifications_enabled": false
}
```

### ファイル監視の追加
```json
{
  "include_files": [".py", ".json", ".md", ".yml", ".yaml", ".txt", ".csv", ".xml"]
}
```

### 除外ディレクトリの追加
```json
{
  "exclude_dirs": ["__pycache__", ".git", "node_modules", ".vscode", "backup", "temp", "logs"]
}
```

## セキュリティ注意事項

1. **Personal Access Token**: 絶対に他人に教えない
2. **設定ファイル**: `cursor_github_config.json`を`.gitignore`に追加
3. **環境変数**: 可能であれば環境変数でトークンを管理
4. **リポジトリ権限**: 必要最小限の権限のみ付与

## パフォーマンス最適化

### 同期間隔の調整
```json
{
  "sync_interval": 120  // 2分間隔に変更（負荷軽減）
}
```

### 閾値の調整
```json
{
  "auto_commit_threshold": 5  // 5ファイルで自動送信（頻度軽減）
}
```

### ファイル監視の最適化
```json
{
  "include_files": [".py", ".json"]  // 必要最小限のファイルのみ監視
}
```

## 高度な機能

### バッチ送信
- 複数の変更をまとめて送信
- 効率的なネットワーク使用
- コミット履歴の整理

### 差分表示
- 送信前の変更内容を確認
- ファイルごとの変更量を表示
- 送信するかどうかの判断支援

### 履歴管理
- 送受信履歴の保存
- 操作ログの検索
- エラー履歴の追跡
