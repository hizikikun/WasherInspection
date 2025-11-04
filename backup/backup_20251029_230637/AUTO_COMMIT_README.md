# GitHub自動転送システム

このシステムは、プロジェクトファイルの変更を自動的に検出してGitHubにコミット・プッシュする機能を提供します。

## 機能

- **自動変更検出**: ファイルの変更をリアルタイムで監視
- **自動コミット**: 変更を自動的にGitにコミット
- **自動プッシュ**: コミットをGitHubに自動プッシュ
- **大きな変更の処理**: 大きな変更は自動的にブランチを作成してPRを作成
- **ログ記録**: すべての操作をログファイルに記録

## 使用方法

### 1. 手動起動
```batch
start-auto-commit.bat
```

### 2. 自動起動設定（推奨）
1. タスクスケジューラーを開く
2. 「タスクの作成」を選択
3. 「XMLファイルからインポート」を選択
4. `setup-auto-commit.xml`を選択
5. ユーザーIDを現在のユーザーに変更
6. タスクを有効化

### 3. PowerShellから直接実行
```powershell
cd C:\Users\tomoh\WasherInspection
.\auto-commit.ps1
```

## 設定パラメータ

`auto-commit.ps1`の先頭で以下のパラメータを調整できます：

- `$QuietSeconds = 60`: 変更検出後の待機時間（秒）
- `$LargeDiffLines = 100`: 大きな変更と判定する行数
- `$LargeDiffFiles = 3`: 大きな変更と判定するファイル数

## ログファイル

- `auto-commit.log`: システムの動作ログ
- タイムスタンプ付きで全ての操作が記録されます

## 動作の流れ

1. **変更検出**: ファイルシステムの変更を監視
2. **待機**: 設定された時間（デフォルト60秒）待機
3. **ステージング**: `git add -A`で変更をステージング
4. **コミット**: 自動生成されたメッセージでコミット
5. **プッシュ**: GitHubにプッシュ
6. **大きな変更の場合**: ブランチ作成→PR作成

## トラブルシューティング

### よくある問題

1. **GitHub CLIが認証されていない**
   ```bash
   gh auth login
   ```

2. **PowerShellの実行ポリシーエラー**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Gitの認証エラー**
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

### ログの確認

```bash
type auto-commit.log
```

### プロセスの確認

```powershell
Get-Process | Where-Object {$_.ProcessName -like "*powershell*"}
```

## 停止方法

- **手動起動の場合**: Ctrl+C
- **タスクスケジューラー**: タスクを無効化
- **プロセス終了**: タスクマネージャーでPowerShellプロセスを終了

## 注意事項

- このシステムは継続的に動作します
- 大きな変更は自動的にPRが作成されます
- ログファイルは定期的に確認してください
- ネットワーク接続が必要です

## サポート

問題が発生した場合は、`auto-commit.log`ファイルの内容を確認してください。



