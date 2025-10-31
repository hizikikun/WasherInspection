# Secret漏洩のセキュリティ対処ガイド

## ⚠️ 重要な警告

**GitHub Secretの許可は推奨されません。** 以下のリスクがあります：

### リスク

1. **トークンの公開**
   - リポジトリにコミットされると、そのトークンが公開されます
   - 履歴を見れば誰でもアクセスできます
   - 公開リポジトリの場合、世界中の誰でも見られます

2. **不正アクセスの可能性**
   - トークンを使って、あなたのアカウントの代理で操作される可能性
   - リポジトリの削除、変更、データの窃取など

3. **長期的なリスク**
   - 一度コミットされると、Git履歴に永続的に残ります
   - 削除しても履歴から見つけることができます

## ✅ 推奨される対処法

### 方法1: トークンを無効化して再作成（最も安全）

1. **既存のトークンを無効化**：
   - GitHub → Settings → Developer settings → Personal access tokens
   - 該当するトークンを削除/無効化

2. **新しいトークンを作成**：
   - 必要な権限のみを付与
   - 必ず `.gitignore` で設定ファイルを除外

3. **コミット履歴から削除**（オプション）：
   ```bash
   # BFG Repo-Cleanerを使用（推奨）
   # または git filter-branch を使用
   ```

### 方法2: コミット履歴から削除（上級者向け）

BFG Repo-Cleanerを使用：
```bash
# Javaが必要
java -jar bfg.jar --delete-files cursor_github_config.json
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force
```

**注意**: `--force` プッシュは危険です。チームで共有している場合は相談してください。

### 方法3: 設定ファイルを環境変数に移行

```python
# 設定ファイルの代わりに環境変数を使用
import os
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
```

## 🛡️ 今後の予防策

1. **`.gitignore` に追加**（既に実施済み）
   ```
   config/cursor_github_config.json
   **/*token*.json
   **/*secret*.json
   ```

2. **環境変数の使用**
   - 設定ファイルではなく環境変数で管理

3. **GitHub Secrets（リポジトリ設定）**
   - GitHub Actionsなどでは、リポジトリのSecrets機能を使用

4. **定期的なトークンのローテーション**
   - 定期的にトークンを更新

## 📋 現在の状況

✅ 実施済み：
- `backup/` フォルダーを除外
- `config/cursor_github_config.json` を除外
- `.gitignore` を更新

⚠️ 残っている問題：
- 過去のコミット履歴にトークンが含まれている
- これが原因でプッシュがブロックされている

## 🎯 次のステップ

1. **トークンを無効化・再作成**（推奨）
2. または、コミット履歴を書き換えて完全に削除
3. プッシュを再試行

**絶対に許可ボタンを押さないでください。** セキュリティリスクが高すぎます。

