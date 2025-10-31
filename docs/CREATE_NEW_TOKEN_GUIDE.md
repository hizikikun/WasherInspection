# 新しいトークン作成ガイド

## 📝 ステップバイステップ

### 1. 新しいトークンを作成

1. **「Generate new token」ボタンをクリック**
2. **「Generate new token (classic)」を選択**
3. **トークン名を設定**
   - 例：「Cursor GitHub Integration v2」
   - または：「WasherInspection Local Dev」
4. **有効期限を設定（重要！）**
   - ✅ 90日、180日など適切な期間を選択
   - ❌ 「No expiration」は選択しない
5. **必要なスコープのみ選択**
   - `repo`（リポジトリへのアクセス）- 必要
   - その他は必要最小限のみ
6. **「Generate token」をクリック**
7. **表示されたトークンをすぐにコピー**
   - ⚠️ この画面を閉じると、2度と表示されません
   - 安全な場所に一時的に保存

### 2. ローカル設定を更新

トークンをコピーしたら、設定ファイルを更新してください。

設定ファイル：`config/cursor_github_config.json`

```json
{
  "github_token": "ここに新しいトークンを貼り付け",
  "github_owner": "hizikikun",
  "github_repo": "WasherInspection",
  ...
}
```

### 3. 設定ファイルの確認

- ✅ `.gitignore` に `config/cursor_github_config.json` が含まれているか確認
- ✅ トークンがコミットされないことを確認

### 4. プッシュ前のチェック

新しいトークンで設定を更新したら、プッシュを試みてください。

ただし、**コミット履歴にまだ古いトークンが含まれている**ため、
プッシュ時にまたブロックされる可能性があります。

## 🔧 コミット履歴のクリーンアップ

コミット履歴からトークンを削除する方法：

### 方法1: コミットを再作成（簡単）

```bash
git reset --soft origin/main
git commit -m "ファイル整理と文字化け修正完了"
git push
```

これで、トークンを含む古いコミットを新しいコミットに置き換えます。

### 方法2: BFG Repo-Cleaner（完全削除）

より徹底的に削除する場合は、BFG Repo-Cleanerを使用：

1. BFG Repo-Cleanerをダウンロード
2. 実行して履歴から完全に削除

ただし、この方法は複雑なので、まず方法1を試すことをお勧めします。

