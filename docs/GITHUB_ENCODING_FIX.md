# GitHub文字化け修正ガイド

## 問題の概要

過去に作成されたGitHubコミットメッセージやファイル名が文字化けしている場合があります。これは、Gitのエンコーディング設定やコミット作成時のエンコーディング処理が不適切だったことが原因です。

## 修正内容

### 1. GitHub APIへのリクエスト修正

以下のファイルで、GitHub APIへのリクエスト時に明示的にUTF-8エンコーディングを使用するように修正しました：

- `github_tools/github_autocommit.py`
- `github_tools/cursor_integration.py`
- `github_tools/auto_sync.py`

**変更内容**:
- `requests.post(..., json=data)` を `requests.post(..., data=json.dumps(data, ensure_ascii=False).encode('utf-8'))` に変更
- `Content-Type` ヘッダーに `charset=utf-8` を明示的に指定

### 2. コミットメッセージ生成の修正

`github_autocommit.py` の `generate_commit_message` メソッドを修正し、不要な encode/decode 処理を削除しました。

**変更前**:
```python
message = message.encode('utf-8').decode('utf-8')
```

**変更後**:
```python
# Message is already a UTF-8 string, no need to encode/decode
# Just ensure it's valid Unicode
if not isinstance(message, str):
    message = str(message)
```

### 3. Git設定

以下のGit設定が推奨されまム（`scripts/fix_git_encoding.py`で設定可能）：

```bash
git config --global i18n.commitencoding utf-8
git config --global i18n.logoutputencoding utf-8
git config --global core.quotepath false
```

### 4. 安全なコミット方法

新しいコミットを作成する際は、`scripts/safe_git_commit.py` を使用することを推奨します：

```bash
python scripts/safe_git_commit.py "コミットメッセージ" [ファイル1] [ファイル2] ...
```

このスクリプトは、GitPythonを直接使用してUTF-8エンコーディングを保証します。

## 過去のコミットメッセージの修正

過去のコミットメッセージを修正するには、`scripts/fix_all_github_commits.py` を使用できまム。ただし、これは既存のコミット履歴を書き換えるため、共有リポジトリで使用する場合は注意が必要です。

### 使用方法

```bash
python scripts/fix_all_github_commits.py
```

**注意**: 
- このスクリプトは `git filter-branch` を使用してコミット履歴を書き換えまム
- 共有リポジトリで使用する場合は、チームメンバーと相談してください
- 修正後は `git push --force` が必要になりまム

## 文字化けファイルの削除

文字化けしているファイル名（例: `ResinWasherInspection.spec` が文字化けしたファイル）は、以下のコマンドで削除できまム：

```powershell
# PowerShellで実行
Get-ChildItem *.spec | Where-Object { $_.Name.Length -gt 30 } | Remove-Item -Force
```

## 今後の予防策

1. **常に `scripts/safe_git_commit.py` を使用する** - これはUTF-8エンコーディングを保証します
2. **Git設定を確認する** - `git config --list | Select-String encoding` で確認
3. **コミット前にエンコーディングを確認する** - 特に日本語を含むコミットメッセージ

## 確認方法

修正が正しく適用されたか確認するには：

```bash
# 最新のコミットメッセージを確認
git log --pretty=format:"%h|%s" -5

# Pythonから直接確認（UTF-8表示）
python -c "import git; repo = git.Repo('.'); [print(f'{c.hexsha[:7]}|{c.message.strip()}') for c in list(repo.iter_commits('HEAD', max_count=5))]"
```

## トラブルシューティング

### まだ文字化けしている場合

1. **PowerShellのエンコーディング設定を確認**:
   ```powershell
   [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
   chcp 65001
   ```

2. **Gitのエンコーディング設定を再確認**:
   ```bash
   git config --global --get i18n.commitencoding
   git config --global --get i18n.logoutputencoding
   ```

3. **コミット作成時に明示的にUTF-8を使用**:
   ```bash
   $env:PYTHONIOENCODING = "utf-8"
   python scripts/safe_git_commit.py "コミットメッセージ"
   ```

