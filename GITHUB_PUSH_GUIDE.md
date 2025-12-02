# GitHubへのPush手順ガイド

久しぶりにGitHubにpushする際の、わかりやすい手順をまとめました。

## 📋 基本的なPush手順

### ステップ1: 現在の状態を確認

まず、変更があるかどうかを確認します：

```bash
git status
```

**表示される内容：**
- `Changes not staged for commit`: 変更されたファイルがある（まだステージングされていない）
- `Changes to be committed`: ステージング済みのファイルがある
- `nothing to commit, working tree clean`: 変更なし

### ステップ2: 変更をステージング（追加）

変更したファイルをGitに追加します：

```bash
# すべての変更を追加
git add .

# または、特定のファイルだけ追加
git add ファイル名
```

**確認方法：**
```bash
git status
```
→ `Changes to be committed` に表示されていればOK

### ステップ3: コミット（変更を記録）

変更内容をコミットメッセージと一緒に記録します：

```bash
git commit -m "変更内容の説明"
```

**コミットメッセージの例：**
- `"機能追加: 新しい機能を実装"`
- `"バグ修正: エラーを修正"`
- `"更新: ドキュメントを更新"`
- `"リファクタリング: コードを整理"`

### ステップ4: GitHubにPush（アップロード）

ローカルの変更をGitHubに送信します：

```bash
# メインブランチが 'main' の場合
git push origin main

# メインブランチが 'master' の場合
git push origin master
```

**ブランチ名の確認方法：**
```bash
git branch
```
→ 現在のブランチ名が `*` で表示されます

---

## 🎯 よくあるシナリオ

### シナリオ1: 初めてPushする場合

1. **リモートリポジトリが設定されているか確認**
   ```bash
   git remote -v
   ```
   → 何も表示されない場合は、リモートリポジトリを追加する必要があります

2. **リモートリポジトリを追加**（GitHubでリポジトリを作成済みの場合）
   ```bash
   git remote add origin https://github.com/ユーザー名/リポジトリ名.git
   ```

3. **通常のPush手順を実行**

### シナリオ2: エラーが出た場合

#### エラー: "Updates were rejected"

**原因：** GitHubの方が新しい変更を持っている

**解決方法：**
```bash
# 1. まずGitHubの最新版を取得
git pull origin main

# 2. 競合があれば解決してから
git add .
git commit -m "マージ: GitHubの変更を取り込み"

# 3. 再度Push
git push origin main
```

#### エラー: "Authentication failed"

**原因：** GitHubへの認証情報が正しくない

**解決方法：**
1. Personal Access Token (PAT) を使用する
2. GitHubでトークンを生成：
   - Settings → Developer settings → Personal access tokens → Tokens (classic)
   - `repo` 権限を付与
3. Push時にトークンをパスワードとして入力

#### エラー: "Branch not found"

**原因：** ブランチ名が間違っている

**解決方法：**
```bash
# 現在のブランチ名を確認
git branch

# 正しいブランチ名でPush
git push origin ブランチ名
```

---

## 🚀 簡単な方法（GUIアプリ使用）

プロジェクトには `change_history_viewer.py` というGUIアプリがあります：

1. **アプリを起動**
   ```bash
   python change_history_viewer.py
   ```

2. **「コミット＆プッシュ」ボタンをクリック**
   - 自動的に変更を検出
   - ステージング
   - コミット
   - Push

---

## 📝 コマンド一覧（まとめ）

```bash
# 1. 状態確認
git status

# 2. 変更を追加
git add .

# 3. コミット
git commit -m "変更内容"

# 4. Push
git push origin main
```

**すべて一度に実行（変更がある場合のみ）：**
```bash
git add . && git commit -m "更新" && git push origin main
```

---

## ⚠️ 注意事項

1. **コミットメッセージはわかりやすく**
   - 何を変更したかが分かるように
   - 日本語でもOK

2. **Push前に確認**
   - `git status` で変更内容を確認
   - 間違ったファイルを追加していないか確認

3. **重要な変更はバックアップ**
   - Push前に重要なファイルはバックアップを取る

4. **共同作業の場合**
   - Push前に `git pull` で最新版を取得
   - 競合を解決してからPush

---

## 🔍 トラブルシューティング

### 変更が表示されない

```bash
# .gitignoreで除外されていないか確認
git check-ignore -v ファイル名

# すべてのファイル（.gitignoreの対象も含む）を確認
git status --ignored
```

### コミット履歴を確認

```bash
# 最近のコミットを確認
git log --oneline -10

# 詳細な履歴
git log
```

### リモートリポジトリの確認

```bash
# リモートリポジトリ一覧
git remote -v

# リモートブランチ一覧
git branch -r
```

---

## 💡 便利なTips

### 変更内容を確認してからPush

```bash
# 変更内容を確認
git diff

# ステージング済みの変更を確認
git diff --staged
```

### コミットメッセージを後から変更

```bash
# 最後のコミットメッセージを変更
git commit --amend -m "新しいメッセージ"

# 変更をPush（既にPush済みの場合）
git push origin main --force
```

⚠️ **注意：** `--force` は履歴を上書きするので、共同作業の場合は注意が必要です。

---

## 📞 さらに詳しく知りたい場合

- `CURSOR_GITHUB_GUIDE.md`: CursorとGitHubの連携方法
- `GITHUB_AUTO_COMMIT_GUIDE.md`: 自動コミット機能の使い方
- `COMMIT_FIX_GUIDE.md`: コミットメッセージの修正方法

---

**最後に：** 困ったときは `git status` で現在の状態を確認することから始めましょう！

