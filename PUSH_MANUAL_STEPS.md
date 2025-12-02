# GitHubへのPush手順（手動実行）

## 方法1: バッチファイルを使用（最も簡単）

1. **`push_to_github.bat`** をダブルクリック
2. 自動的に現在のブランチを検出してpushします

## 方法2: PowerShellスクリプトを使用

1. PowerShellを開く
2. プロジェクトフォルダに移動：
   ```powershell
   cd C:\Users\西村康成\WasherInspection
   ```
3. スクリプトを実行：
   ```powershell
   powershell -ExecutionPolicy Bypass -File push_to_github.ps1
   ```

## 方法3: 手動でコマンドを実行

### ステップ1: 現在のブランチを確認

```bash
git branch
```

現在のブランチ名が `*` で表示されます。

### ステップ2: リモートブランチを確認

```bash
git branch -r
```

リモートにどのブランチが存在するか確認します。

### ステップ3: Pushを実行

**現在のブランチが `master` の場合：**
```bash
git push -u origin master
```

**現在のブランチが `main` の場合：**
```bash
git push -u origin main
```

**初めてpushする場合（リモートブランチが存在しない）：**
```bash
# 現在のブランチ名を確認してから
git push -u origin ブランチ名
```

`-u` オプションは初めてpushする場合に必要です。これにより、ローカルブランチとリモートブランチが関連付けられます。

## エラーが出た場合

### エラー: "src refspec main does not match any"

**原因：** リモートにそのブランチが存在しない、またはローカルブランチ名が異なる

**解決方法：**
1. 現在のブランチ名を確認：
   ```bash
   git branch
   ```
2. そのブランチ名でpush：
   ```bash
   git push -u origin ブランチ名
   ```

### エラー: "Authentication failed"

**解決方法：**
1. GitHubでPersonal Access Tokenを作成
2. Push時にユーザー名とトークンを入力

### エラー: "Updates were rejected"

**解決方法：**
```bash
# 1. 最新版を取得
git pull origin ブランチ名

# 2. 再度push
git push origin ブランチ名
```

## よくある質問

**Q: どのブランチにpushすればいいですか？**
A: `git branch` で確認した現在のブランチ（`*` が付いているもの）にpushしてください。

**Q: `-u` オプションは必要ですか？**
A: 初めてpushする場合は必要です。2回目以降は不要です。

**Q: エラーが出ても続行できますか？**
A: エラーメッセージを確認して、上記の解決方法を試してください。

