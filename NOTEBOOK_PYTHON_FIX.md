# ノートパソコン側 - Python問題の解決方法

`python --version` で「Python」とだけ表示される場合の対処法です。

---

## 🔍 問題の確認

まず、詳しく確認しましょう。

### 確認スクリプトを実行

1. **エクスプローラーでプロジェクトフォルダを開く**

2. **`notebook_python_check.bat` をダブルクリック**

3. **結果を確認**

---

## 🔧 解決方法

### ケース1: Pythonが動作している場合

✅ **`python -c "import sys; print(sys.version)"` でバージョンが表示される**

→ Pythonは動作しています。そのままライブラリをインストールできます:

```bash
python -m pip install mss pillow pyautogui pyperclip cryptography
```

---

### ケース2: Pythonが正しくインストールされていない場合

❌ **Pythonが動作しない、またはエラーが出る**

→ Pythonを再インストールする必要があります。

#### 手順

1. **Pythonをアンインストール**（既にインストールされている場合）
   - コントロールパネル → プログラムと機能
   - Python関連のプログラムをすべてアンインストール

2. **Pythonをダウンロード**
   - https://www.python.org/downloads/ にアクセス
   - 「Download Python」ボタンをクリック

3. **インストール**
   - ダウンロードしたファイルを実行
   - ⚠️ **重要**: 「**Add Python to PATH**」に**必ずチェックを入れる**
   - 「Install Now」をクリック

4. **コマンドプロンプトを再起動**
   - 既に開いているコマンドプロンプトを閉じて、再度開く

5. **確認**:
   ```bash
   python --version
   ```
   - バージョン番号が表示されることを確認（例: `Python 3.13.9`）

---

### ケース3: py ランチャーは使える場合

⚠️ **`py --version` でバージョンが表示されるが、`python --version` が使えない**

→ Pythonはインストールされていますが、PATHが正しく設定されていません。

#### 解決方法1: py コマンドを使用（簡単）

```bash
py -m pip install mss pillow pyautogui pyperclip cryptography
py chrome_like_remote_desktop.py client 192.168.1.100
```

#### 解決方法2: Pythonを再インストール（推奨）

- Pythonを再インストールして、「Add Python to PATH」にチェックを入れる

---

## 📋 次のステップ

Pythonが正しく動作することを確認したら:

### 1. ライブラリをインストール

```bash
python -m pip install --upgrade pip
python -m pip install mss pillow pyautogui pyperclip cryptography
```

### 2. インストール確認

```bash
python -m pip list | findstr "mss pillow pyautogui pyperclip cryptography"
```

### 3. クライアントを起動

```bash
python chrome_like_remote_desktop.py client 192.168.1.100
```
（IPアドレスを変更）

---

## 💡 よくある質問

### Q: 「Python」とだけ表示されるのはなぜ？

A: 以下の可能性があります:
- Pythonが正しくインストールされていない
- PATHの設定に問題がある
- 別のPythonランチャーが干渉している

### Q: py コマンドと python コマンドの違いは？

A: 
- `py`: Python Launcher（複数のPythonバージョンを管理）
- `python`: 直接Pythonを実行

通常は `python` コマンドを使いますが、`py` でも動作します。

---

## 🎯 まとめ

1. **確認**: `notebook_python_check.bat` を実行
2. **問題がある場合**: Pythonを再インストール（「Add Python to PATH」にチェック）
3. **正常な場合**: ライブラリをインストールしてクライアントを起動

---

**まずは `notebook_python_check.bat` を実行して、詳しい状態を確認してください！**



