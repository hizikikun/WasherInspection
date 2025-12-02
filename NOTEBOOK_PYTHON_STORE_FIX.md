# ノートパソコン側 - Microsoft Store版Pythonの問題解決

`where python` で `WindowsApps\python.exe` が表示される場合の対処法です。

---

## 🔍 問題の原因

`C:\Users\kouse\AppData\Local\Microsoft\WindowsApps\python.exe` は**Microsoft Store版のPythonスタブ**です。

このスタブは、実際のPythonがインストールされていない場合、Microsoft Storeにリダイレクトするだけのファイルです。そのため、バージョン番号が表示されません。

---

## ✅ 解決方法: 公式サイトからPythonをインストール

### ステップ1: Microsoft Store版をアンインストール（推奨）

1. **Microsoft Storeを開く**
   - Windowsキー → 「Microsoft Store」と入力

2. **「ライブラリ」または「マイライブラリ」をクリック**

3. **Pythonを探してアンインストール**
   - Pythonが表示されていたら、アンインストールをクリック

### ステップ2: 公式サイトからPythonをダウンロード

1. **ブラウザで以下にアクセス**:
   ```
   https://www.python.org/downloads/
   ```

2. **「Download Python」ボタンをクリック**
   - 最新バージョンが自動的にダウンロードされます
   - 推奨: Python 3.11.x または 3.12.x

3. **ダウンロードが完了するまで待つ**

### ステップ3: Pythonをインストール

1. **ダウンロードしたファイルを実行**（例: `python-3.12.0-amd64.exe`）

2. **インストーラーが開きます**

3. **⚠️ 重要: 以下のオプションに必ずチェックを入れる**:
   - ✅ **「Add Python to PATH」**（最も重要！）
   - ✅ **「Install launcher for all users」**（推奨）

4. **「Install Now」をクリック**
   - または「Customize installation」を選択して詳細設定

5. **インストールが完了するまで待つ**（数分かかります）

6. **「Setup was successful」と表示されたら「Close」をクリック**

### ステップ4: コマンドプロンプトを再起動

⚠️ **重要**: 既に開いているコマンドプロンプトを**完全に閉じて**、再度開いてください。

環境変数の変更を反映させるため、再起動が必要です。

### ステップ5: インストール確認

1. **新しいコマンドプロンプトを開く**

2. **以下を実行**:
   ```bash
   python --version
   ```

3. **期待される結果**:
   ```
   Python 3.12.0
   ```
   （バージョン番号は異なる場合があります）

4. **パスを確認**:
   ```bash
   where python
   ```

5. **期待される結果**:
   ```
   C:\Users\kouse\AppData\Local\Programs\Python\Python3x\python.exe
   ```
   または
   ```
   C:\Python3x\python.exe
   ```
   （`WindowsApps` ではなく、実際のPythonフォルダを指していることを確認）

---

## 📦 次のステップ: ライブラリをインストール

Pythonが正しくインストールされたことを確認したら:

1. **pipをアップグレード**:
   ```bash
   python -m pip install --upgrade pip
   ```

2. **必要なライブラリをインストール**:
   ```bash
   python -m pip install mss pillow pyautogui pyperclip cryptography
   ```

3. **インストール確認**:
   ```bash
   python -m pip list | findstr "mss pillow pyautogui pyperclip cryptography"
   ```

---

## 🔧 トラブルシューティング

### 問題1: インストール後も「Python」とだけ表示される

#### 解決方法

1. **コマンドプロンプトを完全に閉じて、再度開く**

2. **環境変数を確認**:
   - Windowsキー + R → 「sysdm.cpl」と入力 → Enter
   - 「詳細設定」→「環境変数」
   - 「システム環境変数」の「Path」を確認
   - `C:\Users\kouse\AppData\Local\Programs\Python\Python3x\` が含まれているか確認
   - `WindowsApps` のパスが上にある場合、Pythonのパスを上に移動

3. **Pythonを再インストール**
   - アンインストールしてから、再度インストール

---

### 問題2: 複数のPythonが検出される

#### 解決方法

1. **すべてのPythonをアンインストール**
   - コントロールパネル → プログラムと機能
   - Microsoft Store → ライブラリ

2. **公式サイトから1つだけインストール**

---

### 問題3: インストール時にエラーが出る

#### 解決方法

1. **管理者権限で実行**
   - インストーラーを右クリック → 「管理者として実行」

2. **ウイルス対策ソフトを一時的に無効化**

3. **Windows Updateを実行**してから再試行

---

## 📝 チェックリスト

- [ ] Microsoft Store版のPythonをアンインストールした
- [ ] 公式サイトからPythonをダウンロードした
- [ ] 「Add Python to PATH」にチェックを入れてインストールした
- [ ] コマンドプロンプトを再起動した
- [ ] `python --version` でバージョン番号が表示されることを確認した
- [ ] `where python` で正しいパスが表示されることを確認した（WindowsAppsではない）

---

## 🎯 まとめ

1. **Microsoft Store版をアンインストール**
2. **公式サイトからダウンロード**
3. **「Add Python to PATH」にチェックを入れてインストール**
4. **コマンドプロンプトを再起動**
5. **`python --version` で確認**

---

**インストール後、`python --version` でバージョン番号が表示されることを確認してください！**



