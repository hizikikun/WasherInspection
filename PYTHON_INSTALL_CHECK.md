# Pythonがインストールされているか確認する方法

「プログラムと機能」にPythonが表示されない場合の確認方法です。

---

## 🔍 確認方法

### 方法1: コマンドプロンプトで確認（最も確実）

1. **コマンドプロンプトを開く**
   - Windowsキー + R → 「cmd」と入力 → Enter

2. **以下を実行**:
   ```bash
   python --version
   ```

3. **結果を確認**:
   - ✅ **Pythonのバージョンが表示される場合**（例: `Python 3.12.0`）
     → Pythonはインストールされています
   - ❌ **「'python' は、内部コマンドまたは外部コマンド...」と表示される場合**
     → Pythonがインストールされていないか、PATHに追加されていません

4. **py ランチャーも確認**:
   ```bash
   py --version
   ```
   - これでバージョンが表示される場合、Pythonはインストールされていますが、`python`コマンドが使えない状態です

---

### 方法2: 確認スクリプトを実行（簡単）

1. **エクスプローラーでプロジェクトフォルダを開く**
   - `C:\Users\西村康成\WasherInspection`

2. **`check_python_installed.bat` をダブルクリック**

3. **結果を確認**
   - ✅ または ❌ の表示で、Pythonの状態が分かります

---

### 方法3: Microsoft Storeを確認

1. **Microsoft Storeを開く**
   - Windowsキー → 「Microsoft Store」と入力

2. **「ライブラリ」または「マイライブラリ」をクリック**

3. **Pythonがインストールされているか確認**
   - Microsoft Store版のPythonは「プログラムと機能」に表示されない場合があります

---

## 📊 結果に応じた次のステップ

### ケース1: Pythonがインストールされている場合

✅ **`python --version` でバージョンが表示される**

→ そのまま使用できます。必要なライブラリをインストールしてください:
```bash
python -m pip install mss pillow pyautogui pyperclip cryptography
```

---

### ケース2: py ランチャーのみ使える場合

⚠️ **`py --version` でバージョンが表示されるが、`python --version` が使えない**

→ Pythonはインストールされていますが、PATHが正しく設定されていません。

**解決方法1: py コマンドを使用**
```bash
py -m pip install mss pillow pyautogui pyperclip cryptography
```

**解決方法2: Pythonを再インストール**
- Pythonを再インストールして、「Add Python to PATH」にチェックを入れる

---

### ケース3: Pythonがインストールされていない場合

❌ **`python --version` も `py --version` も使えない**

→ Pythonをインストールする必要があります。

**インストール手順**:

1. **Pythonをダウンロード**
   - https://www.python.org/downloads/ にアクセス
   - 「Download Python」ボタンをクリック

2. **インストール**
   - ダウンロードしたファイルを実行
   - ⚠️ **重要**: 「**Add Python to PATH**」に**必ずチェックを入れる**
   - 「Install Now」をクリック

3. **確認**
   - コマンドプロンプトを**再起動**（閉じて再度開く）
   - `python --version` で確認

4. **ライブラリをインストール**
   ```bash
   python -m pip install mss pillow pyautogui pyperclip cryptography
   ```

---

## 🔍 インストール場所の確認

Pythonがインストールされているか、フォルダで確認する方法:

1. **エクスプローラーで以下を開く**:
   ```
   C:\Users\西村康成\AppData\Local\Programs\Python
   ```

   ⚠️ **AppDataフォルダが表示されない場合**:
   - エクスプローラーの「表示」タブ → 「隠しファイル」にチェック

2. **Pythonフォルダが存在する場合**
   - Pythonはインストールされています
   - ただし、PATHに追加されていない可能性があります

---

## 💡 よくある質問

### Q: 「プログラムと機能」にPythonが表示されないのはなぜ？

A: 以下の理由が考えられます:
- Pythonがインストールされていない
- Microsoft Store版のPythonを使用している（「プログラムと機能」に表示されない場合がある）
- インストール方法が異なる（portable版など）

### Q: コマンドプロンプトで確認する方法が最も確実なのはなぜ？

A: 「プログラムと機能」は、Windowsインストーラーでインストールされたプログラムのみを表示します。Microsoft Store版や、手動でインストールしたPythonは表示されない場合があります。コマンドプロンプトで実際にコマンドが使えるかどうかを確認するのが最も確実です。

---

## 🎯 まとめ

1. **まず確認**: `python --version` または `check_python_installed.bat` で確認
2. **インストールされていない場合**: Pythonをインストール（「Add Python to PATH」にチェック）
3. **インストールされている場合**: 必要なライブラリをインストール

---

**まずは `check_python_installed.bat` を実行して、Pythonの状態を確認してください！**



