# Python アンインストール・再インストール完全ガイド

Pythonを完全にアンインストールして、再インストールする手順を詳しく説明します。

---

## 📋 目次

1. [Pythonのアンインストール](#pythonのアンインストール)
2. [残存ファイルの削除](#残存ファイルの削除)
3. [Pythonの再インストール](#pythonの再インストール)
4. [インストール確認](#インストール確認)
5. [トラブルシューティング](#トラブルシューティング)

---

## 🗑️ Pythonのアンインストール

### ステップ1: コントロールパネルからアンインストール

1. **Windowsキー + R** を押す
2. **「appwiz.cpl」と入力してEnter**を押す
   - または: 設定 → アプリ → アプリと機能

3. **「プログラムと機能」または「アプリと機能」が開きます**

4. **Python関連のプログラムを探す**
   - 以下のような項目を探してください:
     - `Python 3.11.x`（バージョン番号は異なる場合があります）
     - `Python 3.10.x`
     - `Python Launcher`
     - `Python 3.x.x (64-bit)`
     - `Python 3.x.x (32-bit)`

5. **すべてのPython関連プログラムをアンインストール**
   - 各項目を右クリック → **「アンインストール」**をクリック
   - 確認ダイアログで「はい」をクリック
   - ⚠️ **すべてのPython関連プログラムを削除してください**

6. **アンインストールが完了するまで待つ**

---

### ステップ2: Microsoft Store版のPythonを確認

1. **Microsoft Storeを開く**
   - Windowsキー → 「Microsoft Store」と入力

2. **「ライブラリ」または「マイライブラリ」をクリック**

3. **Pythonがインストールされている場合**
   - Pythonをクリック
   - 「アンインストール」をクリック

---

## 🧹 残存ファイルの削除

### ステップ1: Pythonフォルダの削除

1. **エクスプローラーを開く**（Windowsキー + E）

2. **以下のフォルダを確認して削除**（存在する場合）:

   ```
   C:\Users\西村康成\AppData\Local\Programs\Python
   ```

3. **以下のフォルダも確認**:
   ```
   C:\Python3x
   C:\Python3x-32
   ```
   - （xはバージョン番号、存在する場合のみ）

4. **フォルダを削除**:
   - フォルダを右クリック → **「削除」**をクリック
   - 確認ダイアログで「はい」をクリック

---

### ステップ2: AppDataフォルダの確認

1. **エクスプローラーで以下を開く**:
   ```
   C:\Users\西村康成\AppData\Local
   ```

   ⚠️ **AppDataフォルダが表示されない場合**:
   - エクスプローラーの「表示」タブ → 「隠しファイル」にチェック

2. **以下のフォルダを確認**:
   ```
   C:\Users\西村康成\AppData\Local\Programs\Python
   C:\Users\西村康成\AppData\Local\pip
   C:\Users\西村康成\AppData\Roaming\Python
   ```

3. **存在する場合は削除**

---

### ステップ3: 環境変数PATHの確認（上級者向け）

1. **Windowsキー + R** → **「sysdm.cpl」と入力** → Enter

2. **「詳細設定」タブをクリック**

3. **「環境変数」ボタンをクリック**

4. **「システム環境変数」の「Path」を選択** → **「編集」をクリック**

5. **Python関連のパスを探す**（以下のような項目）:
   - `C:\Python3x\`
   - `C:\Python3x\Scripts\`
   - `C:\Users\西村康成\AppData\Local\Programs\Python\Python3x\`
   - `C:\Users\西村康成\AppData\Local\Programs\Python\Python3x\Scripts\`

6. **Python関連のパスを削除**:
   - 項目を選択 → **「削除」をクリック**

7. **「OK」をクリックして閉じる**

---

### ステップ4: レジストリの確認（上級者向け・注意が必要）

⚠️ **レジストリの編集は注意が必要です。自信がない場合はスキップしてください。**

1. **Windowsキー + R** → **「regedit」と入力** → Enter

2. **以下のパスを確認**:
   ```
   HKEY_CURRENT_USER\Software\Python
   HKEY_LOCAL_MACHINE\SOFTWARE\Python
   ```

3. **存在する場合は削除**（右クリック → 削除）

4. **レジストリエディタを閉じる**

---

## 🔄 Pythonの再インストール

### ステップ1: Pythonをダウンロード

1. **ブラウザで以下にアクセス**:
   ```
   https://www.python.org/downloads/
   ```

2. **「Download Python」ボタンをクリック**
   - 最新バージョンが自動的にダウンロードされます
   - 推奨: Python 3.11.x または 3.12.x

3. **ダウンロードが完了するまで待つ**

---

### ステップ2: Pythonをインストール

1. **ダウンロードしたファイルを実行**（例: `python-3.12.0-amd64.exe`）

2. **インストーラーが開きます**

3. **⚠️ 重要: 以下のオプションに必ずチェックを入れる**:
   - ✅ **「Add Python to PATH」**（最も重要！）
   - ✅ **「Install launcher for all users」**（推奨）

4. **「Install Now」をクリック**
   - または「Customize installation」を選択して詳細設定

5. **インストールが完了するまで待つ**（数分かかります）

6. **「Setup was successful」と表示されたら「Close」をクリック**

---

### ステップ3: インストール後の確認

1. **コマンドプロンプトを開く**
   - Windowsキー + R → 「cmd」と入力 → Enter
   - ⚠️ **重要**: 新しいコマンドプロンプトを開く（既に開いている場合は閉じて再度開く）

2. **Pythonのバージョンを確認**:
   ```bash
   python --version
   ```
   
   **期待される結果**:
   ```
   Python 3.12.0
   ```
   （バージョン番号は異なる場合があります）

3. **pipのバージョンを確認**:
   ```bash
   pip --version
   ```
   
   **期待される結果**:
   ```
   pip 24.0 from C:\Users\西村康成\AppData\Local\Programs\Python\Python3x\lib\site-packages\pip (python 3.12)
   ```

---

## ✅ インストール確認

### ステップ1: Pythonが正しく動作するか確認

1. **コマンドプロンプトで以下を実行**:
   ```bash
   python -c "print('Hello, Python!')"
   ```
   
   **期待される結果**:
   ```
   Hello, Python!
   ```

2. **エラーが出ないことを確認**

---

### ステップ2: 必要なライブラリをインストール

1. **コマンドプロンプトで以下を実行**:
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Chrome風リモートデスクトップに必要なライブラリをインストール**:
   ```bash
   python -m pip install mss pillow pyautogui pyperclip cryptography
   ```

3. **インストールが完了するまで待つ**

4. **インストール確認**:
   ```bash
   python -m pip list
   ```
   
   - 以下のライブラリが表示されることを確認:
     - mss
     - Pillow
     - pyautogui
     - pyperclip
     - cryptography

---

### ステップ3: セットアップ確認スクリプトを実行

1. **プロジェクトフォルダに移動**:
   ```bash
   cd "C:\Users\西村康成\WasherInspection"
   ```

2. **セットアップ確認を実行**:
   ```bash
   python check_chrome_like_setup.py
   ```
   
   または
   ```bash
   check_setup.bat
   ```

3. **すべてのチェックが✅になることを確認**

---

## 🔧 トラブルシューティング

### 問題1: 「python は、内部コマンドまたは外部コマンド...」と表示される

#### 原因
- PATH環境変数にPythonが追加されていない
- コマンドプロンプトを再起動していない

#### 解決方法

1. **コマンドプロンプトを完全に閉じて、再度開く**

2. **それでも解決しない場合**:
   - Pythonを再インストール
   - インストール時に「Add Python to PATH」に**必ずチェックを入れる**

3. **環境変数を手動で設定**（上級者向け）:
   - Windowsキー + R → 「sysdm.cpl」→ 環境変数
   - 「システム環境変数」の「Path」を編集
   - 以下を追加:
     ```
     C:\Users\西村康成\AppData\Local\Programs\Python\Python3x\
     C:\Users\西村康成\AppData\Local\Programs\Python\Python3x\Scripts\
     ```
   - （3xは実際のバージョン番号に置き換える）

---

### 問題2: pipが使えない

#### 解決方法

1. **python -m pip を使用**:
   ```bash
   python -m pip install mss pillow pyautogui pyperclip cryptography
   ```

2. **pipを再インストール**:
   ```bash
   python -m ensurepip --upgrade
   ```

---

### 問題3: 複数のPythonバージョンがインストールされている

#### 解決方法

1. **すべてのPythonをアンインストール**（上記の手順を参照）

2. **再インストール時は1つのバージョンのみインストール**

3. **py launcherを使用**:
   ```bash
   py --version
   py -m pip install mss pillow pyautogui pyperclip cryptography
   ```

---

### 問題4: インストール時にエラーが出る

#### 解決方法

1. **管理者権限で実行**:
   - インストーラーを右クリック → 「管理者として実行」

2. **ウイルス対策ソフトを一時的に無効化**

3. **Windows Updateを実行**してから再試行

4. **別のインストール方法を試す**:
   - Microsoft StoreからPythonをインストール（ただし、PATHの設定が必要な場合があります）

---

## 📝 チェックリスト

### アンインストール

- [ ] コントロールパネルからPythonをアンインストール
- [ ] Microsoft StoreからPythonをアンインストール（該当する場合）
- [ ] Pythonフォルダを削除
- [ ] AppDataフォルダのPython関連ファイルを削除
- [ ] 環境変数PATHからPython関連のパスを削除

### 再インストール

- [ ] Pythonをダウンロード
- [ ] 「Add Python to PATH」にチェックを入れてインストール
- [ ] コマンドプロンプトを再起動
- [ ] `python --version` でバージョン確認
- [ ] `pip --version` でpip確認
- [ ] 必要なライブラリをインストール
- [ ] セットアップ確認スクリプトを実行

---

## 💡 推奨事項

1. **最新の安定版を使用**
   - Python 3.11.x または 3.12.x を推奨

2. **「Add Python to PATH」を必ずチェック**
   - これがないとコマンドプロンプトからPythonを使えません

3. **管理者権限でインストール**
   - システム全体でPythonを使えるようにするため

4. **インストール後は必ずコマンドプロンプトを再起動**
   - 環境変数の変更を反映させるため

---

## 🎯 まとめ

1. **アンインストール**: コントロールパネル + フォルダ削除
2. **再インストール**: 公式サイトからダウンロード + 「Add Python to PATH」にチェック
3. **確認**: `python --version` と `pip --version` で確認
4. **ライブラリインストール**: 必要なライブラリをインストール
5. **セットアップ確認**: `check_setup.bat` を実行

---

**問題が解決しない場合は、エラーメッセージを確認して、上記のトラブルシューティングを参照してください。**




