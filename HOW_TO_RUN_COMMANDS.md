# コマンドの実行方法

## 📍 コマンドはどこで実行する？

コマンドは **コマンドプロンプト** または **PowerShell** で実行します。

---

## 🖥️ コマンドプロンプトの開き方

### 方法1: スタートメニューから開く（簡単）

1. **Windowsキー**を押す（またはスタートボタンをクリック）
2. 「**cmd**」と入力
3. 「**コマンドプロンプト**」をクリック

### 方法2: ファイル名を指定して実行

1. **Windowsキー + R** を押す
2. 「**cmd**」と入力
3. **Enterキー**を押す

### 方法3: エクスプローラーから開く

1. エクスプローラーで `WasherInspection` フォルダを開く
2. アドレスバーに「**cmd**」と入力してEnterキーを押す
3. そのフォルダでコマンドプロンプトが開きます

### 方法4: 右クリックメニューから開く

1. エクスプローラーで `WasherInspection` フォルダを開く
2. フォルダ内の空白部分を**Shiftキーを押しながら右クリック**
3. 「**PowerShellウィンドウをここで開く**」または「**コマンドプロンプトをここで開く**」を選択

---

## 💻 PowerShellの開き方

### 方法1: スタートメニューから開く

1. **Windowsキー**を押す
2. 「**powershell**」と入力
3. 「**Windows PowerShell**」をクリック

### 方法2: エクスプローラーから開く

1. エクスプローラーで `WasherInspection` フォルダを開く
2. アドレスバーに「**powershell**」と入力してEnterキーを押す

---

## 📂 プロジェクトフォルダに移動する方法

コマンドプロンプトまたはPowerShellが開いたら、まずプロジェクトフォルダに移動します。

### 方法1: cdコマンドで移動

```bash
cd C:\Users\西村康成\WasherInspection
```

### 方法2: エクスプローラーから開いた場合

エクスプローラーから開いた場合は、すでにそのフォルダにいるので移動不要です。

---

## 🎯 実際の手順（デスクトップ側）

### ステップ1: コマンドプロンプトを開く

1. **Windowsキー + R** を押す
2. 「**cmd**」と入力してEnterキーを押す

### ステップ2: プロジェクトフォルダに移動

コマンドプロンプトに以下を入力してEnterキーを押す:

```bash
cd C:\Users\西村康成\WasherInspection
```

**表示例:**
```
C:\Users\西村康成> cd C:\Users\西村康成\WasherInspection
C:\Users\西村康成\WasherInspection>
```

### ステップ3: コマンドを実行

プロンプトが `C:\Users\西村康成\WasherInspection>` になっていることを確認してから、コマンドを実行します。

**例: ライブラリをインストール**
```bash
pip install mss pillow pyautogui
```

**例: サーバーを起動**
```bash
python custom_remote_desktop.py server
```

---

## 🎯 実際の手順（ノートパソコン側）

### ステップ1: コマンドプロンプトを開く

1. **Windowsキー + R** を押す
2. 「**cmd**」と入力してEnterキーを押す

### ステップ2: プロジェクトフォルダに移動

```bash
cd C:\Users\西村康成\WasherInspection
```

### ステップ3: コマンドを実行

**例: クライアントを起動**
```bash
python custom_remote_desktop.py client 192.168.1.100
```

---

## 📸 画面の見え方

### コマンドプロンプトの画面

```
Microsoft Windows [Version 10.0.26200]
(c) Microsoft Corporation. All rights reserved.

C:\Users\西村康成> cd C:\Users\西村康成\WasherInspection

C:\Users\西村康成\WasherInspection> pip install mss pillow pyautogui
Collecting mss
  Downloading mss-9.0.1-py3-none-any.whl
...
Successfully installed mss-9.0.1 pillow-10.1.0 pyautogui-0.9.54

C:\Users\西村康成\WasherInspection> python custom_remote_desktop.py server
=== リモートデスクトップサーバー ===
デスクトップ側で実行してください
クライアントの接続を待機しています...
[サーバー] 待機中: 0.0.0.0:8888
```

---

## 🔍 現在のフォルダを確認する方法

コマンドプロンプトで現在いるフォルダを確認するには:

```bash
cd
```

または

```bash
echo %CD%
```

**表示例:**
```
C:\Users\西村康成\WasherInspection
```

---

## 📁 フォルダ内のファイルを確認する方法

現在のフォルダにあるファイルを確認するには:

```bash
dir
```

または

```bash
ls
```

**表示例:**
```
 Volume in drive C has no label.
 Volume Serial Number is XXXX-XXXX

 Directory of C:\Users\西村康成\WasherInspection

2024/01/01  10:00    <DIR>          .
2024/01/01  10:00    <DIR>          ..
2024/01/01  10:00             1,234 custom_remote_desktop.py
2024/01/01  10:00               567 start_custom_remote_server.bat
...
```

---

## ⚠️ よくあるエラーと対処法

### エラー1: 「'python' は、内部コマンドまたは外部コマンド、操作可能なプログラムまたはバッチ ファイルとして認識されていません。」

**原因:** Pythonがインストールされていない、またはPATHに追加されていない

**対処法:**
1. Pythonがインストールされているか確認
2. インストールされている場合は、フルパスで実行:
   ```bash
   C:\Users\西村康成\AppData\Local\Programs\Python\Python3XX\python.exe custom_remote_desktop.py server
   ```

### エラー2: 「指定されたパスが見つかりません。」

**原因:** フォルダパスが間違っている

**対処法:**
1. エクスプローラーで `WasherInspection` フォルダを開く
2. アドレスバーのパスをコピー
3. コマンドプロンプトで `cd ` の後に貼り付け

### エラー3: 「'pip' は、内部コマンドまたは外部コマンド...」

**原因:** pipがインストールされていない、またはPATHに追加されていない

**対処法:**
1. Pythonを再インストール（「Add Python to PATH」にチェック）
2. または、`python -m pip install mss pillow pyautogui` を実行

---

## 💡 簡単な方法（バッチファイル使用）

コマンドを入力するのが面倒な場合は、**バッチファイルをダブルクリック**するだけです。

### デスクトップ側:
1. エクスプローラーで `WasherInspection` フォルダを開く
2. `start_custom_remote_server.bat` を**ダブルクリック**

### ノートパソコン側:
1. エクスプローラーで `WasherInspection` フォルダを開く
2. `start_custom_remote_client.bat` を**ダブルクリック**
3. IPアドレスを入力

---

## 📝 まとめ

1. **コマンドプロンプトを開く**: Windowsキー + R → 「cmd」と入力
2. **フォルダに移動**: `cd C:\Users\西村康成\WasherInspection`
3. **コマンドを実行**: `python custom_remote_desktop.py server` など

または、**バッチファイルをダブルクリック**するだけでもOKです！






