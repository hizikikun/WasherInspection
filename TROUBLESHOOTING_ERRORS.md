# エラー解決ガイド

## 🔴 エラー1: 「指定されたパスが見つかりません。」

### 原因
パスにバックスラッシュ（`\`）が抜けています。

**間違い:**
```bash
cd C:\Users西村康成\RemoteDesktop
```

**正しい:**
```bash
cd C:\Users\西村康成\RemoteDesktop
```

### 解決方法

#### 方法1: 正しいパスで実行
```bash
cd C:\Users\西村康成\RemoteDesktop
```

#### 方法2: フォルダが存在しない場合は作成
1. エクスプローラーで `C:\Users\西村康成\` を開く
2. 新しいフォルダを作成: `RemoteDesktop`
3. そのフォルダに `custom_remote_desktop.py` をコピー
4. コマンドプロンプトで:
   ```bash
   cd C:\Users\西村康成\RemoteDesktop
   ```

#### 方法3: エクスプローラーからコマンドプロンプトを開く（簡単）
1. エクスプローラーで `C:\Users\西村康成\RemoteDesktop` フォルダを開く
2. アドレスバーに「**cmd**」と入力してEnterキーを押す
3. そのフォルダでコマンドプロンプトが開きます

---

## 🔴 エラー2: 「'pip'は、内部コマンドまたは外部コマンド...」

### 原因
Pythonがインストールされていない、またはpipがPATHに追加されていません。

### 解決方法

#### 方法1: Pythonがインストールされているか確認
```bash
python --version
```

**Pythonが表示されない場合:**
- Pythonをインストールする必要があります
- https://www.python.org/downloads/ からダウンロード
- インストール時に「**Add Python to PATH**」にチェックを入れる

#### 方法2: python -m pip を使用
pipが直接使えない場合、以下のコマンドを試してください:

```bash
python -m pip install mss pillow pyautogui
```

#### 方法3: py コマンドを使用（Windows）
```bash
py -m pip install mss pillow pyautogui
```

#### 方法4: Pythonのフルパスを使用
Pythonがインストールされている場所を確認して、フルパスで実行:

```bash
C:\Users\kouse\AppData\Local\Programs\Python\Python3XX\python.exe -m pip install mss pillow pyautogui
```

（`Python3XX` の部分は、インストールされているPythonのバージョンに合わせて変更）

---

## 🚀 正しい手順（ノートパソコン側）

### ステップ1: フォルダを作成

1. エクスプローラーで `C:\Users\西村康成\` を開く
2. 新しいフォルダを作成: `RemoteDesktop`
3. そのフォルダに `custom_remote_desktop.py` をコピー

### ステップ2: コマンドプロンプトを開く

**方法A: エクスプローラーから開く（簡単）**
1. エクスプローラーで `C:\Users\西村康成\RemoteDesktop` フォルダを開く
2. アドレスバーに「**cmd**」と入力してEnterキーを押す

**方法B: コマンドプロンプトから開く**
1. Windowsキー + R → 「cmd」と入力 → Enter
2. 正しいパスで移動:
   ```bash
   cd C:\Users\西村康成\RemoteDesktop
   ```

### ステップ3: Pythonがインストールされているか確認

```bash
python --version
```

または

```bash
py --version
```

**Pythonが表示されない場合:**
- Pythonをインストールしてください
- https://www.python.org/downloads/ からダウンロード
- インストール時に「**Add Python to PATH**」にチェックを入れる

### ステップ4: ライブラリをインストール

**方法1: python -m pip を使用（推奨）**
```bash
python -m pip install mss pillow pyautogui
```

**方法2: py コマンドを使用**
```bash
py -m pip install mss pillow pyautogui
```

**方法3: pip が使える場合**
```bash
pip install mss pillow pyautogui
```

### ステップ5: クライアントを起動

```bash
python custom_remote_desktop.py client 192.168.1.100
```

または

```bash
py custom_remote_desktop.py client 192.168.1.100
```

---

## 📋 チェックリスト

- [ ] フォルダ `C:\Users\西村康成\RemoteDesktop` が存在する
- [ ] `custom_remote_desktop.py` がそのフォルダにある
- [ ] Pythonがインストールされている（`python --version` で確認）
- [ ] ライブラリがインストールされている（`python -m pip install mss pillow pyautogui`）

---

## 💡 簡単な方法

### エクスプローラーから開く方法（最も簡単）

1. エクスプローラーで `C:\Users\西村康成\RemoteDesktop` フォルダを開く
2. アドレスバーに「**cmd**」と入力してEnterキーを押す
3. コマンドプロンプトがそのフォルダで開きます
4. 以下のコマンドを実行:
   ```bash
   python -m pip install mss pillow pyautogui
   python custom_remote_desktop.py client 192.168.1.100
   ```

---

## ❓ よくある質問

### Q: Pythonがインストールされていない場合は？

A: https://www.python.org/downloads/ からダウンロードしてインストールしてください。インストール時に「**Add Python to PATH**」にチェックを入れることを忘れずに。

### Q: pipが使えない場合は？

A: `python -m pip` または `py -m pip` を使用してください。

### Q: フォルダ名に日本語が含まれていても大丈夫ですか？

A: はい、大丈夫です。ただし、パスにバックスラッシュ（`\`）を忘れないでください。

---

## 📝 まとめ

1. **パスの確認**: `C:\Users\西村康成\RemoteDesktop`（バックスラッシュを忘れずに）
2. **Pythonの確認**: `python --version` または `py --version`
3. **ライブラリのインストール**: `python -m pip install mss pillow pyautogui`
4. **クライアントの起動**: `python custom_remote_desktop.py client 192.168.1.100`





