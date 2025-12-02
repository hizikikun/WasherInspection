# 「Python」とだけ表示される場合の対処法

## 📍 現在の状態

コマンドプロンプトに「Python」とだけ表示されている場合、以下のいずれかの状況です。

---

## 🔍 状況の確認

### 状況1: `python` コマンドを実行した場合

`python` とだけ入力してEnterキーを押した場合、Pythonの対話モードに入ります。

**表示例:**
```
Python
>>>
```

この場合、Pythonの対話モードに入っているので、以下の手順で対処してください。

#### 対処法

1. **Pythonの対話モードを終了**
   - `exit()` と入力してEnterキーを押す
   - または、`Ctrl + Z` を押してEnterキーを押す
   - または、`Ctrl + C` を押す

2. **コマンドプロンプトに戻る**
   - パスが `C:\Users\kouse\OneDrive\ドキュメント\デスクトップ\Remote Desktop>` のように表示されます

3. **正しいコマンドを実行**
   ```bash
   python custom_remote_desktop.py client 192.168.1.100
   ```

---

### 状況2: Pythonのバージョンを確認したい場合

Pythonのバージョンを確認するには、以下のコマンドを実行してください:

```bash
python --version
```

**表示される内容の例:**
```
Python 3.11.5
```

---

### 状況3: ファイルを実行したい場合

`custom_remote_desktop.py` を実行するには、以下のコマンドを実行してください:

```bash
python custom_remote_desktop.py client 192.168.1.100
```

（`192.168.1.100` を実際のIPアドレスに変更）

---

## 🎯 正しい手順

### ステップ1: Pythonの対話モードから抜ける

もしPythonの対話モードに入っている場合（`>>>` が表示されている場合）:

1. **`exit()` と入力してEnterキーを押す**
   ```python
   exit()
   ```

2. **コマンドプロンプトに戻る**
   - パスが `C:\Users\kouse\OneDrive\ドキュメント\デスクトップ\Remote Desktop>` のように表示されます

### ステップ2: ファイルが存在するか確認

コマンドプロンプトに以下を入力してEnterキーを押す:

```bash
dir custom_remote_desktop.py
```

**表示される内容の例:**
```
 Volume in drive C has no label.
 Volume Serial Number is XXXX-XXXX

 Directory of C:\Users\kouse\OneDrive\ドキュメント\デスクトップ\Remote Desktop

2024/01/01  10:00            15,234 custom_remote_desktop.py
               1 File(s)         15,234 bytes
```

**ファイルが表示されればOKです。**

### ステップ3: クライアントを起動

コマンドプロンプトに以下を入力してEnterキーを押す:

```bash
python custom_remote_desktop.py client 192.168.1.100
```

（`192.168.1.100` を実際のIPアドレスに変更）

---

## 💡 よくある間違い

### 間違い1: `python` だけを実行

```bash
python
```

これはPythonの対話モードに入るだけです。ファイルを実行するには、ファイル名も指定する必要があります。

### 正しい方法

```bash
python custom_remote_desktop.py client 192.168.1.100
```

### 間違い2: ファイル名を間違える

```bash
python custom_remote_desktop client 192.168.1.100
```

`.py` 拡張子を忘れています。

### 正しい方法

```bash
python custom_remote_desktop.py client 192.168.1.100
```

---

## 🔍 トラブルシューティング

### ファイルが見つからない場合

1. **エクスプローラーでフォルダを確認**
   - `C:\Users\kouse\OneDrive\ドキュメント\デスクトップ\Remote Desktop` フォルダを開く
   - `custom_remote_desktop.py` が存在するか確認

2. **ファイルが存在しない場合**
   - デスクトップ側からファイルをコピーしてください

### Pythonが認識されない場合

1. **Pythonのバージョンを確認**
   ```bash
   python --version
   ```

2. **Pythonが表示されない場合**
   - Pythonをインストールしてください
   - https://www.python.org/downloads/ からダウンロード
   - インストール時に「Add Python to PATH」にチェックを入れる

---

## 📝 正しいコマンドの例

### クライアントを起動する場合

```bash
python custom_remote_desktop.py client 192.168.1.100
```

### Pythonのバージョンを確認する場合

```bash
python --version
```

### ファイルの一覧を表示する場合

```bash
dir
```

### ファイルが存在するか確認する場合

```bash
dir custom_remote_desktop.py
```

---

## 📋 チェックリスト

- [ ] Pythonの対話モードから抜けた（`exit()` を実行）
- [ ] コマンドプロンプトが正しいフォルダで開いている
- [ ] `custom_remote_desktop.py` ファイルが存在する（`dir custom_remote_desktop.py` で確認）
- [ ] 正しいコマンドを実行した（`python custom_remote_desktop.py client [IPアドレス]`）

---

## 📝 まとめ

1. ✅ もしPythonの対話モードに入っている場合は、`exit()` で抜ける
2. ✅ ファイルが存在するか確認（`dir custom_remote_desktop.py`）
3. ✅ 正しいコマンドを実行（`python custom_remote_desktop.py client 192.168.1.100`）

これで、クライアントが起動するはずです！





