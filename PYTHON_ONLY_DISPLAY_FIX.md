# 「Python」とだけ表示される問題の解決法

## 🔴 問題の症状

すべての条件を満たしているが、実行すると「Python」とだけ表示される。

---

## 🔍 原因の確認

「Python」とだけ表示される場合、以下のいずれかが原因です:

1. **`python` コマンドだけを実行している**（ファイル名を指定していない）
2. **コマンドの入力方法が間違っている**
3. **ファイル名が間違っている**

---

## ✅ 正しい実行方法

### ステップ1: コマンドプロンプトを開く

1. **エクスプローラーでフォルダを開く**
   - `C:\Users\kouse\OneDrive\ドキュメント\デスクトップ\Remote Desktop` フォルダを開く

2. **アドレスバーに「cmd」と入力**
   - フォルダを開いた状態で、上部のアドレスバーをクリック
   - 「**cmd**」と入力してEnterキーを押す

3. **コマンドプロンプトが開く**
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

### ステップ3: 正しいコマンドを実行

**重要**: 以下のコマンドを**そのまま**コピーして貼り付けてください。

```bash
python custom_remote_desktop.py client 192.168.1.100
```

**注意点:**
- `python` の後に**スペース**を入れる
- `custom_remote_desktop.py` の後に**スペース**を入れる
- `client` の後に**スペース**を入れる
- `192.168.1.100` を実際のIPアドレスに変更

---

## ❌ よくある間違い

### 間違い1: `python` だけを実行

```bash
python
```

これはPythonの対話モードに入るだけです。ファイルを実行するには、ファイル名も指定する必要があります。

### 間違い2: ファイル名を忘れる

```bash
python client 192.168.1.100
```

ファイル名（`custom_remote_desktop.py`）が抜けています。

### 間違い3: スペースがない

```bash
pythoncustom_remote_desktop.py client 192.168.1.100
```

`python` と `custom_remote_desktop.py` の間にスペースがありません。

### 間違い4: 拡張子を忘れる

```bash
python custom_remote_desktop client 192.168.1.100
```

`.py` 拡張子が抜けています。

---

## ✅ 正しいコマンドの例

### 例1: IPアドレスが `192.168.1.100` の場合

```bash
python custom_remote_desktop.py client 192.168.1.100
```

### 例2: IPアドレスが `192.168.0.50` の場合

```bash
python custom_remote_desktop.py client 192.168.0.50
```

---

## 🔍 コマンドの確認方法

### コマンドを入力する前に確認

1. **現在のディレクトリを確認**
   ```bash
   cd
   ```
   または
   ```bash
   echo %CD%
   ```
   
   **表示される内容の例:**
   ```
   C:\Users\kouse\OneDrive\ドキュメント\デスクトップ\Remote Desktop
   ```

2. **ファイルが存在するか確認**
   ```bash
   dir custom_remote_desktop.py
   ```

3. **正しいコマンドを入力**
   ```bash
   python custom_remote_desktop.py client 192.168.1.100
   ```

---

## 💡 コマンドをコピー&ペーストする方法

### 方法1: このページからコピー

1. **コマンドを選択**
   - 以下のコマンドをマウスで選択（ドラッグ）:
     ```bash
     python custom_remote_desktop.py client 192.168.1.100
     ```

2. **コピー**
   - Ctrl + C を押す

3. **コマンドプロンプトに貼り付け**
   - コマンドプロンプトをクリック
   - 右クリック → 「貼り付け」
   - または、Ctrl + V を押す

4. **IPアドレスを変更**
   - 貼り付けたコマンドの `192.168.1.100` の部分を、実際のIPアドレスに変更

5. **Enterキーを押す**

---

## 🔄 完全な手順（最初から）

### ステップ1: コマンドプロンプトを開く

1. **エクスプローラーを開く**（Windowsキー + E）
2. **フォルダを開く**
   - `C:\Users\kouse\OneDrive\ドキュメント\デスクトップ\Remote Desktop` を開く
3. **アドレスバーに「cmd」と入力**
   - アドレスバーをクリック
   - 「cmd」と入力してEnter

### ステップ2: ファイルを確認

```bash
dir custom_remote_desktop.py
```

### ステップ3: コマンドを実行

```bash
python custom_remote_desktop.py client 192.168.1.100
```

（`192.168.1.100` を実際のIPアドレスに変更）

### ステップ4: 結果を確認

- **GUIウィンドウが開く**: 接続成功！
- **エラーメッセージが表示される**: 内容を確認してください

---

## 📋 チェックリスト

- [ ] コマンドプロンプトが正しいフォルダで開いている
- [ ] ファイルが存在する（`dir custom_remote_desktop.py` で確認）
- [ ] 正しいコマンドを入力している（`python custom_remote_desktop.py client [IPアドレス]`）
- [ ] コマンドの各部分の間にスペースがある
- [ ] `.py` 拡張子が含まれている
- [ ] IPアドレスが正しい

---

## 🆘 それでも解決しない場合

### デバッグ情報を収集

以下の情報をメモしてください:

1. **実行したコマンド**
   - コマンドプロンプトに入力したコマンドをそのままコピー

2. **コマンドプロンプトに表示された内容**
   - 「Python」と表示された前後の内容も含めて

3. **ファイルの存在確認**
   - `dir custom_remote_desktop.py` の結果

4. **現在のディレクトリ**
   - `cd` コマンドの結果

これらの情報があれば、さらに詳しくサポートできます。

---

## 📝 まとめ

1. ✅ コマンドプロンプトが正しいフォルダで開いていることを確認
2. ✅ ファイルが存在することを確認（`dir custom_remote_desktop.py`）
3. ✅ **正しいコマンドを入力**（`python custom_remote_desktop.py client 192.168.1.100`）
4. ✅ コマンドの各部分の間にスペースがあることを確認
5. ✅ `.py` 拡張子が含まれていることを確認

**重要**: `python` だけではなく、`python custom_remote_desktop.py client 192.168.1.100` と**完全なコマンド**を入力してください。





