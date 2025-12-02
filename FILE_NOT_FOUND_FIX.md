# ファイルが見つからないエラーの対処法

## 🔴 エラーの内容

```
can't open file 'C:\\Users\\kouse\\custom_remote_desktop.py': [Errno 2] No such file or directory
```

このエラーは、`custom_remote_desktop.py` ファイルが見つからないことを意味します。

---

## 🎯 原因

コマンドプロンプトが `C:\Users\kouse\` ディレクトリで実行されているため、そこに `custom_remote_desktop.py` ファイルがないことが原因です。

ファイルは、デスクトップの `RemoteDesktop` フォルダにあるはずです。

---

## ✅ 解決方法

### 方法1: 正しいフォルダに移動してから実行（推奨）

#### ステップ1: エクスプローラーでフォルダを開く

1. **エクスプローラーを開く**
   - Windowsキー + E を押す

2. **デスクトップの `RemoteDesktop` フォルダを開く**
   - 左側のサイドバーで「デスクトップ」をクリック
   - `RemoteDesktop` フォルダをダブルクリック

#### ステップ2: コマンドプロンプトを開く

1. **アドレスバーに「cmd」と入力**
   - フォルダを開いた状態で、上部のアドレスバーをクリック
   - 「**cmd**」と入力してEnterキーを押す

2. **コマンドプロンプトが開く**
   - パスが `C:\Users\kouse\Desktop\RemoteDesktop>` のように表示されます

#### ステップ3: ファイルが存在するか確認

コマンドプロンプトに以下を入力してEnterキーを押す:

```bash
dir custom_remote_desktop.py
```

**表示される内容の例:**
```
 Volume in drive C has no label.
 Volume Serial Number is XXXX-XXXX

 Directory of C:\Users\kouse\Desktop\RemoteDesktop

2024/01/01  10:00            15,234 custom_remote_desktop.py
               1 File(s)         15,234 bytes
```

**ファイルが表示されればOKです。**

#### ステップ4: クライアントを起動

コマンドプロンプトに以下を入力してEnterキーを押す:

```bash
python custom_remote_desktop.py client 192.168.1.100
```

（`192.168.1.100` を実際のIPアドレスに変更）

---

### 方法2: フルパスを指定して実行

現在のディレクトリから実行する場合、フルパスを指定します。

コマンドプロンプトに以下を入力してEnterキーを押す:

```bash
python C:\Users\kouse\Desktop\RemoteDesktop\custom_remote_desktop.py client 192.168.1.100
```

（`192.168.1.100` を実際のIPアドレスに変更）

---

### 方法3: cdコマンドでフォルダに移動

コマンドプロンプトに以下を入力してEnterキーを押す:

```bash
cd C:\Users\kouse\Desktop\RemoteDesktop
```

その後、クライアントを起動:

```bash
python custom_remote_desktop.py client 192.168.1.100
```

---

## 🔍 ファイルが存在しない場合

### ファイルがフォルダにない場合

1. **エクスプローラーでフォルダを確認**
   - デスクトップの `RemoteDesktop` フォルダを開く
   - `custom_remote_desktop.py` が存在するか確認

2. **ファイルが存在しない場合**
   - デスクトップ側からファイルをコピーしてください
   - または、`WasherInspection` フォルダから直接コピーしてください

### ファイルのコピー方法

1. **デスクトップ側の `WasherInspection` フォルダを開く**
   - `C:\Users\西村康成\WasherInspection` を開く

2. **ファイルをコピー**
   - `custom_remote_desktop.py` を右クリック → 「コピー」

3. **ノートパソコンのフォルダに貼り付け**
   - ノートパソコンのデスクトップの `RemoteDesktop` フォルダを開く
   - 右クリック → 「貼り付け」

---

## 📋 確認手順

### ステップ1: ファイルの存在確認

エクスプローラーで以下を確認:

1. **デスクトップの `RemoteDesktop` フォルダを開く**
2. **以下のファイルがあるか確認:**
   - ✅ `custom_remote_desktop.py`
   - ✅ `start_custom_remote_client.bat`（オプション）

### ステップ2: コマンドプロンプトで確認

1. **エクスプローラーでフォルダを開く**
2. **アドレスバーに「cmd」と入力してEnter**
3. **ファイルを確認:**
   ```bash
   dir
   ```

**表示される内容の例:**
```
 Volume in drive C has no label.
 Volume Serial Number is XXXX-XXXX

 Directory of C:\Users\kouse\Desktop\RemoteDesktop

2024/01/01  10:00    <DIR>          .
2024/01/01  10:00    <DIR>          ..
2024/01/01  10:00            15,234 custom_remote_desktop.py
2024/01/01  10:00               567 start_custom_remote_client.bat
               2 File(s)         15,803 bytes
```

**`custom_remote_desktop.py` が表示されればOKです。**

---

## 💡 正しい実行方法

### 最も簡単な方法

1. **エクスプローラーでフォルダを開く**
   - デスクトップの `RemoteDesktop` フォルダを開く

2. **アドレスバーに「cmd」と入力**
   - フォルダを開いた状態で、上部のアドレスバーをクリック
   - 「**cmd**」と入力してEnterキーを押す

3. **コマンドプロンプトが開く**
   - パスが `C:\Users\kouse\Desktop\RemoteDesktop>` のように表示されます

4. **クライアントを起動**
   ```bash
   python custom_remote_desktop.py client 192.168.1.100
   ```

---

## 📝 まとめ

1. ✅ エクスプローラーで `RemoteDesktop` フォルダを開く
2. ✅ アドレスバーに「cmd」と入力してEnter
3. ✅ コマンドプロンプトが正しいフォルダで開くことを確認
4. ✅ `python custom_remote_desktop.py client 192.168.1.100` を実行

これで、ファイルが見つからなくなることはありません！





