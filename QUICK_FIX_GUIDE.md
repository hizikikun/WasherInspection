# エラー解決ガイド

## 🔍 原因

コマンドプロンプトが**プロジェクトフォルダ**ではなく、**ホームディレクトリ**で実行されています。

- **現在の場所**: `C:\Users\西村康成`
- **ファイルがある場所**: `C:\Users\西村康成\WasherInspection`

---

## ✅ 解決方法

### 方法1: プロジェクトフォルダに移動（推奨）

コマンドプロンプトで以下を実行:

```bash
cd "C:\Users\西村康成\WasherInspection"
```

その後、再度コマンドを実行:

```bash
check_setup.bat
```

または

```bash
python check_chrome_like_setup.py
```

---

### 方法2: エクスプローラーから開く（最も簡単）

1. **エクスプローラーでプロジェクトフォルダを開く**
   - `C:\Users\西村康成\WasherInspection`

2. **アドレスバーに「cmd」と入力してEnter**
   - そのフォルダでコマンドプロンプトが開きます

3. **コマンドを実行**
   ```bash
   check_setup.bat
   ```

---

### 方法3: フルパスで実行

現在のディレクトリからでも実行できます:

```bash
python "C:\Users\西村康成\WasherInspection\check_chrome_like_setup.py"
```

---

## 🎯 確認方法

正しいディレクトリにいるか確認:

```bash
cd
```

または

```bash
echo %CD%
```

**正しい場合の表示**:
```
C:\Users\西村康成\WasherInspection
```

**間違っている場合の表示**:
```
C:\Users\西村康成
```

この場合は、`cd "C:\Users\西村康成\WasherInspection"` で移動してください。



