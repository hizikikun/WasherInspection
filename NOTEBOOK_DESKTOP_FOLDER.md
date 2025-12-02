# ノートパソコンのデスクトップにフォルダを作成した場合

## 📁 フォルダを作成した後の手順

ノートパソコンのデスクトップに必要なファイルをまとめたフォルダを作成したら、以下の手順で進めてください。

---

## 🎯 ノートパソコン側で必要なこと

### ステップ1: フォルダの内容を確認

デスクトップに作成したフォルダに、以下のファイルがあるか確認してください:

- ✅ `custom_remote_desktop.py`（必須）
- ✅ `start_custom_remote_client.bat`（オプション、推奨）

### ステップ2: コマンドプロンプトを開く

#### 方法A: エクスプローラーから開く（簡単・推奨）

1. エクスプローラーでデスクトップのフォルダを開く
2. アドレスバーに「**cmd**」と入力してEnterキーを押す
3. そのフォルダでコマンドプロンプトが開きます

#### 方法B: コマンドプロンプトから開く

1. Windowsキー + R → 「cmd」と入力 → Enter
2. フォルダに移動:
   ```bash
   cd C:\Users\kouse\Desktop\RemoteDesktop
   ```
   （フォルダ名に合わせて変更してください）

### ステップ3: Pythonがインストールされているか確認

```bash
python --version
```

または

```bash
py --version
```

**Pythonが表示されない場合:**
- Pythonをインストールする必要があります
- https://www.python.org/downloads/ からダウンロード
- インストール時に「**Add Python to PATH**」にチェックを入れる

### ステップ4: ライブラリをインストール

```bash
python -m pip install mss pillow pyautogui
```

または

```bash
py -m pip install mss pillow pyautogui
```

**エラーが出る場合:**
- `pip` が使えない場合は、`python -m pip` を使用してください
- Pythonがインストールされていない場合は、先にPythonをインストールしてください

### ステップ5: デスクトップ側のIPアドレスを確認

デスクトップ側で `ipconfig` を実行して、IPv4アドレスを確認してください。

例: `192.168.1.100`

### ステップ6: クライアントを起動

#### 方法A: バッチファイルから起動（簡単）

1. エクスプローラーでデスクトップのフォルダを開く
2. `start_custom_remote_client.bat` をダブルクリック
3. プロンプトが表示されたら、デスクトップ側のIPアドレスを入力:
   ```
   サーバーIPアドレスを入力してください (デフォルト: 192.168.1.100): 192.168.1.100
   ```
4. Enterキーを押す

#### 方法B: コマンドプロンプトから起動

```bash
python custom_remote_desktop.py client 192.168.1.100
```

（`192.168.1.100` の部分を、デスクトップ側のIPアドレスに変更）

または

```bash
py custom_remote_desktop.py client 192.168.1.100
```

### ステップ7: 接続成功

GUIウィンドウが開き、デスクトップの画面が表示されれば成功です！

- **マウス操作**: 画面内でマウスを動かすと、デスクトップ側のマウスが動きます
- **クリック**: 画面をクリックすると、デスクトップ側でクリックされます
- **キーボード**: キーを押すと、デスクトップ側で入力されます

---

## 📋 チェックリスト

- [ ] デスクトップにフォルダを作成した
- [ ] `custom_remote_desktop.py` がフォルダにある
- [ ] `start_custom_remote_client.bat` がフォルダにある（オプション）
- [ ] Pythonがインストールされている（`python --version` で確認）
- [ ] ライブラリをインストールした（`python -m pip install mss pillow pyautogui`）
- [ ] デスクトップ側のIPアドレスを確認した
- [ ] クライアントを起動した（`python custom_remote_desktop.py client [IPアドレス]`）

---

## 🔍 フォルダの場所の例

ノートパソコンのデスクトップにフォルダを作成した場合のパス例:

- `C:\Users\kouse\Desktop\RemoteDesktop\`
- `C:\Users\kouse\Desktop\リモートデスクトップ\`
- `C:\Users\kouse\Desktop\MyRemote\`

どこでも構いませんが、パスに日本語が含まれない方が安全です。

---

## 💡 簡単な方法（エクスプローラーから開く）

### 最も簡単な手順

1. **エクスプローラーでデスクトップのフォルダを開く**
   - デスクトップ上のフォルダをダブルクリック

2. **アドレスバーに「cmd」と入力**
   - フォルダを開いた状態で、上部のアドレスバーをクリック
   - 「**cmd**」と入力してEnterキーを押す

3. **コマンドプロンプトが開く**
   - そのフォルダでコマンドプロンプトが開きます

4. **コマンドを実行**
   ```bash
   python -m pip install mss pillow pyautogui
   python custom_remote_desktop.py client 192.168.1.100
   ```

---

## 🚀 実際の使用例

### ノートパソコンのデスクトップに `RemoteDesktop` フォルダを作成した場合

1. **エクスプローラーでフォルダを開く**
   - デスクトップ上の `RemoteDesktop` フォルダをダブルクリック

2. **コマンドプロンプトを開く**
   - アドレスバーに「cmd」と入力してEnter

3. **ライブラリをインストール**
   ```bash
   python -m pip install mss pillow pyautogui
   ```

4. **クライアントを起動**
   ```bash
   python custom_remote_desktop.py client 192.168.1.100
   ```
   （`192.168.1.100` をデスクトップ側のIPアドレスに変更）

---

## ⚠️ 注意事項

1. **デスクトップ側でサーバーを起動している必要があります**
   - ノートパソコン側から接続する前に、デスクトップ側でサーバーを起動してください

2. **同じネットワークに接続されている必要があります**
   - 両方のPCが同じWiFiまたは有線LANに接続されている必要があります

3. **IPアドレスが正しいか確認**
   - デスクトップ側で `ipconfig` を実行して、正しいIPアドレスを確認してください

---

## 📝 まとめ

1. ✅ デスクトップにフォルダを作成した
2. ✅ フォルダに `custom_remote_desktop.py` がある
3. ✅ コマンドプロンプトを開く（エクスプローラーから「cmd」と入力）
4. ✅ ライブラリをインストール（`python -m pip install mss pillow pyautogui`）
5. ✅ デスクトップ側のIPアドレスを確認
6. ✅ クライアントを起動（`python custom_remote_desktop.py client [IPアドレス]`）

これで、ノートパソコン側からデスクトップに接続できます！





