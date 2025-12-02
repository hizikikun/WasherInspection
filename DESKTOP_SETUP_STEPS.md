# デスクトップ側のセットアップ手順

## 📁 フォルダを作成した後の手順

デスクトップ側にフォルダを作成したら、以下の手順で進めてください。

---

## 🎯 デスクトップ側で必要なこと

### ステップ1: 必要なファイルをフォルダに配置

作成したフォルダに、以下のファイルをコピーします:

1. **`custom_remote_desktop.py`**
   - メインのPythonアプリケーション
   - `WasherInspection` フォルダからコピー

2. **`start_custom_remote_server.bat`**（オプション、推奨）
   - サーバーを簡単に起動するためのバッチファイル
   - `WasherInspection` フォルダからコピー

### ステップ2: ファイルの配置場所を確認

例: デスクトップに `RemoteDesktop` フォルダを作成した場合

```
C:\Users\西村康成\Desktop\RemoteDesktop\
├── custom_remote_desktop.py
└── start_custom_remote_server.bat
```

### ステップ3: ライブラリをインストール

1. コマンドプロンプトを開く
   - Windowsキー + R → 「cmd」と入力 → Enter

2. 作成したフォルダに移動
   ```bash
   cd C:\Users\西村康成\Desktop\RemoteDesktop
   ```
   （フォルダの場所に合わせて変更してください）

3. ライブラリをインストール
   ```bash
   python -m pip install mss pillow pyautogui
   ```
   または
   ```bash
   pip install mss pillow pyautogui
   ```

### ステップ4: IPアドレスを確認

サーバーを起動する前に、デスクトップのIPアドレスを確認します:

```bash
ipconfig
```

**IPv4アドレスをメモしてください**（例: `192.168.1.100`）

このIPアドレスをノートパソコン側で使用します。

### ステップ5: サーバーを起動

#### 方法A: バッチファイルから起動（簡単）
1. エクスプローラーで `RemoteDesktop` フォルダを開く
2. `start_custom_remote_server.bat` をダブルクリック

#### 方法B: コマンドプロンプトから起動
```bash
cd C:\Users\西村康成\Desktop\RemoteDesktop
python custom_remote_desktop.py server
```

**表示される内容:**
```
=== リモートデスクトップサーバー ===
デスクトップ側で実行してください
クライアントの接続を待機しています...
[サーバー] 待機中: 0.0.0.0:8888
```

**この状態で待機します。** ノートパソコン側からの接続を待っています。

---

## 📋 チェックリスト

- [ ] フォルダを作成した
- [ ] `custom_remote_desktop.py` をフォルダにコピーした
- [ ] `start_custom_remote_server.bat` をフォルダにコピーした（オプション）
- [ ] ライブラリをインストールした（`python -m pip install mss pillow pyautogui`）
- [ ] IPアドレスを確認した（`ipconfig`）
- [ ] サーバーを起動した（`python custom_remote_desktop.py server`）

---

## 🔍 ファイルのコピー方法

### 方法1: エクスプローラーでコピー

1. エクスプローラーで `WasherInspection` フォルダを開く
2. `custom_remote_desktop.py` を右クリック → 「コピー」
3. 作成したフォルダ（例: `Desktop\RemoteDesktop`）を開く
4. 右クリック → 「貼り付け」

### 方法2: コマンドプロンプトでコピー

```bash
copy C:\Users\西村康成\WasherInspection\custom_remote_desktop.py C:\Users\西村康成\Desktop\RemoteDesktop\
```

---

## 💡 フォルダの場所の例

デスクトップにフォルダを作成した場合のパス例:

- `C:\Users\西村康成\Desktop\RemoteDesktop\`
- `C:\Users\西村康成\Desktop\リモートデスクトップ\`
- `C:\Users\西村康成\Desktop\MyRemote\`

どこでも構いませんが、パスに日本語が含まれない方が安全です。

---

## 🚀 実際の使用例

### デスクトップ側のフォルダが `C:\Users\西村康成\Desktop\RemoteDesktop\` の場合

1. **ファイルをコピー**
   - `custom_remote_desktop.py` をコピー

2. **コマンドプロンプトを開く**
   ```bash
   cd C:\Users\西村康成\Desktop\RemoteDesktop
   ```

3. **ライブラリをインストール**
   ```bash
   python -m pip install mss pillow pyautogui
   ```

4. **IPアドレスを確認**
   ```bash
   ipconfig
   ```
   例: `192.168.1.100`

5. **サーバーを起動**
   ```bash
   python custom_remote_desktop.py server
   ```

---

## ⚠️ 注意事項

1. **フォルダ名に日本語が含まれていても大丈夫です**
   - ただし、パスにバックスラッシュ（`\`）を忘れないでください

2. **Pythonがインストールされている必要があります**
   - `python --version` で確認
   - インストールされていない場合は、https://www.python.org/downloads/ からダウンロード

3. **ファイアウォールの設定**
   - ポート8888がブロックされている場合は、ファイアウォールで開放する必要があります

---

## 📝 まとめ

1. ✅ フォルダを作成した
2. ✅ `custom_remote_desktop.py` をフォルダにコピー
3. ✅ ライブラリをインストール（`python -m pip install mss pillow pyautogui`）
4. ✅ IPアドレスを確認（`ipconfig`）
5. ✅ サーバーを起動（`python custom_remote_desktop.py server`）

これで、デスクトップ側の準備は完了です！





