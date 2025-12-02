# ノートパソコン側のセットアップ（簡単版）

## ❓ プロジェクト全体をダウンロードする必要はありますか？

**いいえ、必要ありません！**

カスタムリモートデスクトップを使う場合、**必要なファイルだけ**をコピーすればOKです。

---

## 📋 必要なファイル（最小構成）

### カスタムリモートデスクトップを使う場合

**必要なファイルは1つだけ:**

1. ✅ **`custom_remote_desktop.py`**

これだけで動作します！

### オプション（あると便利）

2. ✅ **`start_custom_remote_client.bat`**
   - 簡単に起動するためのバッチファイル
   - なくても動作します

---

## 📁 ファイルの配置

ノートパソコン側で、**新しいフォルダを作成**して、そこにファイルを配置します。

例:
```
C:\Users\西村康成\RemoteDesktop\
├── custom_remote_desktop.py
└── start_custom_remote_client.bat
```

**WasherInspectionフォルダ全体は不要です！**

---

## 🚀 セットアップ手順

### ステップ1: フォルダを作成

ノートパソコン側で、新しいフォルダを作成します。

例: `C:\Users\西村康成\RemoteDesktop\`

### ステップ2: ファイルをコピー

デスクトップ側の `WasherInspection` フォルダから、以下のファイルだけをコピー:

- `custom_remote_desktop.py`
- `start_custom_remote_client.bat`（オプション）

### ステップ3: ライブラリをインストール

ノートパソコン側で、コマンドプロンプトを開いて:

```bash
cd C:\Users\西村康成\RemoteDesktop
pip install mss pillow pyautogui
```

### ステップ4: クライアントを起動

```bash
python custom_remote_desktop.py client 192.168.1.100
```

または、バッチファイルを使用:

```bash
start_custom_remote_client.bat
```

---

## 📊 比較表

| 方法 | 必要なファイル | フォルダサイズ |
|------|---------------|----------------|
| **プロジェクト全体** | WasherInspectionフォルダ全体 | 数百MB〜数GB |
| **必要なファイルだけ** | `custom_remote_desktop.py` のみ | 約20KB |

**必要なファイルだけをコピーする方が圧倒的に軽量です！**

---

## 💡 リモートデスクトップ管理アプリも使いたい場合

もし `remote_desktop_manager.py`（リモートデスクトップ管理アプリ）も使いたい場合は、以下のファイルも追加でコピーします:

- `remote_desktop_manager.py`
- `start_remote_desktop_manager.bat`（オプション）

**それでも、プロジェクト全体は不要です！**

---

## ❓ よくある質問

### Q: WasherInspectionフォルダ全体をコピーする必要がありますか？

A: **いいえ、必要ありません。** `custom_remote_desktop.py` 1つだけで動作します。

### Q: 他のファイル（requirements.txtなど）は必要ですか？

A: いいえ。`custom_remote_desktop.py` 1つだけで動作します。`requirements.txt` は、必要なライブラリを確認するための参考資料ですが、なくても動作します。

### Q: フォルダ名は何でもいいですか？

A: はい。`RemoteDesktop`、`MyRemote`、`Desktop` など、何でも構いません。

### Q: デスクトップ側と同じフォルダ名にする必要がありますか？

A: いいえ。フォルダ名は何でも構いません。

---

## 📝 まとめ

- ❌ **WasherInspectionフォルダ全体は不要**
- ✅ **`custom_remote_desktop.py` 1つだけでOK**
- ✅ **新しいフォルダを作成して、そこにファイルを配置**

これで、ノートパソコン側でリモートデスクトップクライアントを使用できます！





