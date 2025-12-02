# ノートパソコン側に必要なファイル

## 📋 必要なファイル一覧

ノートパソコン側でカスタムリモートデスクトップを使用するために必要なファイルは以下の通りです。

### 必須ファイル

1. **`custom_remote_desktop.py`**
   - メインのPythonアプリケーション
   - サーバー/クライアントの両方の機能を含む

2. **`start_custom_remote_client.bat`**（オプション、推奨）
   - クライアントを簡単に起動するためのバッチファイル
   - なくても動作しますが、あると便利

### オプションファイル

3. **`requirements.txt`**（オプション）
   - 必要なライブラリのリスト
   - ライブラリをインストールする際に使用

---

## 📥 ファイルの取得方法

### 方法1: GitHubからクローン（推奨）

プロジェクト全体をGitHubから取得する場合:

```bash
git clone https://github.com/your-username/WasherInspection.git
cd WasherInspection
```

### 方法2: 必要なファイルだけをコピー

以下のファイルだけをノートパソコンにコピーします:

1. `custom_remote_desktop.py`
2. `start_custom_remote_client.bat`（オプション）
3. `requirements.txt`（オプション）

### 方法3: USBメモリやクラウドストレージで転送

1. デスクトップ側で必要なファイルをUSBメモリやクラウドストレージにコピー
2. ノートパソコン側でそれらのファイルをダウンロード/コピー

---

## 📁 ファイルの配置場所

ノートパソコン側で、以下のようなフォルダ構造に配置してください:

```
C:\Users\西村康成\WasherInspection\
├── custom_remote_desktop.py          ← 必須
├── start_custom_remote_client.bat    ← 推奨
└── requirements.txt                  ← オプション
```

**注意**: フォルダ名は何でも構いませんが、パスに日本語が含まれない方が安全です。

---

## 🔍 ファイルの確認方法

ノートパソコン側でファイルが正しく配置されているか確認する方法:

### エクスプローラーで確認

1. エクスプローラーで `WasherInspection` フォルダを開く
2. 以下のファイルがあるか確認:
   - `custom_remote_desktop.py`
   - `start_custom_remote_client.bat`

### コマンドプロンプトで確認

```bash
cd C:\Users\西村康成\WasherInspection
dir custom_remote_desktop.py
dir start_custom_remote_client.bat
```

ファイルが存在する場合、ファイル情報が表示されます。

---

## 📦 最小構成（ファイル数が少ない場合）

**最小限必要なファイルは1つだけです:**

- `custom_remote_desktop.py`

このファイル1つがあれば、以下のコマンドでクライアントを起動できます:

```bash
python custom_remote_desktop.py client 192.168.1.100
```

バッチファイルがなくても動作します。

---

## 🚀 セットアップ手順

### ステップ1: ファイルをコピー

デスクトップ側からノートパソコン側に以下のファイルをコピー:

- `custom_remote_desktop.py`
- `start_custom_remote_client.bat`（オプション）

### ステップ2: 同じフォルダに配置

ノートパソコン側で、同じフォルダに配置します。

例:
```
C:\Users\西村康成\WasherInspection\
```

### ステップ3: ライブラリをインストール

ノートパソコン側で、コマンドプロンプトを開いて:

```bash
cd C:\Users\西村康成\WasherInspection
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

## 📋 ファイルサイズの目安

- `custom_remote_desktop.py`: 約 15-20 KB
- `start_custom_remote_client.bat`: 約 1 KB
- `requirements.txt`: 約 5 KB

**合計**: 約 20-30 KB（非常に軽量）

---

## 🔄 ファイルの更新

ファイルを更新する場合は、デスクトップ側で最新版を取得して、ノートパソコン側にコピーし直してください。

---

## ❓ よくある質問

### Q: すべてのファイルが必要ですか？

A: いいえ。`custom_remote_desktop.py` 1つだけで動作します。バッチファイルは便利ですが必須ではありません。

### Q: フォルダ名は何でもいいですか？

A: はい。フォルダ名は何でも構いません。ただし、パスに日本語やスペースが含まれない方が安全です。

### Q: GitHubから取得する必要がありますか？

A: いいえ。必要なファイルだけをコピーすればOKです。

### Q: ファイルをどこに置けばいいですか？

A: どこでも構いませんが、わかりやすい場所（例: `C:\Users\西村康成\WasherInspection\`）に置くことを推奨します。

---

## 📝 まとめ

**ノートパソコン側に必要なファイル:**

1. ✅ **`custom_remote_desktop.py`**（必須）
2. ✅ **`start_custom_remote_client.bat`**（推奨、オプション）

この2つのファイルがあれば、ノートパソコン側でクライアントを起動できます。






