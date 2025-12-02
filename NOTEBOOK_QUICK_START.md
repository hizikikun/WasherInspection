# ノートパソコン側 - クイックスタートガイド

最短でクライアントをセットアップする手順です。

---

## 🚀 5分で始める

### ステップ1: Pythonの確認

```bash
python --version
```

✅ バージョンが表示されればOK  
❌ エラーが出る場合は、Pythonをインストール（「Add Python to PATH」にチェック）

---

### ステップ2: プロジェクトファイルをコピー

デスクトップ側から以下のファイルをノートパソコンにコピー:
- `chrome_like_remote_desktop.py`（必須）
- `start_chrome_like_client.bat`（オプション）

---

### ステップ3: ライブラリをインストール

プロジェクトフォルダでコマンドプロンプトを開き:

```bash
python -m pip install mss pillow pyautogui pyperclip cryptography
```

---

### ステップ4: デスクトップ側の情報を確認

- **IPアドレス**: デスクトップ側で `ipconfig` → IPv4アドレスをメモ
- **パスワード**: デスクトップ側のサーバー起動画面で表示されたパスワードをメモ

---

### ステップ5: クライアントを起動

```bash
python chrome_like_remote_desktop.py client 192.168.1.100
```
（IPアドレスを変更）

1. GUIが開いたら「接続」をクリック
2. パスワードを入力
3. 完了！

---

## 📋 必要なもの

- Python 3.7以上
- デスクトップ側のIPアドレス
- デスクトップ側のパスワード
- 同じネットワークに接続された2台のPC

---

詳細な手順は `NOTEBOOK_CLIENT_SETUP.md` を参照してください。



