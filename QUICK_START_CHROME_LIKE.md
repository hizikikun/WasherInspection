# Chrome風リモートデスクトップ - クイックスタートガイド

最短で使えるようになる手順です。

---

## 🚀 5分で始める

### ステップ1: ライブラリをインストール（両方のPCで）

1. **エクスプローラーでプロジェクトフォルダを開く**
   - `C:\Users\西村康成\WasherInspection`

2. **アドレスバーに「cmd」と入力してEnter**

3. **以下を実行**:
   ```bash
   python -m pip install mss pillow pyautogui pyperclip cryptography
   ```

---

### ステップ2: デスクトップ側でサーバーを起動

1. **コマンドプロンプトで以下を実行**:
   ```bash
   python chrome_like_remote_desktop.py server
   ```

2. **パスワードをメモ**（例: `xYz123AbC456DeF`）

3. **IPアドレスを確認**:
   ```bash
   ipconfig
   ```
   - IPv4アドレスをメモ（例: `192.168.1.100`）

---

### ステップ3: ノートパソコン側でクライアントを起動

1. **コマンドプロンプトで以下を実行**（IPアドレスを変更）:
   ```bash
   python chrome_like_remote_desktop.py client 192.168.1.100
   ```

2. **GUIが開いたら「接続」ボタンをクリック**

3. **パスワードを入力**（デスクトップ側で表示されたパスワード）

4. **完了！** リモート画面が表示されます

---

## 📋 必要なもの

- Python 3.7以上
- 同じネットワークに接続された2台のPC

---

## ⚡ もっと簡単に起動する方法

### バッチファイルを使用

1. **デスクトップ側**: `start_chrome_like_server.bat` をダブルクリック
2. **ノートパソコン側**: `start_chrome_like_client.bat` をダブルクリック

---

## ❓ エラーが出た場合

詳細な手順は `CHROME_LIKE_REMOTE_DESKTOP_SETUP_GUIDE.md` を参照してください。




