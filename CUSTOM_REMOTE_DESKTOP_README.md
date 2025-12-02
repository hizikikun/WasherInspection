# カスタムリモートデスクトップシステム

自分で作成したリモートデスクトップ接続ソフトウェアです。

## 🎯 機能

- ✅ **画面共有**: デスクトップの画面をリアルタイムで共有
- ✅ **リモート制御**: マウスとキーボードのリモート制御
- ✅ **シンプルな設定**: サーバー/クライアントモードで簡単に使用
- ✅ **既存ソフト自動起動**: TeamViewer/AnyDeskの自動起動ラッパー

## 📋 必要なもの

### インストールが必要なライブラリ

```bash
pip install mss pillow pyautogui
```

または、requirements.txtからインストール：

```bash
pip install -r requirements.txt
```

### 必要なライブラリの説明

- **mss**: 画面キャプチャ（Windows、macOS、Linux対応）
- **pillow**: 画像処理（既にインストール済みの可能性あり）
- **pyautogui**: マウス/キーボード制御（Windows、macOS、Linux対応）

## 🚀 使い方

### 方法1: カスタムリモートデスクトップシステム（自分で作成）

#### デスクトップ側（サーバー）

1. **サーバーを起動**
   ```bash
   start_custom_remote_server.bat
   ```
   または
   ```bash
   python custom_remote_desktop.py server
   ```

2. **クライアントの接続を待機**
   - 画面に「待機中: 0.0.0.0:8888」と表示されます

#### ノートパソコン側（クライアント）

1. **クライアントを起動**
   ```bash
   start_custom_remote_client.bat
   ```
   または
   ```bash
   python custom_remote_desktop.py client [デスクトップのIPアドレス]
   ```

2. **接続**
   - サーバーアドレスを入力（例: `192.168.1.100`）
   - 「接続」ボタンをクリック

3. **リモート制御**
   - 画面が表示されたら、マウスとキーボードで操作できます

### 方法2: GUIモード

```bash
python custom_remote_desktop.py
```

サーバー/クライアントの選択画面が表示されます。

## ⚙️ 設定

### ポート番号の変更

デフォルトポートは `8888` です。変更する場合は、コード内の以下の部分を編集：

```python
# サーバー側
server = RemoteDesktopServer(host='0.0.0.0', port=8888)

# クライアント側
client = RemoteDesktopClient(server_host, 8888)
```

### 画面キャプチャの品質

`custom_remote_desktop.py` の以下の部分で品質を調整：

```python
img.save(img_buffer, format='JPEG', quality=70)  # 70を変更（1-100）
```

### フレームレート

`custom_remote_desktop.py` の以下の部分でフレームレートを調整：

```python
time.sleep(0.1)  # 10 FPS（0.1秒 = 10フレーム/秒）
```

## 🔧 トラブルシューティング

### 接続できない場合

1. **ファイアウォールの確認**
   - ポート8888がブロックされていないか確認
   - Windowsファイアウォールでポートを開放する必要がある場合があります

2. **IPアドレスの確認**
   - デスクトップ側のIPアドレスが正しいか確認
   - `ipconfig`（Windows）または`ifconfig`（Linux/macOS）で確認

3. **ネットワーク接続の確認**
   - 両方のPCが同じネットワークに接続されているか確認
   - 大学のWiFiから自宅に接続する場合は、ポートフォワーディングが必要な場合があります

### 画面が表示されない場合

1. **mssライブラリの確認**
   ```bash
   pip install mss
   ```

2. **画面キャプチャの権限確認**
   - Windows: 画面キャプチャの権限が必要な場合があります

### リモート制御が動作しない場合

1. **pyautoguiライブラリの確認**
   ```bash
   pip install pyautogui
   ```

2. **マウス/キーボード制御の権限確認**
   - 一部のシステムでは、リモート制御の権限が必要な場合があります

## 📊 パフォーマンス

### 推奨設定

- **画面品質**: 70%（バランス型）
- **フレームレート**: 10 FPS（標準）
- **ネットワーク**: 有線接続推奨（WiFiでも動作可能）

### 最適化のヒント

1. **画面品質を下げる**: ネットワークが遅い場合
   ```python
   img.save(img_buffer, format='JPEG', quality=50)  # 50に変更
   ```

2. **フレームレートを下げる**: CPU使用率が高い場合
   ```python
   time.sleep(0.2)  # 5 FPSに変更
   ```

3. **解像度を下げる**: 画面サイズを小さくする

## 🔒 セキュリティ

### 注意事項

- **暗号化なし**: 現在のバージョンでは通信は暗号化されていません
- **認証なし**: パスワード認証などのセキュリティ機能は実装されていません
- **信頼できるネットワークでのみ使用**: インターネット経由での使用は推奨しません

### セキュリティを向上させるには

1. **VPNを使用**: インターネット経由で接続する場合はVPNを使用
2. **ポートを変更**: デフォルトポートを変更
3. **ファイアウォール設定**: 特定のIPアドレスのみ接続を許可

## 🆚 既存ソフトとの比較

| 機能 | カスタムシステム | TeamViewer | AnyDesk |
|------|------------------|------------|---------|
| 画面共有 | ✅ | ✅ | ✅ |
| リモート制御 | ✅ | ✅ | ✅ |
| 暗号化 | ❌ | ✅ | ✅ |
| 認証 | ❌ | ✅ | ✅ |
| ファイアウォール通過 | ⚠️ 要設定 | ✅ | ✅ |
| 大学WiFi対応 | ⚠️ 要設定 | ✅ | ✅ |
| 無料 | ✅ | ✅（個人） | ✅（個人） |

## 💡 使用例

### 例1: 同じネットワーク内で接続

1. **デスクトップ側**
   ```bash
   python custom_remote_desktop.py server
   ```

2. **ノートパソコン側**
   ```bash
   python custom_remote_desktop.py client 192.168.1.100
   ```

### 例2: インターネット経由で接続（VPN使用）

1. **VPNに接続**（両方のPC）

2. **デスクトップ側**
   ```bash
   python custom_remote_desktop.py server
   ```

3. **ノートパソコン側**
   ```bash
   python custom_remote_desktop.py client [VPNのIPアドレス]
   ```

## 📝 既存ソフトの自動起動ラッパー

`RemoteDesktopAutoLauncher`クラスを使用して、TeamViewerやAnyDeskを自動起動できます：

```python
from custom_remote_desktop import RemoteDesktopAutoLauncher

# TeamViewerを自動起動
success, message = RemoteDesktopAutoLauncher.launch_teamviewer("123456789", "password")

# AnyDeskを自動起動
success, message = RemoteDesktopAutoLauncher.launch_anydesk("123456789")
```

## 🎓 学習ポイント

このコードから学べること：

1. **ソケット通信**: TCPソケットを使った通信
2. **画面キャプチャ**: mssライブラリを使った画面キャプチャ
3. **画像圧縮**: JPEG圧縮によるデータサイズ削減
4. **マルチスレッド**: 非同期処理によるリアルタイム通信
5. **GUI開発**: tkinterを使ったGUIアプリケーション

## 🔄 今後の改善案

- [ ] 暗号化通信（TLS/SSL）
- [ ] パスワード認証
- [ ] 複数クライアント対応
- [ ] ファイル転送機能
- [ ] 音声転送機能
- [ ] クリップボード共有

---

**注意**: このシステムは教育目的で作成されています。本格的なリモートデスクトップ用途には、TeamViewerやAnyDeskなどの既存ソフトウェアの使用を推奨します。






