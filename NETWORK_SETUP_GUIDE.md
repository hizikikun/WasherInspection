# ネットワーク対応ワッシャー検査システム セットアップガイド

## 概要
デスクトップPCとノートPC間でリアルタイムにデータを送受信する検査システムです。

## システム構成
- **デスクトップPC**: WebSocketサーバーとして動作
- **ノートPC**: WebSocketクライアントとして動作
- **通信**: WebSocket (ポート8765)

## セットアップ手順

### 1. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 2. ネットワーク設定
`network_config.json`でサーバーのIPアドレスを設定：
```json
{
    "network_settings": {
        "server_host": "192.168.1.100",  // デスクトップPCのIPアドレス
        "server_port": 8765
    }
}
```

### 3. 起動方法

#### デスクトップPC（サーバー）で実行：
```bash
# 方法1: バッチファイル
start_desktop_server.bat

# 方法2: 直接実行
python network_washer_inspection_system.py server
```

#### ノートPC（クライアント）で実行：
```bash
# 方法1: バッチファイル
start_notebook_client.bat

# 方法2: 直接実行
python network_washer_inspection_system.py client
```

#### 自動検出モード：
```bash
# 方法1: バッチファイル
start_auto.bat

# 方法2: 直接実行
python network_washer_inspection_system.py auto
```

## 機能

### データ同期
- **検査結果**: リアルタイムで検査結果を送受信
- **パフォーマンス統計**: FPS、処理時間、フレーム数
- **システム情報**: CPU、メモリ、GPU情報

### 表示情報
- 検査結果（Good/NG）
- ネットワーク接続状態
- パフォーマンス統計
- システム情報

### 操作方法
- **ESC**: 終了
- **1**: Good フィードバック
- **2**: Black Spot フィードバック
- **3**: Chipped フィードバック
- **4**: Scratched フィードバック

## トラブルシューティング

### 接続できない場合
1. ファイアウォール設定を確認
2. IPアドレスが正しいか確認
3. ポート8765が使用可能か確認

### パフォーマンスが悪い場合
1. システム設定を確認（notebook/desktop）
2. フレームスキップ間隔を調整
3. 処理解像度を下げる

## 設定ファイル

### network_config.json
```json
{
    "network_settings": {
        "server_host": "192.168.1.100",
        "server_port": 8765,
        "auto_detect_ip": true
    },
    "data_sync": {
        "inspection_results": true,
        "performance_stats": true,
        "system_info": true,
        "real_time_updates": true
    }
}
```

## ログ出力例
```
[NOTEBOOK] Notebook PC detected
[OPTIMIZE] Applying notebook optimizations...
[CLIENT] Connecting to server at ws://192.168.1.100:8765
[CLIENT] Connected to server
[CLIENT] Received inspection result: good (0.85)
```
