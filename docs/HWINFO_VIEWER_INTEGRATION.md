# 進捗ビューアーのHWiNFO統合

## ✅ 実装完了

進捗ビューアーの使用率表示が**HWiNFOから直接取得**するように更新しました。

## 🎯 変更内容

### 1. 学習スクリプト側（既に実装済み）
- `train_4class_sparse_ensemble.py`の`sample_system_metrics()`メソッド
- HWiNFOを**最優先**で使用
- 取得したデータを`logs/training_status.json`に保存

### 2. 進捗ビューアー側（新規実装）
- `dashboard/qt_status_viewer.py`（PyQt5版）
- `dashboard/status_viewer_app.py`（Tkinter版）
- **HWiNFOから直接取得**する機能を追加
- JSONから読み込んだ値よりも**HWiNFOの値を優先**

## 🔄 動作の流れ

```
HWiNFO Shared Memory
    ↓
進捗ビューアーが直接読み取り（優先）
    ↓
学習スクリプトもHWiNFOから読み取り
    ↓
JSONファイルに保存（フォールバック用）
    ↓
ビューアーがJSONから読み取り（HWiNFO取得失敗時のみ）
```

## 📊 取得する情報

### HWiNFOから直接取得（優先）
- **CPU使用率** (`cpu_percent`)
- **CPU温度** (`cpu_temp_c`)
- **メモリ使用率** (`mem_percent`)
- **メモリ使用量** (`mem_used_mb`, `mem_total_mb`)
- **GPU使用率** (`gpu_util_percent`) ← **最も重要**
- **GPUメモリ** (`gpu_mem_used_mb`, `gpu_mem_total_mb`)
- **GPU温度** (`gpu_temp_c`)
- **GPU電力** (`gpu_power_w`)

### フォールバック（HWiNFO取得失敗時）
- JSONファイルから読み込み
- またはpsutil/pynvmlから取得

## 🎯 メリット

1. **より正確な値**
   - HWiNFOは最も信頼できる情報源
   - GPU使用率などの精度が向上

2. **リアルタイム更新**
   - 2秒ごとにHWiNFOから直接取得
   - JSONファイルの更新を待つ必要がない

3. **冗長性**
   - HWiNFO取得失敗時はJSONから読み込み
   - システムがクラッシュしない

## ⚙️ 動作確認

### 起動時のメッセージ
進捗ビューアー起動時に以下のメッセージが表示されます：

```
進捗ビューアー: HWiNFO統合有効（使用率はHWiNFOから直接取得）
```

または

```
進捗ビューアー: HWiNFO統合無効（JSONから読み込み）
```

### 確認方法
1. HWiNFOが起動していることを確認
2. HWiNFOの「Shared Memory Support」が有効になっていることを確認
3. 進捗ビューアーを起動
4. GPU使用率などが正しく表示されることを確認

## 🔧 トラブルシューティング

### HWiNFOが取得できない場合
1. HWiNFOが起動しているか確認
2. Shared Memory Supportが有効か確認
3. HWiNFOを管理者権限で実行しているか確認
4. 自動再起動システムが動作しているか確認

### 値が表示されない場合
- JSONファイルからフォールバック読み込み
- 学習スクリプトがHWiNFOから取得した値を使用

## 📝 注意事項

- HWiNFOは**最も信頼できる情報源**として使用されます
- JSONファイルの値よりも**常に優先**されます
- 正常な範囲外の値（異常値）は除外されます
  - GPU使用率: 0-100%
  - GPU温度: 20-150℃
  - GPU電力: 0-1000W




















