# 進捗ビューアー + リソース選択統合ガイド

## ✅ 実装完了

進捗ビューアー（HWiNFO統合）とリソース選択システムを統合しました。

## 🎯 統合内容

### 1. 学習スクリプトにリソース選択を統合

`train_4class_sparse_ensemble.py`を実行すると：

1. **リソースレベル選択画面が表示**
   - [1] 軽量（省エネモード）
   - [2] 標準（バランス）
   - [3] 高性能（推奨）
   - [4] 最大性能（フル活用）

2. **選択された設定が適用**
   - Batch Size
   - Workers
   - Max Epochs
   - などが環境変数に設定

3. **進捗ビューアーを自動起動**
   - HWiNFOから直接使用率を取得
   - リアルタイムで表示

4. **学習を開始**

## 🚀 使い方

### 基本的な使い方

```bash
python scripts/train_4class_sparse_ensemble.py
```

実行すると：
1. リソース選択画面が表示される
2. 1-4の数字を入力して選択
3. 進捗ビューアーが自動起動（HWiNFO統合済み）
4. 学習が開始される

### 別のスクリプトから実行

```bash
python scripts/train_with_viewer_and_resource_selection.py
```

## 📊 動作フロー

```
学習開始
  ↓
リソース選択（1-4）
  ↓
設定を環境変数に適用
  ↓
進捗ビューアー起動（HWiNFO統合）
  ├─ HWiNFOから直接使用率取得（優先）
  └─ JSONから読み込み（フォールバック）
  ↓
学習開始
  ├─ 選択された設定で学習
  └─ HWiNFOからシステムメトリクス取得
  ↓
進捗ビューアーでリアルタイム表示
  ├─ GPU/CPU使用率（HWiNFOから）
  ├─ GPU温度・電力（HWiNFOから）
  └─ 学習進捗・精度など
```

## 🎯 リソースレベル詳細

### [1] 軽量（省エネモード）
- **Batch Size**: 8
- **Workers**: 1
- **Max Epochs**: 100
- **用途**: 他の作業をしながら学習

### [2] 標準（バランス）
- **Batch Size**: 16
- **Workers**: 4
- **Max Epochs**: 150
- **用途**: バランス重視

### [3] 高性能（推奨）
- **Batch Size**: 32
- **Workers**: 8
- **Max Epochs**: 200
- **用途**: 速度と精度のバランス ⭐推奨

### [4] 最大性能（フル活用）
- **Batch Size**: 64
- **Workers**: 16
- **Max Epochs**: 300
- **用途**: 最高の精度と速度を追求

## 📈 HWiNFO統合の利点

### 進捗ビューアーで表示される情報

1. **CPU使用率** - HWiNFOから直接取得
2. **CPU温度** - HWiNFOから直接取得
3. **メモリ使用率** - HWiNFOから直接取得
4. **GPU使用率** - HWiNFOから直接取得（最も信頼できる）
5. **GPU温度** - HWiNFOから直接取得
6. **GPU電力** - HWiNFOから直接取得
7. **GPUメモリ** - HWiNFOから直接取得

### 更新頻度
- **2秒ごとに更新**
- HWiNFOから直接取得するため、リアルタイム性が高い

## 🔧 環境変数による設定

リソース選択を使わずに、環境変数で直接設定することも可能：

```bash
# PowerShell
$env:TRAINING_BATCH_SIZE = "32"
$env:TRAINING_WORKERS = "8"
$env:TRAINING_MAX_EPOCHS = "200"
python scripts/train_4class_sparse_ensemble.py
```

この場合、リソース選択画面はスキップされます。

## ⚠️ 注意事項

1. **HWiNFOの起動**
   - HWiNFOが起動している必要があります
   - Shared Memory Supportが有効である必要があります

2. **Windowsでのmultiprocessing**
   - レベル2以上では`use_multiprocessing=True`になります
   - エラーが発生する場合は、手動で環境変数を設定してください

3. **メモリ不足**
   - レベル4（最大性能）ではメモリ使用量が多くなります
   - エラーが発生する場合は、レベルを下げてください

## 📝 まとめ

- ✅ リソース選択システム統合済み
- ✅ 進捗ビューアーにHWiNFO統合済み
- ✅ 2秒ごとにHWiNFOから直接取得
- ✅ リアルタイムで使用率を表示

これで、**リソース選択 → ビューアー起動（HWiNFO統合） → 学習開始**の流れが完成しました！



















