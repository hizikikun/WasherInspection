# GPU/CPU負荷レベル選択ガイド

## 🎯 概要

学習時にGPU/CPUをどれくらいの負荷で使用するかを選択できます。

## 🚀 使い方

### 基本的な使い方

```bash
python scripts/start_training_with_resource_selection.py
```

### 選択できる負荷レベル

#### [1] 軽量（省エネモード）
- **用途**: 他の作業をしながら学習したい場合
- **GPU**: 軽度使用
- **CPU**: 軽度使用
- **Batch Size**: 8
- **Workers**: 1
- **Max Epochs**: 100

#### [2] 標準（バランス）
- **用途**: バランスの取れた学習
- **GPU**: 中度使用
- **CPU**: 中度使用
- **Batch Size**: 16
- **Workers**: 4
- **Max Epochs**: 150

#### [3] 高性能（推奨）
- **用途**: 速度と精度のバランスが良い
- **GPU**: 高度使用
- **CPU**: 高度使用
- **Batch Size**: 32
- **Workers**: 8
- **Max Epochs**: 200

#### [4] 最大性能（フル活用）
- **用途**: 最高の精度と速度を追求
- **GPU**: 最大使用
- **CPU**: 最大使用
- **Batch Size**: 64
- **Workers**: 16
- **Max Epochs**: 300

## 📊 各レベルの特徴

| レベル | Batch Size | Workers | Max Epochs | GPU使用率 | CPU使用率 | 学習時間 | 推奨 |
|--------|-----------|---------|------------|----------|----------|---------|------|
| 1 | 8 | 1 | 100 | 低 | 低 | 長い | 他の作業中 |
| 2 | 16 | 4 | 150 | 中 | 中 | 中程度 | 標準 |
| 3 | 32 | 8 | 200 | 高 | 高 | 短い | **推奨** |
| 4 | 64 | 16 | 300 | 最大 | 最大 | 最短 | 最高性能 |

## ⚙️ 設定の詳細

### Batch Size（バッチサイズ）
- **大きい**: より多くの画像を一度に処理、メモリ使用量が多い
- **小さい**: メモリ使用量が少ないが、学習が遅い

### Workers（ワーカー数）
- **多い**: データ読み込みが速い、CPU使用率が高い
- **少ない**: CPU使用率が低いが、データ読み込みが遅い

### Max Epochs（最大エポック数）
- **多い**: より長く学習、精度向上の可能性
- **少ない**: 早期終了、時間短縮

## 🔧 環境変数で直接設定

リソース選択スクリプトを使わずに、環境変数で直接設定することもできます：

```bash
# Windows PowerShell
$env:TRAINING_BATCH_SIZE = "32"
$env:TRAINING_WORKERS = "8"
$env:TRAINING_MAX_QUEUE_SIZE = "20"
$env:TRAINING_USE_MULTIPROCESSING = "True"
$env:TRAINING_MAX_EPOCHS = "200"
$env:TRAINING_PATIENCE = "30"
$env:TRAINING_USE_MIXED_PRECISION = "True"
python scripts/train_4class_sparse_ensemble.py
```

```bash
# Linux/Mac
export TRAINING_BATCH_SIZE=32
export TRAINING_WORKERS=8
export TRAINING_MAX_QUEUE_SIZE=20
export TRAINING_USE_MULTIPROCESSING=True
export TRAINING_MAX_EPOCHS=200
export TRAINING_PATIENCE=30
export TRAINING_USE_MIXED_PRECISION=True
python scripts/train_4class_sparse_ensemble.py
```

## ⚠️ 注意事項

### Windowsの場合
- `use_multiprocessing`を`True`にすると、エラーが発生する可能性があります
- デフォルトでは`False`に設定されています
- レベル2以上では`True`になりますが、エラーが発生した場合は手動で`False`に設定してください

### メモリ不足の場合
- Batch Sizeを小さくする
- Workersを減らす
- Max Queue Sizeを小さくする

### GPU使用率が低い場合
- Batch Sizeを大きくする
- レベルを上げる（3または4）

## 📈 推奨設定

### 一般的な用途
**レベル3（高性能）を推奨**
- 速度と精度のバランスが良い
- GPU/CPUを効率的に使用

### 最高精度を求めるとき
**レベル4（最大性能）**
- フル精度で学習（混合精度なし）
- 最大エポック数（300）
- より多くのデータ拡張

### 他の作業をしながら学習
**レベル1（軽量）**
- リソースを抑える
- 他のアプリケーションが使用可能

## 🎯 次のステップ

1. **リソースレベルを選択**
   ```bash
   python scripts/start_training_with_resource_selection.py
   ```

2. **学習を開始**
   - 選択された設定で自動的に学習が開始されます

3. **進捗を確認**
   - ビューアーでリアルタイムに進捗を確認できます




















