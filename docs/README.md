# Washer Inspection System

樹脂ワッシャー検査システム - AI搭載の高精度欠陥検出システム

## 🎯 概要

このプロジェクトは、樹脂ワッシャーの品質検査を自動化するAI搭載システムです。複数のカメラを使用してリアルタイムで欠陥を検出し、4クラス分類（good, black_spot, chipping, scratch）を行います。

## ✨ 主な機能

- **マルチカメラ対応**: 複数のカメラを同時起動・選択可能
- **フルHD対応**: C920n Pro HD webcam等の高解像度カメラに対応
- **AI欠陥検出**: EfficientNetベースの4クラス分類システム
- **スパースモデリング**: L1/L2正則化による高精度モデル
- **アンサンブル学習**: 複数モデルの組み合わせによる高精度
- **リアルタイム処理**: ライブカメラフィードでの即座な判定

## 🚀 クイックスタート

### 1. 環境構築
```bash
# 仮想環境作成
python -m venv washer_env
washer_env\Scripts\activate

# 依存関係インストール
pip install -r requirements.txt
```

### 2. カメラ検査システム起動
```bash
# メインの検査システム
python camera_inspection.py
```

### 3. AI学習実行
```bash
# 4クラス分類学習（スパースモデリング）
cd scripts
python train_4class_sparse_ensemble.py
```

## 📁 プロジェクト構成

### メインファイル
- `main.py` - 統合システム
- `camera_inspection.py` - カメラ検査システム

### AI学習システム（scripts/）
- `train_4class_sparse_ensemble.py` - 4クラス分類（スパースモデリング）
- `train_2class_with_augmentation.py` - 2クラス分類（データ拡張付き）
- `train_2class_ensemble.py` - 2クラス分類アンサンブル学習
- `train_4class_ensemble.py` - 4クラス分類アンサンブル学習

### カメラシステム
- `color_camera_fix.py` - カメラ色調整
- `robust_color_camera.py` - 堅牢なカメラ初期化
- `high_resolution_camera_step2.py` - 高解像度対応

## 🧠 AIモデル

### 学習済みモデル
- **EfficientNetB0**: 軽量で高速
- **EfficientNetB1**: バランス型
- **EfficientNetB2**: 高精度型

### 特徴
- **スパースモデリング**: L1/L2正則化による過学習防止
- **データ拡張**: 回転、シフト、ズーム等の空間変換
- **クラス重み調整**: 不均衡データセット対応

## 📊 データセット

### 4クラス分類
- `good`: 正常品 (1,144枚)
- `black_spot`: 黒点欠陥 (88枚)
- `chipping`: 欠け欠陥 (117枚)
- `scratch`: 傷欠陥 (112枚)

## 🔧 設定

### カメラ設定
- **解像度**: フルHD (1920x1080) 対応
- **フレームレート**: 30fps
- **バックエンド**: DirectShow, MSMF, ANY

### AI設定
- **入力サイズ**: 224x224, 240x240, 260x260
- **バッチサイズ**: 16
- **エポック数**: 200
- **学習率**: 0.001

## 📈 性能

### 精度
- **アンサンブル精度**: 80.78%
- **個別モデル精度**: 75-85%
- **推論速度**: リアルタイム処理

### 検出能力
- **ワッシャー検出**: 複数手法による堅牢な検出
- **欠陥分類**: 4クラス高精度分類
- **信頼度**: 動的閾値調整

## 🛠️ 技術スタック

- **Python 3.8+**
- **OpenCV**: カメラ処理・画像処理
- **TensorFlow/Keras**: 深層学習
- **EfficientNet**: 画像分類モデル
- **scikit-learn**: 機械学習ユーティリティ

## 📝 使用方法

### 1. カメラ検査
```bash
python camera_inspection.py
```
- 全カメラを起動
- グリッド表示でカメラ選択
- リアルタイム欠陥検出

### 2. AI学習
```bash
cd scripts
python train_4class_sparse_ensemble.py
```
- 4クラス分類学習
- スパースモデリング適用
- アンサンブル学習実行

## 🔍 トラブルシューティング

### カメラ問題
- **白黒表示**: カメラ設定を確認
- **解像度低下**: フルHD設定を有効化
- **エラー**: 複数バックエンドを試行

### AI問題
- **低精度**: データ拡張を調整
- **過学習**: スパースモデリングを強化
- **メモリ不足**: バッチサイズを削減

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 貢献

プルリクエストやイシューの報告を歓迎します。

## 📞 サポート

問題が発生した場合は、GitHubのIssuesで報告してください。
