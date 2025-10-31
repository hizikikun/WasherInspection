# ワッシャー不良品検出システム

## 概要
AIを活用した樹脂製ワッシャーの不良品検出システムです。6種類の不良品（良品、黒点、欠け、傷、歪み、凹み）を高精度で分類できまム。

## 特徴
- **6クラス分類**: 良品、黒点、欠け、傷、歪み、凹みを検出
- **リアルタイム検査**: カメラからの即座な判定
- **高精度学習**: EfficientNetベースのアンサンブル学習
- **使いやムいUI**: 直感的な操作インターフェース

## ディレクトリ構造

### 📁 trainers/ - 学習システム
- `six_class_trainer.py` - 6クラス学習システム（推奨）
- `high_quality_trainer.py` - 高品質学習システム
- `optimized_trainer.py` - 最適化学習システム
- `basic_trainer.py` - 基本学習システム

### 📁 inspectors/ - 検査システム
- `six_class_inspector.py` - 6クラス検査システム（推奨）
- `camera_inspector.py` - カメラ検査システム
- `realtime_inspector.py` - リアルタイム検査システム
- `multi_camera_inspector.py` - 複数カメラ検査システム

### 📁 utilities/ - ユーティリティ
- `install_dependencies.py` - 依存関係インストール
- `run_training.py` - 学習実行スクリプト
- `generate_samples.py` - サンプルデータ生成
- `system_checker.py` - システムチェック

## セットアップ

### 1. 依存関係のインストール
```bash
python utilities/install_dependencies.py
```

### 2. サンプルデータの生成
```bash
python utilities/generate_samples.py
```

## 使用方法

### 学習の実行
```bash
# 6クラス学習（推奨）
python trainers/six_class_trainer.py

# 高品質学習
python trainers/high_quality_trainer.py
```

### 検査の実行
```bash
# 6クラス検査（推奨）
python inspectors/six_class_inspector.py --camera

# 単一画像検査
python inspectors/six_class_inspector.py image.jpg

# 一括検査
python inspectors/six_class_inspector.py --batch /path/to/images/
```

## 検出可能な不良品
1. **良品 (good)** - 正常なワッシャー
2. **黒点 (black_spot)** - 黒い点状の欠陥
3. **欠け (chipping)** - 破損テ欠損
4. **傷 (scratch)** - 表面の傷
5. **歪み (distortion)** - 形状の歪み
6. **凹み (dent)** - 表面の凹み

## 技術仕様
- **フレームワーク**: TensorFlow/Keras
- **モデル**: EfficientNet (B0, B1, B2)
- **学習方式**: アンサンブル学習
- **データ拡張**: 高度な画像変換
- **最適化**: AdamW, 学習率スケジューリング

## ライセンス
MIT License

## 貢献
プルリクエストやイシューの報告を歓迎します。

## 更新履歴
- v2.0: 6クラス対応、ファイル整理、UI改善
- v1.0: 基本4クラス検出システム
