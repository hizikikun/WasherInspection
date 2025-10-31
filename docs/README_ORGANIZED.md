# ワッシャー不良品検出システム

## ディレクトリ構造

### trainers/ - 学習システム
- `six_class_trainer.py` - 6クラス学習システム（推奨）
- `four_class_trainer.py` - 4クラス学習システム
- `binary_trainer.py` - 2クラス学習システム
- `high_quality_trainer.py` - 高品質学習システム
- `optimized_trainer.py` - 最適化学習システム
- `basic_trainer.py` - 基本学習システム

### inspectors/ - 検査システム
- `six_class_inspector.py` - 6クラス検査システム（推奨）
- `camera_inspector.py` - カメラ検査システム
- `realtime_inspector.py` - リアルタイム検査システム
- `multi_camera_inspector.py` - 複数カメラ検査システム

### utilities/ - ユーティリティ
- `install_dependencies.py` - 依存関係インストール
- `run_training.py` - 学習実行スクリプト
- `generate_samples.py` - サンプルデータ生成
- `system_checker.py` - システムチェック

## 使用方法

### 学習の実行
```bash
# 6クラス学習（推奨）
python trainers/six_class_trainer.py

# 4クラス学習
python trainers/four_class_trainer.py
```

### 検査の実行
```bash
# 6クラス検査（推奨）
python inspectors/six_class_inspector.py --camera

# 単一画像検査
python inspectors/six_class_inspector.py image.jpg
```

## 検出可能な不良品
1. 良品 (good)
2. 黒点 (black_spot)
3. 欠け (chipping)
4. 傷 (scratch)
5. 歪み (distortion)
6. 凹み (dent)
