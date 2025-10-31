# WasherInspection プロジェクト構造

## 📁 ディレクトリ構成

```
WasherInspection/
│
├── models/                    # 学習済みモデル
│   ├── ensemble/              # アンサンブルモデル
│   ├── sparse/                # スパースモデル
│   └── corrected/             # 修正済みモデル
│
├── logs/                      # ログファイル
│   ├── training/              # 学習ログ
│   │   ├── sparse/            # スパース学習ログ
│   │   └── corrected/         # 修正済み学習ログ
│   └── training_status.json   # 学習ステータス
│
├── docs/                      # ドキュメント
│   └── hwinfo/                # HWiNFO関連ドキュメント
│
├── scripts/                   # スクリプト
│   ├── train_4class_sparse_ensemble.py  # メイン学習スクリプト
│   ├── training/              # その他学習スクリプト
│   ├── utils/                 # ユーティリティ
│   │   ├── auto_recovery_watcher.py
│   │   ├── training_health_checker.py
│   │   └── system_detector.py
│   ├── git/                   # Git関連スクリプト
│   └── hwinfo/                # HWiNFO関連スクリプト
│       ├── hwinfo_reader.py
│       ├── hwinfo_auto_restart.py
│       ├── hwinfo_status.py
│       └── *.bat, *.ps1       # セットアップスクリプト
│
├── inspectors/                # 検査カメラ関連
│
├── cs_AItraining_data/        # 学習データ
│
├── temp/                      # 一時ファイル
│
├── main.py                    # メインアプリケーション
└── requirements.txt           # 依存パッケージ

```

## 📋 主要ファイル説明

### 学習スクリプト
- `scripts/train_4class_sparse_ensemble.py`: メインのスパースアンサンブル学習スクリプト

### モデルファイル
- `models/sparse/`: スパースモデル（最新の高精度モデル）
- `models/ensemble/`: アンサンブルモデル
- `models/corrected/`: 修正済みモデル

### ユーティリティ
- `scripts/utils/auto_recovery_watcher.py`: 学習停止の自動検出テ再起動
- `scripts/utils/training_health_checker.py`: 学習ヘルスチェック
- `scripts/hwinfo/hwinfo_reader.py`: HWiNFOからのシステム情報取得

## 🔧 使用方法

### 学習の実行
```bash
python scripts/train_4class_sparse_ensemble.py
```

### HWiNFO連携の確認
```bash
python scripts/hwinfo/hwinfo_status.py
```

### 学習状態の確認
```bash
python scripts/utils/training_health_checker.py
```

## 📝 注意事項

- モデルファイルは `models/` 配下に分類されていまム
- ログファイルは `logs/training/` 配下に分類されていまム
- HWiNFO関連のスクリプトは `scripts/hwinfo/` に配置されていまム

