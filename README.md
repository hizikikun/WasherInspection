# ワッシャー不良品検出システム

AIを活用した樹脂製ワッシャーの高精度不良品検出システムです。

## 🎯 概要

このプロジェクトは、樹脂ワッシャーの品質検査を自動化するAI搭載システムです。複数のカメラを使用してリアルタイムで欠陥を検出し、4クラス分類（良品、黒点、欠け、傷）を行います。

## ✨ 主な機能

- **マルチカメラ対応**: 複数のカメラを同時起動・選択可能
- **フルHD対応**: C920n Pro HD webcam等の高解像度カメラに対応
- **AI欠陥検出**: EfficientNetベースの4クラス分類システム
- **スパースモデリング**: L1/L2正則化による高精度モデル
- **アンサンブル学習**: 複数モデルの組み合わせによる高精度
- **リアルタイム処理**: ライブカメラフィードでの即座な判定
- **GPU/CPU選択**: リソースに応じた最適なデバイス選択

## 🚀 クイックスタート

### 1. 環境構築

```bash
# 仮想環境作成（推奨）
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 依存関係インストール
pip install -r requirements.txt
```

### 2. カメラ検査システム起動

```bash
# メインの検査システム
python main.py

# 統合ダッシュボードアプリ
python dashboard/integrated_washer_app.py
```

### 3. AI学習実行

```bash
# 4クラス分類学習（スパースモデリング・アンサンブル）
cd scripts
python train_4class_sparse_ensemble.py
```

## 📁 プロジェクト構成

```
WasherInspection/
├── main.py                    # メインアプリケーション
├── requirements.txt            # 依存関係
│
├── config/                     # 設定ファイル
│   ├── inspection_settings.json
│   └── *.json
│
├── dashboard/                  # ダッシュボード・UI
│   ├── integrated_washer_app.py
│   └── *.py
│
├── inspectors/                 # 検査システム
│   ├── camera_inspector.py
│   ├── realtime_inspector.py
│   └── *.py
│
├── scripts/                    # スクリプト
│   ├── train_4class_sparse_ensemble.py  # 学習スクリプト
│   ├── check/                 # チェックスクリプト
│   ├── setup/                 # セットアップスクリプト
│   ├── start/                 # 起動スクリプト
│   └── utils/                 # ユーティリティ
│
├── trainers/                   # 学習システム
│   └── *.py
│
├── models/                     # 学習済みモデル
│   └── *.h5
│
├── docs/                       # ドキュメント
│   ├── GITHUB_SETUP_GUIDE.md
│   ├── GPU_SETUP_GUIDE.md
│   └── *.md
│
└── github_tools/               # GitHub関連ツール
    └── *.py
```

## 🔧 セットアップ詳細

詳細なセットアップ手順は以下のドキュメントを参照してください：

- [GPU設定ガイド](docs/GPU_SETUP_GUIDE.md)
- [GitHub設定ガイド](docs/GITHUB_SETUP_GUIDE.md)
- [ネットワーク設定ガイド](docs/NETWORK_SETUP_GUIDE.md)
- [統合アプリガイド](docs/INTEGRATED_APP_GUIDE.md)

## 📊 検出可能な不良品

1. **良品 (good)** - 正常なワッシャー
2. **黒点 (black_spot)** - 黒い点状の欠陥
3. **欠け (chipping)** - 破損・欠損
4. **傷 (scratch)** - 表面の傷

## 🛠️ 技術仕様

- **フレームワーク**: TensorFlow/Keras
- **モデル**: EfficientNet (B0, B1, B2)
- **学習方式**: アンサンブル学習、スパースモデリング
- **データ拡張**: 高度な画像変換
- **最適化**: AdamW, 学習率スケジューリング
- **GPU対応**: CUDA, WSL2対応

## 📝 使用方法

### 学習の実行

```bash
# 4クラス分類学習（推奨）
python scripts/train_4class_sparse_ensemble.py

# リソース選択付き学習
python scripts/train_with_resource_selection.py
```

### 検査の実行

```bash
# カメラ検査
python main.py

# 統合ダッシュボード
python dashboard/integrated_washer_app.py
```

## 📚 ドキュメント

詳細なドキュメントは `docs/` ディレクトリを参照してください：

- [ワークスペース整理サマリー](docs/WORKSPACE_CLEANUP_SUMMARY.md)
- [Git整理サマリー](docs/GIT_CLEANUP_SUMMARY.md)
- [データ収集ガイド](docs/DATA_COLLECTION_GUIDE.md)
- [精度向上ガイド](docs/ACCURACY_IMPROVEMENT_GUIDE.md)

## 🤝 貢献

プルリクエストやイシューの報告を歓迎します。

## 📄 ライセンス

MIT License

## 🔄 更新履歴

- **v2.0** (2025-01): ワークスペース整理、ファイル構造最適化、Git履歴クリーンアップ
- **v1.0**: 基本4クラス検出システム

---

**注意**: このリポジトリは整理済みです。不要なファイル（`.specstory/`, `.venv/`, `venv_wsl2/`など）は`.gitignore`で除外されています。
