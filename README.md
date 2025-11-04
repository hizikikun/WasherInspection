# Washer Inspection System

樹脂ワッシャー検査システム - AI搭載の高精度欠陥検出システム

## 🎯 概要

このプロジェクトは、樹脂ワッシャーの品質検査を自動化するAI搭載システムです。複数のカメラを使用してリアルタイムで欠陥を検出し、4クラス分類（good, black_spot, chipping, scratch）を行います。

## 📁 プロジェクト構造

```
WasherInspection/
├── main.py                              # メインシステム
├── camera_inspection.py                 # カメラ検査システム
│
├── scripts/                             # AI学習スクリプト
│   ├── train_4class_sparse_ensemble.py
│   ├── train_2class_with_augmentation.py
│   └── ...
│
├── github_tools/                       # GitHub統合ツール
│   ├── github_sync.py
│   ├── github_autocommit.py
│   └── ...
│
├── config/                              # 設定ファイル
│   ├── github_config.json
│   ├── auto_sync_config.json
│   └── ...
│
├── batch_files/                         # バッチファイル
│   └── start_*.bat
│
├── docs/                                # ドキュメント
│   ├── README.md
│   ├── PROJECT_STRUCTURE.md
│   └── ...
│
└── cs_AItraining_data/                  # 学習データ（1,461ファイル）
    └── resin/
```

詳細は `docs/PROJECT_STRUCTURE.md` を参照してください。

## 🚀 クイックスタート

### 1. 環境構築
```bash
# 依存関係インストール
pip install -r requirements.txt
```

### 2. メインシステム起動
```bash
# カメラ検査システム
python camera_inspection.py
```

### 3. AI学習実行
```bash
# 4クラス分類学習（スパースモデリング）
cd scripts
python train_4class_sparse_ensemble.py
```

### 4. GitHub自動同期
```bash
# GitHub自動同期起動
cd github_tools
python github_sync.py
```

## ✨ 主な機能

- **マルチカメラ対応**: 複数のカメラを同時起動・選択可能
- **フルHD対応**: C920n Pro HD webcam等の高解像度カメラに対応
- **AI欠陥検出**: EfficientNetベースの4クラス分類システム
- **スパースモデリング**: L1/L2正則化による高精度モデル
- **アンサンブル学習**: 複数モデルの組み合わせによる高精度
- **リアルタイム処理**: ライブカメラフィードでの即座な判定
- **GitHub自動同期**: コード変更と学習データの自動同期

## 📊 データ統計

- **学習データ**: 1,461ファイル (565.03 MB)
  - good: 1,144ファイル (431.64 MB)
  - black_spot: 88ファイル (30.17 MB)
  - chipping: 117ファイル (28.67 MB)
  - scratch: 112ファイル (74.56 MB)

## 🔧 トラブルシューティング

### GitHubモジュールエラー
```bash
pip install GitPython requests
```

### カメラが認識されない
- カメラドライバーの更新
- 他のアプリケーションがカメラを使用していないか確認

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

文字化けしていないよ

