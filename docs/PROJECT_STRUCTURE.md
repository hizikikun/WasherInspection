# WasherInspection プロジェクト構造

## 📁 ディレクトリ構成

```
WasherInspection/
├── 📄 main.py                          # メインシステム（メインの検査システム）
├── 📄 camera_inspection.py             # カメラ検査システム
│
├── 📁 scripts/                         # 学習・分析スクリプト
│   ├── train_4class_sparse_ensemble.py
│   ├── train_2class_with_augmentation.py
│   ├── train_2class_ensemble.py
│   ├── train_4class_ensemble.py
│   └── ...
│
├── 📁 github_tools/                    # GitHub統合ツール
│   ├── github_auto_commit_system.py
│   ├── integrated_github_system.py
│   ├── code_training_auto_sync.py
│   ├── cursor_github_integration.py
│   ├── auto_github_token_creator.py
│   └── ...
│
├── 📁 config/                          # 設定ファイル
│   ├── github_config.json
│   ├── cursor_github_config.json
│   ├── auto_sync_config.json
│   └── ...
│
├── 📁 cs_AItraining_data/              # 学習データ（1,461ファイル）
│   └── resin/
│       ├── good/
│       ├── black_spot/
│       ├── chipping/
│       └── scratch/
│
├── 📁 backup/                          # バックアップファイル
│
├── 📁 .github/                         # GitHub Actions設定
│   └── workflows/
│
├── 📄 requirements.txt                 # Python依存関係
├── 📄 README.md                        # プロジェクト説明
├── 📄 .gitignore                       # Git除外設定
└── 📄 PROJECT_STRUCTURE.md            # このファイル
```

## 🎯 主要ファイル説明

### メインシステム
- **main.py**: メインのワッシャー検査システム
- **camera_inspection.py**: カメラ選択・検査システム

### AI学習スクリプト（scripts/）
- **train_4class_sparse_ensemble.py**: 4クラス分類（スパースモデリング）
- **train_2class_with_augmentation.py**: 2クラス分類（データ拡張付き）
- **train_2class_ensemble.py**: 2クラス分類アンサンブル学習
- **train_4class_ensemble.py**: 4クラス分類アンサンブル学習

### GitHub統合ツール（github_tools/）
- **github_sync.py**: 統合GitHub同期システム
- **github_autocommit.py**: 自動コミットシステム
- **auto_sync.py**: コード・データ自動同期

### 設定ファイル（config/）
- **github_config.json**: GitHub設定
- **cursor_github_config.json**: Cursor連携設定
- **auto_sync_config.json**: 自動同期設定

## 🚀 使用方法

### 1. メインシステム起動
```bash
python main.py
# または
python camera_inspection.py
```

### 2. AI学習実行
```bash
cd scripts
python train_4class_sparse_ensemble.py
```

### 3. GitHub自動同期起動
```bash
cd github_tools
python github_sync.py
```

## 📊 データ統計

- **学習データ**: 1,461ファイル (565.03 MB)
  - good: 1,144ファイル (431.64 MB)
  - black_spot: 88ファイル (30.17 MB)
  - chipping: 117ファイル (28.67 MB)
  - scratch: 112ファイル (74.56 MB)

