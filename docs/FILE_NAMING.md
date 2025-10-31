# ファイル命名規則

## 📋 命名方針

プロジェクト内のファイルは、わかりやムく簡潔な名前を使用します。

### ❌ 避けるべき命名
- `ultra_*` - 誇張的でわかりにくい
- `final_*` - 本当に最後かどうか不明確
- `advanced_*` - 高度さが曖昧
- `clear_progress_*` - 説明が冗長
- `corrected_*` - 修正版であることが不明確
- `improved_*` - 改善の内容が不明確

### ✅ 推奨される命名

#### 学習スクリプト（scripts/）
- `train_2class_ensemble.py` - 2クラス分類アンサンブル学習
- `train_4class_ensemble.py` - 4クラス分類アンサンブル学習
- `train_4class_sparse_ensemble.py` - 4クラス分類スパースモデリング
- `train_2class_with_augmentation.py` - 2クラス分類（データ拡張付き）

命名パターン: `train_{クラス数}class_{手法}.py`

#### GitHub統合ツール（github_tools/）
- `github_sync.py` - GitHub同期メイン
- `github_autocommit.py` - 自動コミット
- `auto_sync.py` - 自動同期
- `cursor_integration.py` - Cursor統合
- `token_setup.py` - トークン設定

命名パターン: `{機能名}.py` または `{サービス}_integration.py`

#### メインシステム（ルート）
- `main.py` - メインシステム
- `camera_inspection.py` - カメラ検査システム

命名パターン: `{機能}.py` または `{システム名}.py`

## 📁 ディレクトリ構成

```
WasherInspection/
├── main.py                          # メインシステム
├── camera_inspection.py             # カメラ検査
│
├── scripts/                         # 学習スクリプト
│   ├── train_2class_ensemble.py
│   ├── train_4class_ensemble.py
│   └── train_4class_sparse_ensemble.py
│
├── github_tools/                     # GitHubツール
│   ├── github_sync.py
│   ├── github_autocommit.py
│   └── auto_sync.py
│
├── old/                              # 旧ファイル（参考用）
│   └── ...
│
└── docs/                             # ドキュメント
    └── FILE_NAMING.md
```

## 🔄 変更履歴

### 2025-01-XX: ファイル名整理
- `clear_progress_sparse_modeling_four_class_ensemble.py` → `train_4class_sparse_ensemble.py`
- `ultra_high_accuracy_ensemble.py` → `train_2class_ensemble.py`
- `integrated_github_system.py` → `github_sync.py`
- `improved_multi_camera_selection_step5.py` → `camera_inspection.py`


