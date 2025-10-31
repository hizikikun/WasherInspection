# 全ファイルテフォルダー整理完了報告

## ✅ 実施内容

### 1. ルートディレクトリのファイル整理

#### Pythonスクリプト（26個）
- **scripts/legacy/**: 古い学習スクリプト 14個
  - correct_path_deep_learning.py
  - correct_path_defect_classifier.py
  - create_dummy_model.py
  - direct_sample_deep_learning.py
  - existing_data_defect_classifier.py
  - image_data_deep_learning.py
  - no_visualization_deep_learning.py
  - real_image_deep_learning.py
  - real_path_defect_classifier.py
  - retrain_from_feedback.py
  - retrain_improved.py
  - resin_washer_deep_learning.py
  - resin_washer_dl.py
  - resin_washer_trainer.py

- **scripts/config/**: 設定と整理スクリプト 6個
  - check_and_train.py
  - organize_files.py
  - organize_github_files.py
  - path_diagnosis.py
  - rename_files.py
  - rename_remaining_files.py

- **tools/**: ツールスクリプト 4個
  - gh_cli_token_creator.py
  - simple_token_setup.py
  - unified_token_creator.py
  - fix_commit_messages.py

- **inspectors/**: 検査関連 2個
  - manual_black_spot_detection.py
  - interactive_learning_system.py

#### バッチファイル（12個）
- **batch/**: すべての.batファイル
  - auto_create_token.bat
  - force_commit.bat
  - start_auto.bat
  - start_cursor_github.bat
  - start_cursor_gui.bat
  - start_desktop_server.bat
  - start_notebook_client.bat
  - start-auto-commit.bat
  - github_sync_once.bat（batch_files/から）
  - start_auto_sync.bat（batch_files/から）
  - start_github_auto_commit.bat（batch_files/から）
  - sync_once.bat（batch_files/から）

#### 設定ファイル（3個）
- **config/**: 設定ファイル
  - github_auto_commit_config.json
  - network_config.json
  - setup-auto-commit.xml

#### PowerShellスクリプト（4個）
- **docs/setup/**: セットアップ用PowerShellスクリプト
  - auto-commit.ps1
  - fix_all_commits.ps1
  - test-encoding.ps1
  - test-japanese.ps1

#### データファイル（2個）
- **temp/**: テストファイル
  - test-file.txt
- **models/legacy/**: 古いモデル
  - best_real_defect_model.h5

#### specファイル（1個）
- **build/**: PyInstaller specファイル
  - ResinWasherInspection.spec

### 2. 整理後の構造

```
WasherInspection/
│
├── batch/              # すべてのバッチファイル
├── config/             # すべての設定ファイル
├── docs/
│   ├── setup/         # セットアップ用PowerShell
│   └── (全ドキュメント)
├── models/
│   ├── ensemble/       # アンサンブルモデル
│   ├── sparse/        # スパースモデル
│   ├── corrected/      # 修正済みモデル
│   └── legacy/        # 古いモデル
├── scripts/
│   ├── train_4class_sparse_ensemble.py  # メイン学習
│   ├── training/      # 学習スクリプト
│   ├── utils/         # ユーティリティ
│   ├── git/           # Git関連
│   ├── hwinfo/        # HWiNFO関連
│   ├── config/        # 設定と整理スクリプト
│   └── legacy/        # 古いスクリプト
├── tools/             # ツールスクリプト
├── inspectors/         # 検査関連（全スクリプト）
├── utilities/         # ユーティリティ
├── trainers/          # トレーナー
├── github_tools/      # GitHubツール
├── dashboard/         # ダッシュボード
├── main.py           # メインアプリケーション
├── requirements.txt  # 依存パッケージ
└── README.md         # プロジェクト説明
```

### 3. 整理結果まとめ

| カテゴリ | 移動数 | 移動先 |
|---------|--------|--------|
| Pythonスクリプト | 26個 | scripts/legacy, scripts/config, tools, inspectors |
| バッチファイル | 12個 | batch/ |
| 設定ファイル | 3個 | config/ |
| PowerShell | 4個 | docs/setup/ |
| データファイル | 2個 | temp/, models/legacy/ |
| specファイル | 1個 | build/ |
| **合計** | **48個** | |

## 📋 ルートディレクトリに残っているファイル（意図的に）

以下のファイルはルートディレクトリに残していまム：

- `main.py` - メインアプリケーション
- `requirements.txt` - 依存パッケージ定義
- `README.md` - プロジェクト説明
- `workspace.code-workspace` - VS Codeワークスペース設定
- `camera_history.json` - カメラ履歴データ
- `feedback_data.json` - フィードバックデータ

## ✅ 完了

全ファイルとフォルダーの整理が完了しました。
プロジェクト構造が明確になり、保守性が大幅に向上しました。

