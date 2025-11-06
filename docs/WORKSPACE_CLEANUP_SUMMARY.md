# ワークスペース整理サマリー

## 実施日
2025年1月（整理実施）

## 整理内容

### 1. 一時ファイル・テストファイルの削除
以下のファイルを削除しました：
- `test_*.py` - テストファイル（5ファイル）
- `check_github_token.py` - 一時的なチェックスクリプト
- `diagnose_startup.py`, `diagnose_output.txt` - 診断ファイル
- `app_error_log.txt`, `app_startup.log` - ログファイル
- `camera_history.json`, `feedback_data.json` - 一時データファイル
- `URGENT_TOKEN_REVOCATION.txt` - 緊急通知ファイル（不要）
- `remove_token_from_history.*` - 一時スクリプト

### 2. ファイルの整理・移動

#### モデルファイル
- `clear_sparse_*.h5` → `models/` に移動
- `retrain_sparse_*.h5` → `models/` に移動
- `clear_sparse_training_log_*.csv` → `logs/` に移動
- `clear_sparse_ensemble_4class_info.json` → `models/` に移動

#### ドキュメント
- `AUTO_SETUP_GUIDE.md` → `docs/` に移動
- `ENABLE_GPU_WINDOWS.md` → `docs/` に移動
- `GPU_*.md` → `docs/` に移動
- `INSTALLATION_STATUS.md` → `docs/` に移動
- `TRAINING_START_GUIDE.md` → `docs/` に移動
- `USAGE_GUIDE.md` → `docs/` に移動
- `WSL2_USAGE.md` → `docs/` に移動
- `README_*.md` → `docs/` に移動
- `使い方_リモートアクセス.txt` → `docs/` に移動

#### スクリプトの整理
- `copy_*.py`, `move_*.ps1`, `sync_*.ps1` → `scripts/utils/` に移動
- `setup_*.bat`, `setup_*.py`, `setup_*.ps1`, `setup_*.sh` → `scripts/setup/` に移動
- `start_*.bat`, `start_*.ps1` → `scripts/start/` に移動
- `check_*.py`, `check_*.bat`, `check_*.sh` → `scripts/check/` に移動
- `monitor_app.*` → `scripts/` に移動
- `update_paths.ps1` → `scripts/utils/` に移動

### 3. 設定ファイルの整理
- `github_auto_commit_config.json` (ルート) → 削除（`config/` に既に存在）
- `.gitignore` に `.specstory/` を追加
- `.gitignore` に `config/github_auto_commit_config.json` を追加（トークン保護）

### 4. 古いファイルの削除
- `old/` フォルダー全体を削除（参照されていない古いコード）

## 整理後のディレクトリ構造

```
WasherInspection/
├── config/          # 設定ファイル
├── docs/            # ドキュメント（統合）
├── scripts/         # スクリプト（整理済み）
│   ├── check/       # チェックスクリプト
│   ├── setup/       # セットアップスクリプト
│   ├── start/       # 起動スクリプト
│   └── utils/       # ユーティリティスクリプト
├── models/          # モデルファイル（整理済み）
├── logs/            # ログファイル（整理済み）
├── inspectors/      # 検査システム
├── trainers/        # 学習システム
├── dashboard/       # ダッシュボード
├── github_tools/    # GitHub関連ツール
└── main.py          # メインアプリケーション
```

## 改善点

1. **ファイル配置の明確化**: 関連ファイルを適切なディレクトリに整理
2. **重複の削減**: 重複するスクリプトやドキュメントを統合
3. **セキュリティ向上**: トークンを含む設定ファイルをGitから除外
4. **保守性向上**: 古いファイルや不要なファイルを削除

## 注意事項

- `config/github_auto_commit_config.json` はローカルにのみ存在（Gitには含まれません）
- 移動したスクリプトのパス参照を更新する必要がある場合があります
- `old/` フォルダーは完全に削除されました（必要に応じてGit履歴から復元可能）

