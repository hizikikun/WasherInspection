# GitHub自動送信システム セットアップガイド

## 概要
コードの変更を監視し、一定の閾値を超えたら自動的にGitHubにコミット・送信するシステムです。

## 機能

### 🔄 自動コミット機能
- **ファイル数閾値**: 指定したファイル数が変更されたら自動コミット
- **サイズ閾値**: 変更されたファイルの総サイズが閾値を超えたら自動コミット
- **時間閾値**: 一定時間経過したら自動コミット
- **インテリジェント判定**: 複数の条件を組み合わせて最適なタイミングでコミット

### 📊 データ送信機能
- **コード変更**: 変更されたファイルを自動的にGitHubに送信
- **学習データ**: 新しい学習用写真を自動的にアップロード
- **統計情報**: データセットの統計情報を自動生成・送信
- **コメント生成**: 変更内容に基づいて自動的にコメントを生成

### 🎯 閾値設定
- **ファイル数**: デフォルト5ファイル
- **サイズ**: デフォルト1MB
- **時間**: デフォルト5分

## セットアップ手順

### 1. GitHubリポジトリの準備
1. GitHubで新しいリポジトリを作成
2. Personal Access Tokenを作成（`repo`権限が必要）
3. リポジトリをクローンまたは初期化

### 2. 設定ファイルの編集
`github_auto_commit_config.json`を編集：
```json
{
  "github_token": "ghp_your_token_here",
  "github_owner": "your-username",
  "github_repo": "washer-inspection-code",
  "code_path": ".",
  "backup_path": "backup",
  "commit_threshold": 5,
  "size_threshold": 1048576,
  "time_threshold": 300,
  "auto_commit": true,
  "include_code_files": [".py", ".json", ".md", ".yml", ".yaml", ".txt"],
  "exclude_dirs": ["__pycache__", ".git", "node_modules", ".vscode", "backup"],
  "commit_message_template": "Auto-commit: {change_count} files changed",
  "branch": "main"
}
```

### 3. 依存関係のインストール
```bash
pip install requests GitPython
```

### 4. システムの起動

#### 自動監視モード（推奨）
```bash
# バッチファイル
start_github_auto_commit.bat

# 直接実行
python integrated_github_system.py
```

#### 一回だけ実行
```bash
# バッチファイル
github_sync_once.bat

# 直接実行
python integrated_github_system.py once
```

#### 強制コミット
```bash
# バッチファイル
force_commit.bat

# 直接実行
python integrated_github_system.py force
```

## 閾値設定

### ファイル数閾値
```json
{
  "commit_threshold": 10  // 10ファイル変更で自動コミット
}
```

### サイズ閾値
```json
{
  "size_threshold": 2097152  // 2MBで自動コミット
}
```

### 時間閾値
```json
{
  "time_threshold": 600  // 10分で自動コミット
}
```

## 送信されるデータ

### コード変更
- 変更されたファイルの内容
- コミットメッセージ（変更タイプとファイル名）
- バックアップファイル
- 変更統計情報

### 学習データ
- 新しい画像ファイル
- クラス別統計情報
- 日付別活動状況
- データ品質レポート
- 改善提案

### システム情報
- システムステータス
- 設定情報
- エラーログ
- パフォーマンス統計

## GitHub Issuesの例

### 自動コミット
```
Auto-commit: 8 files changed (3 new, 5 modified)

## Change Summary
- **New Files**: 3
- **Modified Files**: 5
- **Total Size**: 1.2 MB

## Files Changed
### New Files
- `new_feature.py`
- `config.json`
- `README.md`

### Modified Files
- `main.py`
- `utils.py`
- `models.py`
- `training.py`
- `config.py`

## Commit Details
- **Commit Hash**: a1b2c3d4
- **Branch**: main
- **Timestamp**: 2024-01-15 14:30:25
- **Auto-generated**: Yes
```

### 学習データ更新
```
📊 Training Data Update - 150 files

## Training Data Update
**Timestamp**: 2024-01-15 14:30:25
**Total Files**: 150
**Total Size**: 245.67 MB
**Classes**: 4

## Class Distribution
- **Good**: 120 files (180.45 MB)
- **Chipping**: 15 files (25.30 MB)
- **Black Spot**: 10 files (25.12 MB)
- **Scratch**: 5 files (14.80 MB)

## Daily Activity
- **2024-01-15**: 25 files
- **2024-01-14**: 18 files
- **2024-01-13**: 22 files

## Recommendations
- Class imbalance detected - consider collecting more data for minority classes
- Consider collecting more training data (currently < 100MB)
```

## トラブルシューティング

### GitHub接続エラー
1. Personal Access Tokenが正しいか確認
2. リポジトリ名とオーナー名が正しいか確認
3. インターネット接続を確認
4. リポジトリの権限を確認

### コミットエラー
1. Gitリポジトリが正しく初期化されているか確認
2. ファイルの権限を確認
3. ディスク容量を確認
4. ファイルがロックされていないか確認

### 閾値が機能しない
1. 設定ファイルの値が正しいか確認
2. ファイル拡張子が含まれているか確認
3. 除外ディレクトリに含まれていないか確認
4. ファイルのハッシュが正しく計算されているか確認

## カスタマイズ

### コミットメッセージのカスタマイズ
```json
{
  "commit_message_template": "Custom message: {change_count} files changed at {timestamp}"
}
```

### 監視ファイル形式の追加
```json
{
  "include_code_files": [".py", ".json", ".md", ".yml", ".yaml", ".txt", ".csv", ".xml", ".html"]
}
```

### 除外ディレクトリの追加
```json
{
  "exclude_dirs": ["__pycache__", ".git", "node_modules", ".vscode", "backup", "temp", "logs"]
}
```

## セキュリティ注意事項

1. **Personal Access Token**: 絶対に他人に教えない
2. **設定ファイル**: `github_auto_commit_config.json`を`.gitignore`に追加
3. **バックアップ**: 重要なデータは別途バックアップを取る
4. **リポジトリ権限**: 必要最小限の権限のみ付与

## パフォーマンス最適化

### 監視間隔の調整
```python
# 30秒間隔で監視（デフォルト）
time.sleep(30)

# 60秒間隔に変更
time.sleep(60)
```

### バッチサイズの調整
```json
{
  "commit_threshold": 20,  // より多くのファイルをまとめてコミット
  "size_threshold": 5242880  // 5MBでコミット
}
```

### 除外ファイルの追加
```json
{
  "exclude_dirs": ["__pycache__", ".git", "node_modules", ".vscode", "backup", "temp", "logs", "cache"]
}
```
