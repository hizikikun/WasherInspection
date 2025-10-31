# コードテ学習データ自動送信システム セットアップガイド

## 概要
コードの変更と学習用写真の追加を自動的に検出し、GitHubに送信するシステムです。

## 機能

### 🔄 コード変更の自動送信
- Python、JSON、Markdownファイルの変更を自動検出
- GitHubリポジトリに自動コミット
- バックアップファイルの自動作成

### 📸 学習データの自動管理
- 新しい学習用写真を自動検出
- クラス別テ日付別に自動整理
- データセット統計の自動生成
- GitHub Issuesで進捗レポート

### 📊 自動レポート生成
- 学習データの統計情報
- クラス分布の分析
- データ品質の検証
- 改善提案の自動生成

## セットアップ手順

### 1. GitHubリポジトリの準備
1. GitHubで新しいリポジトリを作成（例：`washer-inspection-code`）
2. Personal Access Tokenを作成（`repo`権限が必要）

### 2. 設定ファイルの編集
`auto_sync_config.json`を編集：
```json
{
  "github_token": "ghp_your_token_here",
  "github_owner": "your-username",
  "github_repo": "washer-inspection-code",
  "code_path": ".",
  "training_data_path": "cs_AItraining_data",
  "backup_path": "backup",
  "sync_interval_minutes": 5,
  "auto_commit": true,
  "auto_organize_training": true
}
```

### 3. 依存関係のインストール
```bash
pip install requests opencv-python
```

### 4. システムの起動

#### 自動監視モード（推奨）
```bash
# バッチファイル
start_auto_sync.bat

# 直接実行
python integrated_auto_sync.py
```

#### 一回だけ実行
```bash
# バッチファイル
sync_once.bat

# 直接実行
python integrated_auto_sync.py once
```

## 使用方法

### コード変更の監視
- 指定されたファイル形式（.py, .json, .md等）の変更を自動検出
- 変更があったファイルをGitHubに自動アップロード
- バックアップフォルダにコピーを保存

### 学習データの管理
- `cs_AItraining_data`フォルダ内の新しい画像を自動検出
- クラス別に自動分類（good, chipping, black_spot, scratch）
- 日付別に自動整理
- 統計情報を自動生成

### 手動操作
```bash
# 学習データのスキャン
python training_data_manager.py scan

# データの整理
python training_data_manager.py organize

# バランスの取れたデータセット作成
python training_data_manager.py balance 1000

# 画像の検証
python training_data_manager.py validate

# レポート生成
python training_data_manager.py report
```

## 送信されるデータ

### コード変更
- 変更されたファイルの内容
- コミットメッセージ（変更タイプとファイル名）
- バックアップファイル

### 学習データ
- 新しい画像ファイル
- クラス別統計情報
- 日付別活動状況
- データ品質レポート
- 改善提案

## GitHub Issuesの例

### 学習データ更新レポート
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

## 設定オプション

### 同期間隔の調整
```json
{
  "sync_interval_minutes": 10  // 10分間隔に変更
}
```

### 監視ファイル形式の変更
```json
{
  "include_code_files": [".py", ".json", ".md", ".yml", ".yaml", ".txt", ".csv"]
}
```

### 除外ディレクトリの設定
```json
{
  "exclude_dirs": ["__pycache__", ".git", "node_modules", ".vscode", "backup", "temp"]
}
```

## トラブルシューティング

### GitHub接続エラー
1. Personal Access Tokenが正しいか確認
2. リポジトリ名とオーナー名が正しいか確認
3. インターネット接続を確認

### ファイル監視エラー
1. ファイルパスが正しいか確認
2. ファイル権限を確認
3. ディスク容量を確認

### 学習データエラー
1. 画像ファイル形式を確認（.jpg, .jpeg, .png, .bmp）
2. ファイルサイズを確認（大きムぎるファイルは除外）
3. 画像の破損を確認

## セキュリティ注意事項

1. **Personal Access Token**: 絶対に他人に教えない
2. **設定ファイル**: `auto_sync_config.json`を`.gitignore`に追加
3. **バックアップ**: 重要なデータは別途バックアップを取る

## カスタマイズ

### 独自のクラス名を追加
```python
# training_data_manager.py内で修正
self.classes = ['good', 'chipping', 'black_spot', 'scratch', 'your_custom_class']
```

### 独自のファイル形式を追加
```json
{
  "include_code_files": [".py", ".json", ".md", ".yml", ".yaml", ".txt", ".csv", ".xml"]
}
```

### 通知機能の追加
```python
# 独自の通知機能を追加
def send_notification(self, message):
    # Slack、Discord、メール等の通知を実装
    pass
```
