# GitHub自動送信システム セットアップガイド

## 概要
検査結果をGitHub Issuesに自動送信するシステムです。デスクトップPCとノートPCのどちらでも動作し、インターネット接続があればどこからでもアクセス可能です。

## セットアップ手順

### 1. GitHubリポジトリの作成
1. GitHubで新しいリポジトリを作成（例：`washer-inspection-results`）
2. リポジトリをPublicまたはPrivateに設定

### 2. GitHub Personal Access Tokenの作成
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. "Generate new token" → "Generate new token (classic)"
3. 以下の権限を選択：
   - `repo` (Full control of private repositories)
   - `issues` (Read and write access to issues)
4. トークンをコピーして保存

### 3. 設定ファイルの編集
`github_config.json`を編集：
```json
{
  "github_token": "ghp_your_token_here",
  "github_owner": "your-username",
  "github_repo": "washer-inspection-results",
  "auto_send": true,
  "send_interval_minutes": 5,
  "include_images": true,
  "issue_labels": ["inspection", "automated"]
}
```

### 4. 依存関係のインストール
```bash
pip install requests
```

### 5. システムの起動
```bash
python github_washer_inspection_system.py
```

## 機能

### 自動送信
- **NG検出時**: 即座にGitHub Issueを作成
- **定期送信**: 設定した間隔（デフォルト5分）で結果を送信
- **画像付き**: 検査時の画像も自動アップロード

### GitHub Issueの内容
- 検査結果（Good/NG）
- 信頼度スコア
- システム情報（CPU、メモリ、GPU）
- 統計情報（成功率、検査数）
- パフォーマンス情報（FPS、処理時間）
- 検査画像（オプション）

### GitHub Actions自動処理
- **検査結果の集計**: 自動的に統計を計算
- **定期レポート**: 6時間ごとに処理状況を確認
- **日次レポート**: 毎日の検査結果をまとめ
- **NG通知**: NG検出時に自動通知

## 操作方法

### キーボード操作
- **ESC**: 終了
- **S**: 手動でGitHubに送信
- **1**: Good フィードバック
- **2**: Black Spot フィードバック
- **3**: Chipped フィードバック
- **4**: Scratched フィードバック

### 表示情報
- 検査結果（Good/NG）
- GitHub接続状態
- 統計情報（Good数、NG数、総数）
- パフォーマンス情報（FPS、実行時間）
- システム情報

## GitHub Issuesの例

### Good検査結果
```
✅ Inspection Result: GOOD - 2024-01-15 14:30:25

## Inspection Result
**Status**: ✅ GOOD
**Defect Type**: Good
**Confidence**: 0.95
**Timestamp**: 2024-01-15 14:30:25

## System Information
- **System Type**: DESKTOP
- **CPU**: Intel Core i7-12700K (12 cores)
- **Memory**: 32.0 GB
- **GPU**: NVIDIA GeForce RTX 4070

## Statistics
- **Total Inspections**: 150
- **Good Count**: 142
- **NG Count**: 8
- **Success Rate**: 94.7%
```

### NG検査結果
```
❌ Inspection Result: NG - BLACK_SPOT - 2024-01-15 14:35:10

## Inspection Result
**Status**: ❌ NG
**Defect Type**: Black Spot
**Confidence**: 0.87
**Timestamp**: 2024-01-15 14:35:10

## System Information
- **System Type**: NOTEBOOK
- **CPU**: Intel Core i7-1165G7 (4 cores)
- **Memory**: 32.0 GB
- **GPU**: No NVIDIA GPU detected

## Statistics
- **Total Inspections**: 151
- **Good Count**: 142
- **NG Count**: 9
- **Success Rate**: 94.0%

## Inspection Image
![Inspection Image](https://github.com/user/washer-inspection-results/raw/main/images/inspection_2024-01-15_14-35-10.jpg)
```

## トラブルシューティング

### GitHub接続エラー
1. Personal Access Tokenが正しいか確認
2. リポジトリ名とオーナー名が正しいか確認
3. インターネット接続を確認

### 画像アップロードエラー
1. リポジトリの権限を確認
2. 画像サイズが大きすぎないか確認
3. GitHubのレート制限に引っかかっていないか確認

### パフォーマンス問題
1. 送信間隔を長くする（`send_interval_minutes`を増加）
2. 画像送信を無効にする（`include_images: false`）
3. システム最適化設定を確認

## セキュリティ注意事項

1. **Personal Access Token**: 絶対に他人に教えない
2. **設定ファイル**: `github_config.json`を`.gitignore`に追加
3. **リポジトリ**: 機密情報を含む場合はPrivateリポジトリを使用

## カスタマイズ

### 送信間隔の変更
```json
{
  "send_interval_minutes": 10  // 10分間隔に変更
}
```

### ラベルの追加
```json
{
  "issue_labels": ["inspection", "automated", "production", "line-a"]
}
```

### 画像送信の無効化
```json
{
  "include_images": false
}
```
