# GitHub統合ツール説明

## 📁 github_tools/ ディレクトリ構成

### ✅ 使用中ファイル

#### **github_sync.py** - 統合GitHub同期システム（メイン）
- **用途**: コード変更とトレーニングデータの自動同期
- **機能**:
  - コード変更の自動検出とコミット
  - トレーニングデータの自動アップロード
  - GitHub Issuesでの進捗レポート生成
- **使用方法**:
  ```bash
  cd github_tools
  python github_sync.py
  ```

#### **github_autocommit.py** - 自動コミットシステム
- **用途**: コード変更の自動検出とコミット
- **依存**: github_sync.pyから使用される
- **機能**:
  - ファイル変更の監視
  - 自動Gitコミット
  - GitHub APIでのプッシュ

#### **auto_sync.py** - 自動同期システム
- **用途**: コードとトレーニングデータの自動同期
- **依存**: github_sync.pyから使用される
- **機能**:
  - トレーニングデータの自動検出
  - クラス別・日付別整理
  - 統計レポート生成

#### **cursor_integration.py** - Cursor統合
- **用途**: Cursor IDEとの統合
- **機能**:
  - Cursorからの直接操作
  - 同期ステータス表示
  - 手動同期トリガー

#### **token_setup.py** - トークン設定
- **用途**: GitHub Personal Access Tokenの設定
- **使用方法**:
  ```bash
  cd github_tools
  python token_setup.py
  ```

### 📦 old/github_unused/ に移動されたファイル

以下のファイルは使用されていないため、`old/github_unused/`に移動されました：

#### **sync.py**
- **理由**: github_sync.pyと機能が重複（IntegratedAutoSyncは未使用）

#### **cursor_extension.py**
- **理由**: GUIツール（実際には使用されていない可能性）
- **注意**: 将来Cursor拡張機能として使う可能性がある場合は復元可能

#### **github_integration.py**
- **理由**: 古い統合システム（github_sync.pyに置き換え済み）

## 🔧 設定ファイル

設定ファイルは `config/` ディレクトリにあります：
- `github_config.json` - GitHub設定
- `cursor_github_config.json` - Cursor連携設定
- `auto_sync_config.json` - 自動同期設定

## 📖 使用方法

### 1. 初回セットアップ
```bash
# 依存関係インストール
pip install GitPython requests

# トークン設定
cd github_tools
python token_setup.py
```

### 2. 自動同期起動
```bash
cd github_tools
python github_sync.py
```

### 3. 一回だけ実行
```bash
cd github_tools
python github_sync.py once
```

## 🔄 ファイル依存関係

```
github_tools/
├── github_sync.py (メイン)
│   ├── github_autocommit.py
│   ├── auto_sync.py
│   └── training_data_manager.py
├── cursor_integration.py (独立)
└── token_setup.py (独立)
```

## ⚠️ 注意事項

- `training_data_manager.py`が参照されていますが、ファイルの場所を確認してください
- 設定ファイル（`*.json`）は`config/`ディレクトリに移動されています
- バッチファイル（`*.bat`）は`batch_files/`ディレクトリにあります

