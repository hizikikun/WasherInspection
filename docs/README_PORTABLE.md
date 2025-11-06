# ノートPC対応（ポータブル化）ガイド

このプロジェクトをノートPCやFドライブなど、任意の場所で動作させることができます。

## 📁 Fドライブへの移動方法

### 方法1: PowerShellスクリプトを使用（推奨）

```powershell
# プロジェクトルートで実行
.\move_to_f_drive.ps1
```

このスクリプトは：
- プロジェクト全体をFドライブに移動
- パス設定ファイル（`config/project_path.json`）を作成
- ハードコードされたパスを自動更新

### 方法2: 手動移動

1. プロジェクトフォルダ全体をFドライブにコピー
2. パス更新スクリプトを実行：
   ```powershell
   cd F:\WasherInspection
   .\update_paths.ps1
   ```

## 🔧 パスの自動検出機能

プロジェクトは以下の方法でパスを自動検出します：

1. **設定ファイル優先**: `config/project_path.json`が存在する場合、そこから読み込み
2. **スクリプト位置**: 実行中のスクリプトの場所からプロジェクトルートを推定
3. **フォールバック**: 現在の作業ディレクトリを使用

## 📝 設定ファイル

`config/project_path.json`が自動生成されます：

```json
{
  "project_root": "F:\\WasherInspection",
  "project_root_wsl": "/mnt/f/WasherInspection",
  "original_path": "C:\\Users\\tomoh\\WasherInspection",
  "moved_date": "2025-01-XX XX:XX:XX"
}
```

## ⚙️ WSL2環境での使用

WSL2環境を使用する場合：

1. プロジェクトをFドライブに移動後、WSL2環境で仮想環境を再作成：
   ```bash
   cd /mnt/f/WasherInspection
   bash setup_wsl2_tensorflow_gpu.sh
   ```

2. または、既存の仮想環境がある場合は、パスを更新：
   ```bash
   # venv_wsl2内のパスを確認（自動的に更新される場合がある）
   cd /mnt/f/WasherInspection
   source venv_wsl2/bin/activate
   ```

## 🚀 使用方法

### 移動前

```powershell
cd C:\Users\tomoh\WasherInspection
python dashboard\integrated_washer_app.py
```

### 移動後

```powershell
cd F:\WasherInspection
python dashboard\integrated_washer_app.py
```

パスは自動的に検出されるため、特別な設定は不要です。

## 📋 注意事項

1. **仮想環境（venv_wsl2）**: WSL2仮想環境にはハードコードされたパスが含まれている可能性があります。移動後は再作成を推奨します。

2. **チェックポイント**: 学習済みのチェックポイントは`checkpoints/`フォルダに保存されます。これらも一緒に移動されます。

3. **モデルファイル**: 保存済みのモデルファイル（`.h5`）も一緒に移動されます。

## 🔍 トラブルシューティング

### パスが見つからないエラー

```powershell
# パスを手動で更新
.\update_paths.ps1 -ProjectPath "F:\WasherInspection"
```

### WSL2環境でパスが見つからない

```bash
# WSL2環境でプロジェクトディレクトリを確認
cd /mnt/f/WasherInspection
pwd
```

## ✅ 確認事項

移動後、以下を確認してください：

- [ ] アプリが正常に起動する
- [ ] WSL2環境で学習が実行できる（WSL2使用時）
- [ ] チェックポイントが読み込める
- [ ] モデルファイルが読み込める





