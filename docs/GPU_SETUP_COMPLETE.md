# ✅ GPUセットアップ完了！

## 🎉 成功！

WSL2環境でTensorFlow GPUサポートが正常に設定されました！

### 検出結果

```
✅ TensorFlow version: 2.20.0
✅ GPU devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
✅ CUDA build: True
✅ CUDA version: 12.5.1
✅ cuDNN version: 9
```

## 📋 使用方法

### 1. WSL2内で直接実行（推奨・最も高速）

```bash
# WSL2を起動
wsl

# プロジェクトディレクトリに移動
cd /mnt/c/Users/tomoh/WasherInspection

# 仮想環境をアクティベート
source venv_wsl2/bin/activate

# 学習スクリプトを実行（GPU使用）
python3 scripts/train_4class_sparse_ensemble.py
```

### 2. PowerShellから実行

```powershell
# Windows PowerShellから実行
.\run_wsl2_training.ps1
```

### 3. WSL2ラッパースクリプトを使用

```bash
# WSL2内で実行
bash run_wsl2_training.sh
```

## 🚀 パフォーマンス

**GPUモード（WSL2）:**
- ✅ RTX 4070をフル活用
- ✅ 学習速度: CPUの10-50倍高速
- ✅ バッチサイズ: 最大96まで可能（システムスペック次第）

**CPUモード（Windowsアプリ）:**
- ✅ 完全に機能する
- ✅ すべての機能が利用可能
- ✅ 学習速度: GPUより遅いが、精度は同じ

## 📝 注意事項

1. **2つの環境が利用可能:**
   - Windowsアプリ: CPUモード（外観検査などに最適）
   - WSL2環境: GPUモード（高速学習に最適）

2. **ファイルアクセス:**
   - WSL2からWindowsファイルシステム（`C:\Users\tomoh\WasherInspection`）は`/mnt/c/Users/tomoh/WasherInspection`としてアクセス可能

3. **仮想環境:**
   - WSL2用: `venv_wsl2`（GPU対応）
   - Windows用: システムPython環境（CPUモード）

## 🔍 確認方法

### GPU検出を確認

```bash
# WSL2内で実行
wsl bash -c "cd /mnt/c/Users/tomoh/WasherInspection && source venv_wsl2/bin/activate && python3 -c 'import tensorflow as tf; gpus = tf.config.list_physical_devices(\"GPU\"); print(f\"GPU: {len(gpus)}\")'"
```

### nvidia-smiで確認

```bash
wsl nvidia-smi
```

## 🎯 次のステップ

1. **すぐに学習を開始:**
   ```bash
   wsl
   cd /mnt/c/Users/tomoh/WasherInspection
   source venv_wsl2/bin/activate
   python3 scripts/train_4class_sparse_ensemble.py
   ```

2. **Windowsアプリも継続使用:**
   - 外観検査などはWindowsアプリ（CPUモード）で実行可能
   - GPU学習はWSL2環境で実行

## 💡 ヒント

- **高速学習が必要な場合**: WSL2環境を使用
- **リアルタイム検査**: Windowsアプリを使用（CPUモードでも十分高速）
- **両方の環境を同時に使用可能**: 用途に応じて使い分け

**GPUサポートが完全に有効になりました！高速学習が可能です！** 🚀













