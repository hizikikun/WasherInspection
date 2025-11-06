# WSL2環境でのGPU学習使用方法

## ✅ GPU検出完了

WSL2環境でTensorFlow GPUサポートが正常に設定されました！

```
TensorFlow version: 2.20.0
GPU devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
CUDA build: True
CUDA version: 12.5.1
cuDNN version: 9
```

## 使用方法

### 方法1: PowerShellから実行（推奨）

```powershell
# Windows PowerShellから実行
.\run_wsl2_training.ps1
```

### 方法2: WSL2内で直接実行

```bash
# WSL2を起動
wsl

# プロジェクトディレクトリに移動
cd /mnt/c/Users/tomoh/WasherInspection

# 仮想環境をアクティベート
source venv_wsl2/bin/activate

# 学習スクリプトを実行
python3 scripts/train_4class_sparse_ensemble.py
```

### 方法3: ラッパースクリプトを使用

```bash
# WSL2内で実行
bash run_wsl2_training.sh
```

## 確認事項

### GPUが検出されているか確認

```bash
# WSL2内で実行
wsl bash -c "cd /mnt/c/Users/tomoh/WasherInspection && source venv_wsl2/bin/activate && python3 -c 'import tensorflow as tf; gpus = tf.config.list_physical_devices(\"GPU\"); print(f\"GPU devices: {len(gpus)}\"); [print(f\"  - {g}\") for g in gpus]'"
```

### nvidia-smiで確認

```bash
# WSL2内で実行
nvidia-smi
```

## 注意事項

1. **仮想環境**: WSL2用の仮想環境（`venv_wsl2`）が作成されています
2. **ファイルアクセス**: Windowsのファイルシステム（`C:\Users\tomoh\WasherInspection`）はWSL2から`/mnt/c/Users/tomoh/WasherInspection`としてアクセス可能です
3. **パフォーマンス**: WSL2でのファイルアクセスは若干遅くなる可能性がありますが、GPUの恩恵は大きいです

## トラブルシューティング

### GPUが検出されない場合

1. **WSL2でnvidia-smiを実行**:
   ```bash
   wsl nvidia-smi
   ```
   GPUが表示されれば、Windowsドライバーは正常に動作しています。

2. **TensorFlowを再インストール**:
   ```bash
   wsl bash setup_wsl2_tensorflow_gpu.sh
   ```

3. **仮想環境を再作成**:
   ```bash
   # WSL2内で実行
   cd /mnt/c/Users/tomoh/WasherInspection
   rm -rf venv_wsl2
   bash setup_wsl2_tensorflow_gpu.sh
   ```

### WindowsアプリとWSL2環境の違い

- **Windowsアプリ**: CPUモードで動作（完全に機能します）
- **WSL2環境**: GPUモードで動作（高速学習が可能）

両方の環境が利用可能です。用途に応じて使い分けてください。













