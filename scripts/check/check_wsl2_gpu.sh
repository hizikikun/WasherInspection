#!/bin/bash
cd /mnt/c/Users/tomoh/WasherInspection
source venv_wsl2/bin/activate
python3 << 'EOF'
import tensorflow as tf
print("=" * 60)
print("WSL2環境のTensorFlow状態")
print("=" * 60)
print(f"TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU devices: {len(gpus)}")
for g in gpus:
    print(f"  - {g}")
build_info = tf.sysconfig.get_build_info()
print(f"CUDA build: {build_info.get('is_cuda_build', False)}")
if build_info.get('is_cuda_build', False):
    print("✅ WSL2環境でGPU版TensorFlowがインストールされています！")
else:
    print("❌ WSL2環境でもCPU版です")
EOF













