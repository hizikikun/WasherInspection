#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU状態確認スクリプト"""

print("=" * 60)
print("Windows環境のTensorFlow状態")
print("=" * 60)
try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
    build_info = tf.sysconfig.get_build_info()
    print(f"CUDA build: {build_info.get('is_cuda_build', False)}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU devices: {len(gpus)}")
    if len(gpus) == 0:
        print("→ Windows環境ではCPU版TensorFlowがインストールされています（正常）")
except Exception as e:
    print(f"エラー: {e}")

print()
print("=" * 60)
print("WSL2環境のTensorFlow状態（確認中...）")
print("=" * 60)













