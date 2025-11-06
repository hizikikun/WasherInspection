#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPUサポートテストスクリプト"""

import os
import sys

# Windows環境でのGPUサポート設定
if os.name == 'nt':
    # DirectMLを優先的に有効化
    if 'TF_USE_DIRECTML' not in os.environ:
        os.environ['TF_USE_DIRECTML'] = '1'
    
    # CUDA環境変数を設定
    cuda_path = os.environ.get('CUDA_PATH', 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6')
    if os.path.exists(cuda_path):
        os.environ['CUDA_PATH'] = cuda_path
        cuda_bin = os.path.join(cuda_path, 'bin')
        if cuda_bin not in os.environ.get('PATH', ''):
            os.environ['PATH'] = cuda_bin + os.pathsep + os.environ.get('PATH', '')
        
        # DLLパスを追加
        try:
            import ctypes
            try:
                ctypes.windll.kernel32.SetDllDirectoryW(cuda_path + '\\bin')
            except:
                pass
            os.add_dll_directory(os.path.join(cuda_path, 'bin'))
            os.add_dll_directory(os.path.join(cuda_path, 'lib', 'x64'))
        except Exception:
            pass

print("=" * 60)
print("TensorFlow GPU サポートテスト")
print("=" * 60)
print()

try:
    import tensorflow as tf
    print(f"[OK] TensorFlow version: {tf.__version__}")
    
    # ビルド情報
    build_info = tf.sysconfig.get_build_info()
    print(f"\nビルド情報:")
    print(f"  - CUDA build: {build_info.get('is_cuda_build', False)}")
    print(f"  - ROCm build: {build_info.get('is_rocm_build', False)}")
    print(f"  - TensorRT build: {build_info.get('is_tensorrt_build', False)}")
    
    # デバイス検出
    print(f"\nデバイス検出:")
    dml_devices = tf.config.list_physical_devices('DML')
    print(f"  - DirectML devices: {len(dml_devices)} ({dml_devices})")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"  - CUDA GPU devices: {len(gpus)} ({gpus})")
    
    all_devices = tf.config.list_physical_devices()
    print(f"  - All devices: {len(all_devices)}")
    for d in all_devices:
        print(f"    * {d.name} ({d.device_type})")
    
    # 推奨事項
    print(f"\n推奨事項:")
    if dml_devices:
        print("  [OK] DirectML GPUが検出されました！Windows GPUが使用可能です。")
    elif gpus:
        print("  [OK] CUDA GPUが検出されました！GPUが使用可能です。")
    else:
        print("  [WARN] GPUが検出されませんでした。CPUモードで動作します。")
        print("  完全なGPUサポートを得るには、WSL2を使用することを推奨します。")
    
    # 簡単な動作テスト
    print(f"\n動作テスト:")
    try:
        with tf.device('/CPU:0'):
            test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            result = tf.matmul(test_tensor, test_tensor)
            print(f"  [OK] TensorFlowは正常に動作しています")
            print(f"    テスト結果: {result.numpy()}")
    except Exception as e:
        print(f"  [ERROR] エラー: {e}")
    
except ImportError as e:
    print(f"✗ TensorFlowのインポートエラー: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("テスト完了")
print("=" * 60)

