#!/usr/bin/env python3
"""
TensorFlow GPUサポート設定スクリプト
Windows環境でTensorFlowがGPUを認識できるようにするための設定を行います。
"""
import os
import sys
import subprocess
from pathlib import Path

def check_nvidia_gpu():
    """NVIDIA GPUが検出されるか確認"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            print(f"[OK] GPU検出: {result.stdout.strip()}")
            return True
    except Exception as e:
        print(f"[WARN] nvidia-smiエラー: {e}")
    return False

def install_cuda_packages():
    """必要なCUDAパッケージをインストール"""
    packages = [
        'nvidia-cudnn-cu12',
        'nvidia-cublas-cu12',
        'nvidia-cuda-nvrtc-cu12',
        'nvidia-cuda-runtime-cu12',
        'nvidia-cuda-cupti-cu12',
        'nvidia-cuda-nvcc-cu12',
        'nvidia-cufft-cu12',
        'nvidia-curand-cu12',
        'nvidia-cusolver-cu12',
        'nvidia-cusparse-cu12',
    ]
    
    print("CUDAパッケージをインストールしています...")
    for package in packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package, '--upgrade'], check=True)
            print(f"  [OK] {package}")
        except Exception as e:
            print(f"  [WARN] {package}: {e}")

def test_tensorflow_gpu():
    """TensorFlowがGPUを認識するかテスト"""
    print("\nTensorFlow GPU認識テスト...")
    try:
        import tensorflow as tf
        
        # ログレベルを設定
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        
        print(f"TensorFlow version: {tf.__version__}")
        
        # ビルド情報を確認
        build_info = tf.sysconfig.get_build_info()
        print(f"CUDA build: {build_info.get('is_cuda_build', False)}")
        
        # デバイス一覧を取得
        all_devices = tf.config.list_physical_devices()
        print(f"All devices: {all_devices}")
        
        gpu_devices = tf.config.list_physical_devices('GPU')
        dml_devices = [d for d in all_devices if 'DML' in d.name]
        
        if gpu_devices:
            print(f"[OK] GPU devices found: {gpu_devices}")
            return True
        elif dml_devices:
            print(f"[OK] DirectML devices found: {dml_devices}")
            return True
        else:
            print("[WARN] GPU devices not found")
            return False
            
    except Exception as e:
        print(f"[ERROR] エラー: {e}")
        return False

def main():
    print("=" * 60)
    print("TensorFlow GPUサポート設定")
    print("=" * 60)
    
    # GPU検出
    if not check_nvidia_gpu():
        print("\n[WARN] NVIDIA GPUが検出されませんでした。")
        print("nvidia-smiが正しく動作するか確認してください。")
        return
    
    # CUDAパッケージのインストール（必要に応じて）
    response = input("\nCUDAパッケージを再インストールしますか？ (y/n): ")
    if response.lower() == 'y':
        install_cuda_packages()
    
    # TensorFlow GPUテスト
    print("\n" + "=" * 60)
    success = test_tensorflow_gpu()
    
    print("\n" + "=" * 60)
    if success:
        print("[OK] TensorFlow GPUサポートが有効です！")
    else:
        print("[WARN] TensorFlow GPUサポートが有効になっていません。")
        print("\n推奨される解決策:")
        print("1. WSL2を使用してLinux環境でTensorFlowを実行")
        print("2. Python 3.11以下にダウングレードしてTensorFlow 2.10を使用")
        print("3. CPUモードで継続して使用（GPUは使えませんが動作します）")

if __name__ == '__main__':
    main()

