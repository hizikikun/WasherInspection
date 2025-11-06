#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows環境でのTensorFlow GPUサポート設定スクリプト
CUDA 12.x+対応
"""

import sys
import os
import subprocess
import platform

# Windowsでのエンコーディング設定
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

def check_nvidia_gpu():
    """NVIDIA GPUが検出されているか確認"""
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        if result.returncode == 0:
            print("✅ NVIDIA GPU検出成功")
            print(result.stdout[:500])  # 最初の500文字を表示
            return True
        else:
            print("❌ nvidia-smiの実行に失敗しました")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smiが見つかりません。NVIDIAドライバーがインストールされているか確認してください。")
        return False
    except Exception as e:
        print(f"❌ GPU検出エラー: {e}")
        return False

def check_cuda_version():
    """CUDAバージョンを確認"""
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        if result.returncode == 0:
            # CUDAバージョンを抽出
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    print(f"✅ {line.strip()}")
                    return True
        return False
    except Exception as e:
        print(f"⚠️ CUDAバージョン確認エラー: {e}")
        return False

def install_tensorflow_gpu():
    """TensorFlow GPU対応版をインストール"""
    print("\n" + "="*60)
    print("TensorFlow GPU対応版のインストール")
    print("="*60)
    
    # pipをアップグレード
    print("\n[1] pipをアップグレード...")
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        print("  ✅ pipアップグレード完了")
    except Exception as e:
        print(f"  ⚠️ pipアップグレードエラー: {e}")
    
    # TensorFlow GPU対応版をインストール
    print("\n[2] TensorFlow GPU対応版をインストール...")
    print("  tensorflow[and-cuda]をインストール中...")
    
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'tensorflow[and-cuda]'],
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        print("  ✅ TensorFlow GPU対応版インストール完了")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ インストールエラー: {e}")
        print("\n  代替方法を試します...")
        
        # 代替方法: 個別にインストール
        try:
            print("  tensorflow本体をインストール...")
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '--upgrade', 'tensorflow'],
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            
            print("  CUDA関連ライブラリをインストール...")
            cuda_packages = [
                'nvidia-cublas-cu12',
                'nvidia-cudnn-cu12',
                'nvidia-cuda-nvrtc-cu12',
                'nvidia-cuda-runtime-cu12',
                'nvidia-cuda-cupti-cu12',
                'nvidia-cufft-cu12',
                'nvidia-curand-cu12',
                'nvidia-cusolver-cu12',
                'nvidia-cusparse-cu12',
                'nvidia-nvjitlink-cu12'
            ]
            
            for package in cuda_packages:
                try:
                    subprocess.check_call(
                        [sys.executable, '-m', 'pip', 'install', package],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                    )
                    print(f"    ✅ {package}")
                except:
                    print(f"    ⚠️ {package} のインストールに失敗（続行）")
            
            print("  ✅ 代替インストール完了")
            return True
        except Exception as e2:
            print(f"  ❌ 代替インストールも失敗: {e2}")
            return False
    except Exception as e:
        print(f"  ❌ 予期しないエラー: {e}")
        return False

def test_tensorflow_gpu():
    """TensorFlow GPUサポートをテスト"""
    print("\n" + "="*60)
    print("TensorFlow GPUサポートテスト")
    print("="*60)
    
    try:
        import tensorflow as tf
        print(f"\n✅ TensorFlow version: {tf.__version__}")
        
        # GPUデバイスを確認
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✅ GPU devices: {gpus}")
        
        if gpus:
            print("\n🎉 GPUサポートが有効です！")
            for gpu in gpus:
                print(f"  - {gpu}")
            
            # CUDAビルド情報を確認
            build_info = tf.sysconfig.get_build_info()
            is_cuda_build = build_info.get('is_cuda_build', False)
            print(f"\n✅ CUDA build: {is_cuda_build}")
            
            if 'cuda_version' in build_info:
                print(f"✅ CUDA version: {build_info['cuda_version']}")
            if 'cudnn_version' in build_info:
                print(f"✅ cuDNN version: {build_info['cudnn_version']}")
            
            return True
        else:
            print("\n⚠️ GPUが認識されていません")
            
            # ビルド情報を確認
            build_info = tf.sysconfig.get_build_info()
            is_cuda_build = build_info.get('is_cuda_build', False)
            print(f"CUDA build: {is_cuda_build}")
            
            if not is_cuda_build:
                print("\n⚠️ TensorFlowがCUDAビルドではありません")
                print("Windows + Python 3.12では、TensorFlowのCUDAサポートが制限されている可能性があります。")
                print("WSL2環境の使用を検討してください。")
            
            return False
    except ImportError:
        print("❌ TensorFlowがインポートできません")
        return False
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("Windows環境 TensorFlow GPUサポート設定")
    print("="*60)
    print()
    
    # システム情報
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print()
    
    # GPU検出
    print("[STEP 1] NVIDIA GPU検出")
    if not check_nvidia_gpu():
        print("\n❌ GPUが検出されませんでした。")
        print("NVIDIAドライバーがインストールされているか確認してください。")
        input("\n何かキーを押して終了...")
        return
    
    # CUDAバージョン確認
    print("\n[STEP 2] CUDAバージョン確認")
    check_cuda_version()
    
    # TensorFlow GPU版インストール
    print("\n[STEP 3] TensorFlow GPU対応版インストール")
    if not install_tensorflow_gpu():
        print("\n❌ TensorFlow GPU対応版のインストールに失敗しました。")
        input("\n何かキーを押して終了...")
        return
    
    # テスト
    print("\n[STEP 4] GPUサポートテスト")
    if test_tensorflow_gpu():
        print("\n" + "="*60)
        print("✅ セットアップ完了！GPUサポートが有効です。")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠️ GPUが認識されていません。")
        print("Windows + Python 3.12では、TensorFlowのCUDAサポートが制限されている可能性があります。")
        print("WSL2環境の使用を検討してください。")
        print("="*60)
    
    input("\n何かキーを押して終了...")

if __name__ == '__main__':
    main()






