#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSL2環境のGPU検出をテストするスクリプト
アプリがWSL2環境を正しく検出できるか確認
"""

import subprocess
import sys
import os

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

def test_wsl2_gpu():
    """WSL2環境でGPUが検出できるかテスト"""
    print("="*60)
    print("WSL2環境 GPU検出テスト")
    print("="*60)
    print()
    
    # テスト1: WSL2が利用可能か
    print("[1] WSL2が利用可能か確認...")
    try:
        result = subprocess.run(
            ['wsl', '--list', '--quiet'],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        if result.returncode == 0:
            print("  [OK] WSL2が利用可能です")
        else:
            print("  [ERROR] WSL2が利用できません")
            return False
    except Exception as e:
        print(f"  [ERROR] エラー: {e}")
        return False
    
    # テスト2: venv_wsl2ディレクトリが存在するか
    print("\n[2] venv_wsl2ディレクトリが存在するか確認...")
    try:
        result = subprocess.run(
            ['wsl', 'bash', '-c', 'test -d /mnt/c/Users/tomoh/WasherInspection/venv_wsl2 && echo "OK"'],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        if result.returncode == 0 and 'OK' in result.stdout:
            print("  [OK] venv_wsl2ディレクトリが存在します")
        else:
            print("  [ERROR] venv_wsl2ディレクトリが見つかりません")
            print("  -> setup_wsl2_tensorflow_gpu.sh を実行してください")
            return False
    except Exception as e:
        print(f"  [ERROR] エラー: {e}")
        return False
    
    # テスト3: TensorFlow GPUが検出できるか
    print("\n[3] TensorFlow GPUが検出できるか確認...")
    try:
        # シンプルなコマンドでテスト
        cmd = 'cd /mnt/c/Users/tomoh/WasherInspection && source venv_wsl2/bin/activate && python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices(\\\"GPU\\\"); print(\\\"OK\\\" if gpus else \\\"NO\\\")"'
        result = subprocess.run(
            ['wsl', 'bash', '-c', cmd],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        
        print(f"  戻り値: {result.returncode}")
        print(f"  標準出力: {result.stdout[:200]}")
        if result.stderr:
            print(f"  エラー出力: {result.stderr[:200]}")
        
        if result.returncode == 0 and 'OK' in result.stdout:
            print("  [OK] TensorFlow GPUが検出できました！")
            
            # GPU数を取得
            try:
                gpu_cmd = 'cd /mnt/c/Users/tomoh/WasherInspection && source venv_wsl2/bin/activate && python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices(\\\"GPU\\\"); print(len(gpus))"'
                gpu_result = subprocess.run(
                    ['wsl', 'bash', '-c', gpu_cmd],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )
                if gpu_result.returncode == 0:
                    gpu_count = gpu_result.stdout.strip()
                    if gpu_count.isdigit():
                        print(f"  [OK] GPU数: {gpu_count}個")
            except:
                pass
            
            return True
        else:
            print("  [ERROR] TensorFlow GPUが検出できませんでした")
            return False
    except Exception as e:
        print(f"  [ERROR] エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_wsl2_gpu()
    print("\n" + "="*60)
    if success:
        print("[OK] WSL2環境でGPUが利用可能です！")
        print("アプリを再起動すると、WSL2環境が検出されるはずです。")
    else:
        print("[WARN] WSL2環境でGPUが検出できませんでした。")
        print("setup_wsl2_tensorflow_gpu.sh を実行してセットアップしてください。")
    print("="*60)
    input("\n何かキーを押して終了...")

