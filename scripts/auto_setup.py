#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自動セットアップスクリプト
システム構成を自動検出して、必要なライブラリを自動インストール
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

# UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# 必要なパッケージのリスト
REQUIRED_PACKAGES = {
    # 基本パッケージ
    'numpy': 'numpy',
    'PIL': 'pillow',
    'cv2': 'opencv-python',
    'sklearn': 'scikit-learn',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'pandas': 'pandas',
    'PyQt5': 'PyQt5',
    # TensorFlow（環境によって異なる）
    'tensorflow': 'tensorflow',
    # その他
    'psutil': 'psutil',
    'scipy': 'scipy',
}

# WSL2環境用の追加パッケージ
WSL2_ADDITIONAL_PACKAGES = [
    'opencv-python-headless',  # WSL2ではheadless版も推奨
]

# Windows環境でGPU使用時に推奨されるパッケージ
WINDOWS_GPU_PACKAGES = [
    'tensorflow-directml-plugin',  # Windows GPUサポート（オプション）
]


def check_python_version():
    """Pythonバージョンをチェック"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7以上が必要です")
        print(f"   現在のバージョン: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_package(package_name, import_name=None):
    """パッケージがインストールされているかチェック"""
    if import_name is None:
        import_name = package_name
    
    try:
        # TensorFlowの場合は特別に処理（インポートに時間がかかる場合がある）
        if package_name.lower() == 'tensorflow' or import_name.lower() == 'tensorflow':
            try:
                import tensorflow as tf
                # インポート成功した場合、バージョンも確認
                _ = tf.__version__
                return True
            except (ImportError, AttributeError, Exception):
                # インポートエラーやその他のエラーはFalse
                return False
        
        __import__(import_name)
        return True
    except ImportError:
        return False
    except Exception:
        # その他のエラー（インポートは成功したが、何か問題がある場合）はTrue
        # 例: バージョン情報の取得エラーなど
        return True


def install_package(package_name):
    """パッケージをインストール"""
    try:
        print(f"  インストール中: {package_name}...")
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', package_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            encoding='utf-8',
            errors='replace'
        )
        print(f"  ✅ {package_name} インストール完了")
        return True
    except subprocess.CalledProcessError:
        print(f"  ❌ {package_name} インストール失敗")
        return False


def detect_system_environment():
    """システム環境を検出"""
    env_info = {
        'platform': platform.system(),
        'is_windows': sys.platform.startswith('win'),
        'is_linux': sys.platform.startswith('linux'),
        'is_wsl': False,
        'has_nvidia_gpu': False,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    
    # WSL2環境かチェック
    if env_info['is_linux']:
        try:
            with open('/proc/version', 'r') as f:
                version_info = f.read().lower()
                if 'microsoft' in version_info or 'wsl' in version_info:
                    env_info['is_wsl'] = True
        except:
            pass
    
    # NVIDIA GPUをチェック（Windows）
    if env_info['is_windows']:
        try:
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            if result.returncode == 0:
                env_info['has_nvidia_gpu'] = True
        except:
            pass
    
    # NVIDIA GPUをチェック（Linux/WSL2）
    if env_info['is_linux']:
        try:
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                env_info['has_nvidia_gpu'] = True
        except:
            pass
    
    return env_info


def install_missing_packages(env_info, recommended_packages=None, check_only=False):
    """不足しているパッケージをインストール"""
    print("=" * 60)
    print("パッケージ依存関係チェック")
    print("=" * 60)
    
    missing_packages = []
    installed_packages = []
    
    # 基本パッケージをチェック
    for import_name, package_name in REQUIRED_PACKAGES.items():
        if check_package(import_name, import_name):
            print(f"✅ {package_name} インストール済み")
            installed_packages.append(package_name)
        else:
            print(f"❌ {package_name} が見つかりません")
            missing_packages.append(package_name)
    
    # WSL2環境用の追加パッケージ
    if env_info.get('is_wsl', False):
        for package_name in WSL2_ADDITIONAL_PACKAGES:
            if not check_package(package_name.split('-')[0] if '-' in package_name else package_name):
                if package_name not in missing_packages:
                    missing_packages.append(package_name)
                    print(f"⚠️  {package_name} (WSL2推奨) が見つかりません")
    
    # 推奨パッケージをチェック
    if recommended_packages:
        for package_name in recommended_packages:
            # パッケージ名からインポート名を推測
            import_name = package_name.split('-')[0].replace('_', '')
            if not check_package(import_name, import_name):
                if package_name not in missing_packages:
                    missing_packages.append(package_name)
                    print(f"⚠️  {package_name} (推奨) が見つかりません")
    
    if check_only:
        return missing_packages, installed_packages
    
    # 不足しているパッケージをインストール
    if missing_packages:
        print(f"\n{len(missing_packages)}個のパッケージをインストールします...")
        print("-" * 60)
        
        failed_packages = []
        for package_name in missing_packages:
            if not install_package(package_name):
                failed_packages.append(package_name)
        
        print("-" * 60)
        if failed_packages:
            print(f"⚠️  インストール失敗: {', '.join(failed_packages)}")
            print("   手動でインストールしてください:")
            print(f"   pip install {' '.join(failed_packages)}")
        else:
            print("✅ すべてのパッケージがインストールされました")
        
        return failed_packages, installed_packages
    else:
        print("\n✅ すべての必要なパッケージがインストールされています")
        return [], installed_packages


def check_tensorflow_gpu_support(env_info):
    """TensorFlowのGPUサポートをチェック"""
    print("\n" + "=" * 60)
    print("TensorFlow GPUサポートチェック")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} インストール済み")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU検出: {len(gpus)}個")
            for gpu in gpus:
                print(f"   - {gpu}")
        else:
            if env_info.get('has_nvidia_gpu', False):
                print("⚠️  NVIDIA GPUが検出されていますが、TensorFlowがGPUを認識していません")
                if env_info.get('is_wsl', False):
                    print("   → WSL2環境: CUDAライブラリが不足している可能性があります")
                    print("   → 以下のコマンドでインストールしてください:")
                    print("      pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
                else:
                    print("   → Windows環境: CUDA対応TensorFlowが必要です")
                    print("   → WSL2環境でのGPU使用を推奨します")
            else:
                print("ℹ️  GPU未検出: CPUモードで動作します")
        
        build_info = tf.sysconfig.get_build_info()
        if build_info.get('is_cuda_build', False):
            print("✅ CUDA build: True")
            if 'cuda_version' in build_info:
                print(f"   CUDA version: {build_info['cuda_version']}")
            if 'cudnn_version' in build_info:
                print(f"   cuDNN version: {build_info['cudnn_version']}")
        else:
            print("ℹ️  CUDA build: False (CPU版)")
        
    except ImportError:
        print("❌ TensorFlowがインストールされていません")
        return False
    
    return True


def detect_cpu_info():
    """CPU情報を検出"""
    cpu_info = {
        'name': 'Unknown',
        'cores_logical': 0,
        'cores_physical': 0,
    }
    
    try:
        if platform.system() == 'Windows':
            # Windows: wmicを使用
            try:
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'name,NumberOfCores,NumberOfLogicalProcessors'],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        parts = lines[1].strip().split()
                        if len(parts) >= 3:
                            cpu_info['name'] = ' '.join(parts[:-2])
                            cpu_info['cores_physical'] = int(parts[-2])
                            cpu_info['cores_logical'] = int(parts[-1])
            except:
                pass
        else:
            # Linux: /proc/cpuinfoを使用
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            cpu_info['name'] = line.split(':', 1)[1].strip()
                        elif line.startswith('cpu cores'):
                            cpu_info['cores_physical'] = int(line.split(':')[1].strip())
                
                cpu_info['cores_logical'] = os.cpu_count() or 0
            except:
                cpu_info['cores_logical'] = os.cpu_count() or 0
        
        # psutilがあれば使用
        try:
            import psutil
            cpu_info['cores_logical'] = psutil.cpu_count(logical=True)
            cpu_info['cores_physical'] = psutil.cpu_count(logical=False)
        except:
            pass
    except Exception:
        pass
    
    return cpu_info


def detect_memory_info():
    """メモリ情報を検出"""
    memory_gb = 0
    
    try:
        if platform.system() == 'Windows':
            try:
                result = subprocess.run(
                    ['wmic', 'computersystem', 'get', 'TotalPhysicalMemory'],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip() and line.strip().isdigit():
                            memory_gb = int(line.strip()) / (1024**3)
                            break
            except:
                pass
        else:
            # Linux: /proc/meminfoを使用
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            memory_kb = int(line.split()[1])
                            memory_gb = memory_kb / (1024**2)
                            break
            except:
                pass
        
        # psutilがあれば使用
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
        except:
            pass
    except Exception:
        pass
    
    return memory_gb


def detect_gpu_info():
    """GPU情報を検出"""
    gpu_info = {
        'has_nvidia': False,
        'name': 'Unknown',
        'vram_gb': 0,
    }
    
    try:
        # nvidia-smiをチェック
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') and platform.system() == 'Windows' else 0
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info['has_nvidia'] = True
            parts = result.stdout.strip().split(',')
            if len(parts) >= 2:
                gpu_info['name'] = parts[0].strip()
                gpu_info['vram_gb'] = float(parts[1].strip()) / 1024
    except:
        pass
    
    return gpu_info


def recommend_packages(env_info, cpu_info, memory_info, gpu_info):
    """システム構成に応じて推奨パッケージを返す"""
    recommended = []
    
    # WSL2環境の場合
    if env_info.get('is_wsl', False):
        recommended.extend(WSL2_ADDITIONAL_PACKAGES)
        # CUDAライブラリ（WSL2でGPU使用時）
        if gpu_info.get('has_nvidia', False):
            recommended.extend([
                'nvidia-cublas-cu12',
                'nvidia-cudnn-cu12',
                'nvidia-cuda-nvrtc-cu12',
                'nvidia-cuda-runtime-cu12',
            ])
    
    # 高メモリシステムの場合（オプションパッケージ）
    if memory_info >= 32:
        # 大きなデータセットを扱う場合の追加パッケージ
        pass
    
    return recommended


def main():
    """メイン関数"""
    print("=" * 60)
    print("自動セットアップツール")
    print("=" * 60)
    print()
    
    # Pythonバージョンチェック
    if not check_python_version():
        return False
    
    print()
    
    # システム環境を検出
    print("=" * 60)
    print("システム環境検出")
    print("=" * 60)
    env_info = detect_system_environment()
    print(f"プラットフォーム: {env_info['platform']}")
    print(f"Pythonバージョン: {env_info['python_version']}")
    print(f"WSL2環境: {'はい' if env_info.get('is_wsl', False) else 'いいえ'}")
    print(f"NVIDIA GPU: {'検出' if env_info.get('has_nvidia_gpu', False) else '未検出'}")
    
    # CPU情報を検出
    print("\nCPU情報:")
    cpu_info = detect_cpu_info()
    print(f"  CPU: {cpu_info['name']}")
    print(f"  コア数（物理）: {cpu_info['cores_physical']}")
    print(f"  コア数（論理）: {cpu_info['cores_logical']}")
    
    # メモリ情報を検出
    print("\nメモリ情報:")
    memory_info = detect_memory_info()
    print(f"  メモリ: {memory_info:.1f} GB")
    
    # GPU情報を検出
    if env_info.get('has_nvidia_gpu', False):
        print("\nGPU情報:")
        gpu_info = detect_gpu_info()
        print(f"  GPU: {gpu_info['name']}")
        print(f"  VRAM: {gpu_info['vram_gb']:.1f} GB")
    else:
        gpu_info = {'has_nvidia': False}
    
    print()
    
    # pipをアップグレード
    print("=" * 60)
    print("pipをアップグレード中...")
    print("=" * 60)
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅ pipアップグレード完了")
    except:
        print("⚠️  pipアップグレードに失敗しました（続行します）")
    print()
    
    # システム構成に応じた推奨パッケージを取得
    recommended_packages = recommend_packages(env_info, cpu_info, memory_info, gpu_info)
    if recommended_packages:
        print("=" * 60)
        print("推奨パッケージ（システム構成に基づく）")
        print("=" * 60)
        for pkg in recommended_packages:
            print(f"  • {pkg}")
        print()
    
    # パッケージをチェック・インストール
    failed_packages, installed_packages = install_missing_packages(env_info, recommended_packages)
    
    # TensorFlow GPUサポートをチェック
    check_tensorflow_gpu_support(env_info)
    
    print()
    print("=" * 60)
    if failed_packages:
        print("⚠️  セットアップ完了（一部パッケージのインストールに失敗しました）")
        print("=" * 60)
        return False
    else:
        print("✅ セットアップ完了")
        print("=" * 60)
        
        # システム構成に応じた推奨設定を表示
        print("\n" + "=" * 60)
        print("推奨設定")
        print("=" * 60)
        if cpu_info['cores_logical'] >= 16 and memory_info >= 32:
            print("✅ 高性能システムが検出されました")
            print("   → 「超高性能」設定での学習を推奨します")
        elif cpu_info['cores_logical'] >= 8 and memory_info >= 16:
            print("✅ 中～高性能システムが検出されました")
            print("   → 「高性能」設定での学習を推奨します")
        else:
            print("ℹ️  標準システム")
            print("   → 「標準」設定での学習を推奨します")
        
        if env_info.get('is_wsl', False) and gpu_info.get('has_nvidia', False):
            print("✅ WSL2 GPUモードが利用可能です")
            print("   → 学習タブで「WSL2 GPUモード」を選択してください")
        
        print("=" * 60)
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

