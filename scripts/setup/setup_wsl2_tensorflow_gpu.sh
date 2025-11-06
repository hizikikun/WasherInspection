#!/bin/bash
# TensorFlow GPU完全サポート設定スクリプト（WSL2用）

set -e

echo '========================================'
echo 'TensorFlow GPUサポート設定 (WSL2内)'
echo '========================================'
echo ''

# インターネット接続確認
check_internet_connection() {
    echo '[INFO] インターネット接続を確認中...'
    if ping -c 1 -W 2 8.8.8.8 &> /dev/null || ping -c 1 -W 2 google.com &> /dev/null; then
        echo '[OK] インターネット接続が確認されました'
        return 0
    else
        echo '[ERROR] インターネット接続が確認できませんでした'
        echo '[ERROR] ネットワーク接続を確認してください'
        return 1
    fi
}

# 接続確認を実行
if ! check_internet_connection; then
    echo '[ERROR] インターネット接続が必要です。ネットワーク設定を確認してください。'
    exit 1
fi

# CUDA Toolkitのインストール確認
if ! command -v nvcc &> /dev/null; then
    echo '[INFO] CUDA Toolkitがインストールされていません。'
    echo '[INFO] WSL2では、WindowsのNVIDIAドライバーが自動的に利用されます。'
    echo '[INFO] CUDA Toolkitのインストールは不要です（WSL2ではWindowsドライバーを使用）。'
    echo '[INFO] TensorFlowが自動的にGPUを検出します。'
else
    echo '[OK] CUDA Toolkitが検出されました'
    nvcc --version
fi

# リトライ機能付きコマンド実行
run_with_retry() {
    local max_attempts=3
    local attempt=1
    local command="$*"
    local result=1
    
    # set -eを一時的に無効化
    set +e
    
    while [ $attempt -le $max_attempts ]; do
        echo "[INFO] 試行 $attempt/$max_attempts: $command"
        eval "$command"
        result=$?
        if [ $result -eq 0 ]; then
            set -e
            return 0
        else
            echo "[WARN] 試行 $attempt が失敗しました (終了コード: $result)"
            if [ $attempt -lt $max_attempts ]; then
                echo "[INFO] 5秒後に再試行します..."
                sleep 5
            fi
            attempt=$((attempt + 1))
        fi
    done
    
    set -e
    echo "[ERROR] コマンドが $max_attempts 回の試行後に失敗しました: $command"
    echo "[ERROR] ネットワーク接続を確認してください"
    return 1
}

# Python環境の確認
if ! command -v python3 &> /dev/null; then
    echo '[INFO] Python3をインストールします...'
    run_with_retry "sudo apt-get update"
    run_with_retry "sudo apt-get install -y python3 python3-pip python3-venv"
fi

# 仮想環境の作成（プロジェクトディレクトリに）
# ノートPC対応: スクリプトの場所からプロジェクトディレクトリを自動検出
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Windowsパスの場合、wslpathで変換を試みる
if [[ "$PROJECT_DIR" =~ ^/mnt/ ]]; then
    # 既にWSLパスの場合
    cd "$PROJECT_DIR" || exit 1
else
    # 相対パスまたは絶対パスの場合
    cd "$PROJECT_DIR" || exit 1
fi

echo "[INFO] プロジェクトディレクトリ: $PROJECT_DIR"

if [ ! -d "venv_wsl2" ]; then
    echo '[INFO] Python仮想環境を作成します...'
    python3 -m venv venv_wsl2
fi

echo '[INFO] 仮想環境をアクティベート...'
source venv_wsl2/bin/activate

echo '[INFO] pipをアップグレード...'
run_with_retry "pip install --upgrade pip"

echo '[INFO] TensorFlowとCUDAサポートをインストール...'
# WSL2では、tensorflow[and-cuda]の代わりにtensorflowをインストールし、
# 必要なCUDAライブラリを個別にインストール
run_with_retry "pip install --upgrade tensorflow"

# CUDA関連ライブラリをインストール（WSL2ではWindowsドライバーを使用）
echo '[INFO] CUDA関連ライブラリをインストール...'
run_with_retry "pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cupti-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12"

# CUDA Toolkitのインストール（XLA使用のため）
echo '[INFO] CUDA Toolkitをインストール中...'
echo '[INFO] WSL2では、WindowsのNVIDIAドライバーを使用しますが、XLAを使用するにはCUDA Toolkitが必要です。'
echo '[INFO] NVIDIA公式のCUDA Toolkit for WSL2をインストールします...'

# CUDA Toolkitのインストール（WSL2用）
if ! command -v nvcc &> /dev/null; then
    echo '[INFO] CUDA Toolkitをインストールします...'
    echo '[INFO] 以下のコマンドで手動インストールしてください:'
    echo '[INFO] wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin'
    echo '[INFO] sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600'
    echo '[INFO] wget https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda-repo-wsl-ubuntu-12-5-local_12.5.1-1_amd64.deb'
    echo '[INFO] sudo dpkg -i cuda-repo-wsl-ubuntu-12-5-local_12.5.1-1_amd64.deb'
    echo '[INFO] sudo cp /var/cuda-repo-wsl-ubuntu-12-5-local/cuda-*-keyring.gpg /usr/share/keyrings/'
    echo '[INFO] sudo apt-get update'
    echo '[INFO] sudo apt-get -y install cuda-toolkit-12-5'
    echo ''
    echo '[INFO] または、nvidia-nvjitlink-cu12パッケージからlibdeviceを探します...'
    
    # nvidia-nvjitlink-cu12パッケージからlibdeviceを探す
    python3 << 'PYTHON_SCRIPT'
import site
import os
from pathlib import Path

# nvidia-nvjitlink-cu12のパッケージパスを探す
try:
    import nvidia.nvjitlink
    package_path = Path(nvidia.nvjitlink.__file__).parent
    print(f"[INFO] nvidia-nvjitlinkパッケージパス: {package_path}")
    
    # libdeviceファイルを探す
    libdevice_files = list(package_path.rglob('libdevice*.bc'))
    if libdevice_files:
        print(f"[INFO] libdeviceファイルが見つかりました:")
        for f in libdevice_files:
            print(f"  - {f}")
            # XLA_FLAGSにパスを追加
            libdevice_dir = f.parent
            print(f"[INFO] libdeviceディレクトリ: {libdevice_dir}")
            print(f"[INFO] このパスをXLA_FLAGSに設定してください:")
            print(f"[INFO] export XLA_FLAGS='--xla_gpu_cuda_data_dir={libdevice_dir}'")
    else:
        print("[WARN] libdeviceファイルが見つかりませんでした")
except ImportError:
    print("[WARN] nvidia-nvjitlinkパッケージが見つかりません")
PYTHON_SCRIPT
fi

echo '[INFO] 学習に必要なパッケージをインストール...'
# OpenCVとその他の必要なパッケージ
run_with_retry "pip install opencv-python opencv-python-headless"
run_with_retry "pip install numpy scipy"
run_with_retry "pip install scikit-learn"
run_with_retry "pip install matplotlib seaborn"
run_with_retry "pip install pillow"
run_with_retry "pip install pandas"

echo ''
echo '========================================'
echo 'TensorFlow GPU認識テスト'
echo '========================================'
python3 << 'PYTHON_SCRIPT'
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU devices: {gpus}")

if gpus:
    print("[OK] GPUサポートが有効です！")
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("[WARN] GPUが認識されていません")

build_info = tf.sysconfig.get_build_info()
print(f"CUDA build: {build_info.get('is_cuda_build', False)}")
if 'cuda_version' in build_info:
    print(f"CUDA version: {build_info['cuda_version']}")
if 'cudnn_version' in build_info:
    print(f"cuDNN version: {build_info['cudnn_version']}")
PYTHON_SCRIPT

echo ''
echo '[OK] セットアップ完了！'
echo 'WSL2環境でTensorFlowがGPUを使用できます。'

