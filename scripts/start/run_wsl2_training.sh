#!/bin/bash
# WSL2環境で学習スクリプトを実行（GPU使用）

# ノートPC対応: スクリプトの場所からプロジェクトディレクトリを自動検出
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

echo "[INFO] プロジェクトディレクトリ: $PROJECT_DIR"

# 仮想環境をアクティベート
if [ -d "venv_wsl2" ]; then
    source venv_wsl2/bin/activate
else
    echo "[ERROR] venv_wsl2 が見つかりません。"
    echo "先に setup_wsl2_tensorflow_gpu.sh を実行してください。"
    exit 1
fi

# GPU検出テスト
echo "========================================="
echo "GPU検出確認"
echo "========================================="
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPU devices: {len(gpus)}'); [print(f'  - {g}') for g in gpus]"
echo ""

# 学習スクリプトを実行
echo "========================================="
echo "学習スクリプトを実行（GPU使用）"
echo "========================================="
python3 scripts/train_4class_sparse_ensemble.py "$@"








