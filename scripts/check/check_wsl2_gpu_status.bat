@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"

echo ============================================================
echo WSL2環境 GPU状態確認
echo ============================================================
echo.

echo [1] WSL2環境の確認...
wsl --list --quiet >nul 2>&1
if errorlevel 1 (
    echo   ❌ WSL2がインストールされていません
    echo   WSL2をインストールしてください: wsl --install
    pause
    exit /b 1
)
echo   ✅ WSL2が利用可能です

echo.
echo [2] WSL2環境でTensorFlow GPUを確認...
wsl bash -c "cd /mnt/c/Users/tomoh/WasherInspection && [ -d 'venv_wsl2' ] && source venv_wsl2/bin/activate && python3 -c 'import tensorflow as tf; gpus = tf.config.list_physical_devices(\"GPU\"); print(f\"GPU devices: {len(gpus)}\"); [print(f\"  - {g}\") for g in gpus]; build_info = tf.sysconfig.get_build_info(); print(f\"CUDA build: {build_info.get(\"is_cuda_build\", False)}\"); print(f\"CUDA version: {build_info.get(\"cuda_version\", \"N/A\")}\")' 2>&1" 2>nul

if errorlevel 1 (
    echo   ⚠️ WSL2環境でTensorFlow GPUが確認できませんでした
    echo   セットアップが必要です: setup_wsl2_tensorflow_gpu.sh
) else (
    echo   ✅ WSL2環境でGPUが利用可能です
)

echo.
echo ============================================================
pause






