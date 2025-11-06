#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Health Checker and Auto-Recovery
進捗停止を自動検出して学習を再起動する
"""

import json
import time
import os
import subprocess
from pathlib import Path
import sys
# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

STATUS_FILE = Path(__file__).parent.parent / 'logs' / 'training_status.json'
TRAINING_SCRIPT = Path(__file__).parent / 'train_4class_sparse_ensemble.py'
STUCK_THRESHOLD = 180  # 3分以上更新がなければ停止とみなす
STUCK_AT_15_PERCENT = True  # 15%で止まっているかチェック

def check_training_health():
    """学習の健康状態をチェック"""
    if not STATUS_FILE.exists():
        return False, "Status file not found"
    
    try:
        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
            status = json.load(f)
    except Exception as e:
        return False, f"Failed to load status: {e}"
    
    # タイムスタンプチェック
    timestamp = status.get('timestamp')
    if not timestamp:
        return False, "No timestamp in status"
    
    elapsed = time.time() - float(timestamp)
    
    # 進捗が15%で止まっているかチェック
    overall_progress = status.get('overall_progress_percent', 0)
    stage = status.get('stage', '')
    
    # 15%で止まっていて、タイムスタンプが古い場合
    if overall_progress <= 15.0 and elapsed > STUCK_THRESHOLD:
        return False, f"Stuck at {overall_progress}% for {int(elapsed)} seconds"
    
    # 通常の更新停止チェック
    if elapsed > STUCK_THRESHOLD:
        return False, f"No update for {int(elapsed)} seconds"
    
    return True, "OK"

def restart_training():
    """学習を再起動"""
    python_exe = Path(sys.executable)
    script_path = TRAINING_SCRIPT.resolve()
    
    env = os.environ.copy()
    env['TF_USE_DIRECTML'] = '1'
    env['GPU_USE'] = '1'
    env['FORCE_ACCURACY'] = '1'
    
    # Windows用のコマンド
    if os.name == 'nt':
        cmd = [
            'cmd', '/c',
            'cd', '/d', str(script_path.parent.parent),
            '&&',
            'set', 'TF_USE_DIRECTML=1',
            '&&',
            'set', 'GPU_USE=1',
            '&&',
            'set', 'FORCE_ACCURACY=1',
            '&&',
            'start', '/B',
            str(python_exe),
            '-X', 'utf8',
            '-u',
            str(script_path)
        ]
        # subprocess.Popenでバックグラウンド実行
        subprocess.Popen(
            [str(python_exe), '-X', 'utf8', '-u', str(script_path)],
            cwd=str(script_path.parent.parent),
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
    else:
        subprocess.Popen(
            [str(python_exe), '-X', 'utf8', '-u', str(script_path)],
            cwd=str(script_path.parent.parent),
            env=env
        )
    
    return True

def main():
    """メイン関数"""
    is_healthy, message = check_training_health()
    
    if not is_healthy:
        print(f"Training health check failed: {message}")
        print("Restarting training...")
        restart_training()
        print("Training restarted!")
    else:
        print(f"Training is healthy: {message}")

if __name__ == '__main__':
    main()


