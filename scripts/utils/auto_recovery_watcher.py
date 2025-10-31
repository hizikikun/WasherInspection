#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-Recovery Watcher - 進捗停止を監視して自動再起動
"""

import json
import time
import subprocess
import sys
# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
import os
from pathlib import Path

STATUS_FILE = Path(__file__).parent.parent / 'logs' / 'training_status.json'
TRAINING_SCRIPT = Path(__file__).parent / 'train_4class_sparse_ensemble.py'
CHECK_INTERVAL = 60  # 60秒ごとにチェック
STUCK_THRESHOLD = 180  # 3分以上更新なしで停止と判定

def check_and_recover():
    """チェックして必要なら再起動"""
    if not STATUS_FILE.exists():
        return False
    
    try:
        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
            status = json.load(f)
    except Exception:
        return False
    
    timestamp = status.get('timestamp')
    if not timestamp:
        return False
    
    elapsed = time.time() - float(timestamp)
    progress = status.get('overall_progress_percent', 0)
    
    # 15%で止まっていて3分以上更新なし
    if progress <= 15.0 and elapsed > STUCK_THRESHOLD:
        print(f"[Auto-Recovery] Detected stuck at {progress}% for {int(elapsed)}s. Restarting...")
        restart_training()
        return True
    
    # 通常の更新停止チェック
    if elapsed > STUCK_THRESHOLD:
        print(f"[Auto-Recovery] No update for {int(elapsed)}s. Restarting...")
        restart_training()
        return True
    
    return False

def restart_training():
    """学習再起動"""
    python_exe = Path(sys.executable)
    script_path = TRAINING_SCRIPT.resolve()
    work_dir = script_path.parent.parent
    
    env = os.environ.copy()
    env['TF_USE_DIRECTML'] = '1'
    env['GPU_USE'] = '1'
    env['FORCE_ACCURACY'] = '1'
    
    if os.name == 'nt':
        subprocess.Popen(
            [str(python_exe), '-X', 'utf8', '-u', str(script_path)],
            cwd=str(work_dir),
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
    else:
        subprocess.Popen(
            [str(python_exe), '-X', 'utf8', '-u', str(script_path)],
            cwd=str(work_dir),
            env=env
        )

if __name__ == '__main__':
    import time
    while True:
        try:
            check_and_recover()
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(CHECK_INTERVAL)


