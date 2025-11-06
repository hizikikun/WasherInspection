#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""アプリ起動監視スクリプト"""

import sys
import os
import time
import subprocess
import psutil
from pathlib import Path
from datetime import datetime

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

def check_app_running():
    """アプリが起動しているかチェック"""
    app_path = Path('dashboard/integrated_washer_app.py')
    if not app_path.exists():
        return False, []
    
    running_processes = []
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline:
                    cmdline_str = ' '.join(str(arg) for arg in cmdline)
                    if 'integrated_washer_app.py' in cmdline_str:
                        proc_info = proc.info.copy()
                        proc_info['memory_mb'] = proc_info['memory_info'].rss / (1024 * 1024) if proc_info.get('memory_info') else 0
                        running_processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        # psutilが使えない場合は代替方法
        try:
            import subprocess
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            if 'integrated_washer_app' in result.stdout:
                return True, [{'pid': 'unknown', 'note': 'プロセス検出中'}]
        except:
            pass
        return False, []
    
    return len(running_processes) > 0, running_processes

def check_error_logs():
    """エラーログをチェック"""
    errors = []
    
    # app_error_log.txtをチェック
    error_log_path = Path('app_error_log.txt')
    if error_log_path.exists():
        try:
            with open(error_log_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 最新のエラー（最後のセクション）を取得
                if 'エラー発生時刻' in content or '起動エラー' in content or 'UIセットアップエラー' in content:
                    lines = content.split('\n')
                    # 最後のエラーセクションを抽出
                    error_section = []
                    in_error = False
                    for line in reversed(lines):
                        if '=' * 60 in line:
                            if in_error:
                                break
                            in_error = True
                        if in_error:
                            error_section.insert(0, line)
                    
                    if error_section:
                        errors.append({
                            'file': 'app_error_log.txt',
                            'content': '\n'.join(error_section[-20:])  # 最後の20行
                        })
        except Exception as e:
            errors.append({
                'file': 'app_error_log.txt',
                'content': f'ログファイル読み込みエラー: {e}'
            })
    
    # app_startup.logをチェック
    startup_log_path = Path('app_startup.log')
    if startup_log_path.exists():
        try:
            with open(startup_log_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                # 最後の50行をチェック
                recent_lines = lines[-50:] if len(lines) > 50 else lines
                # エラーパターンを検索
                error_lines = [line for line in recent_lines if any(keyword in line.lower() for keyword in ['error', 'エラー', 'exception', 'traceback', 'failed', '失敗'])]
                if error_lines:
                    errors.append({
                        'file': 'app_startup.log',
                        'content': '\n'.join(error_lines[-10:])  # 最後の10行
                    })
        except Exception as e:
            pass
    
    return errors

def monitor_app():
    """アプリを監視"""
    log_file = Path('app_monitor.log')
    
    def log_message(msg, to_file=True):
        """メッセージをコンソールとファイルに出力"""
        print(msg)
        if to_file:
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{msg}\n")
            except:
                pass
    
    log_message("=" * 60, False)
    log_message("アプリ起動監視を開始します", False)
    log_message("=" * 60, False)
    log_message(f"監視開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", False)
    log_message("Ctrl+C で監視を終了します", False)
    log_message("=" * 60, False)
    log_message("", False)
    
    check_count = 0
    last_status = None
    
    try:
        while True:
            check_count += 1
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # アプリの実行状態をチェック
            is_running, processes = check_app_running()
            
            # ステータスが変わった場合または10回ごとに詳細表示
            if is_running != last_status or check_count % 10 == 0:
                if is_running:
                    msg = f"[{current_time}] ✓ アプリが起動しています (プロセス数: {len(processes)})"
                    log_message(msg)
                    for proc in processes:
                        pid = proc.get('pid', 'unknown')
                        create_time = proc.get('create_time', 0)
                        memory_mb = proc.get('memory_mb', 0)
                        if create_time:
                            start_time = datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S')
                            log_message(f"  - PID: {pid}, 開始時刻: {start_time}, メモリ: {memory_mb:.1f}MB")
                        else:
                            log_message(f"  - PID: {pid}")
                else:
                    log_message(f"[{current_time}] ✗ アプリが停止しています")
                last_status = is_running
            
            # エラーログをチェック
            errors = check_error_logs()
            if errors:
                log_message(f"\n[{current_time}] ⚠ エラーログを検出しました:")
                for error in errors:
                    log_message(f"  ファイル: {error['file']}")
                    log_message(f"  内容:")
                    for line in error['content'].split('\n')[-5:]:  # 最後の5行
                        if line.strip():
                            log_message(f"    {line}")
                log_message("")
            
            # 5秒待機
            time.sleep(5)
            
            # 10回ごとに簡潔なステータス表示
            if check_count % 10 == 0:
                status_symbol = "✓" if is_running else "✗"
                log_message(f"[{current_time}] {status_symbol} 監視中... (チェック回数: {check_count})")
    
    except KeyboardInterrupt:
        log_message("\n" + "=" * 60, False)
        log_message("監視を終了します", False)
        log_message("=" * 60, False)

if __name__ == '__main__':
    monitor_app()

