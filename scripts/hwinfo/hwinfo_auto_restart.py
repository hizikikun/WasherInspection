#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HWiNFO自動再起動スクリプト
12時間ごとにHWiNFOを再起動してShared Memory制限を回避
"""

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
import time
import os
from pathlib import Path

def find_hwinfo():
    """HWiNFO64.exeのパスを見つける"""
    possible_paths = [
        r"C:\Program Files\HWiNFO64\HWiNFO64.EXE",
        r"C:\Program Files (x86)\HWiNFO64\HWiNFO64.EXE",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # 環境変数から検索を試みる
    try:
        result = subprocess.run(
            ['where', 'HWiNFO64.exe'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    
    return None

def kill_hwinfo():
    """HWiNFO64のプロセスを終了"""
    try:
        # tasklistでプロセスを確認
        result = subprocess.run(
            ['tasklist', '/fi', 'IMAGENAME eq HWiNFO64.EXE'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if 'HWiNFO64.EXE' in result.stdout:
            # プロセスを終了
            subprocess.run(
                ['taskkill', '/f', '/im', 'HWiNFO64.EXE'],
                capture_output=True,
                timeout=10
            )
            print("[INFO] HWiNFO64を終了しました")
            time.sleep(2)  # プロセス終了を待機
            return True
        else:
            print("[INFO] HWiNFO64は実行されていません")
            return True
    except Exception as e:
        print(f"[ERROR] HWiNFO64の終了に失敗: {e}")
        return False

def start_hwinfo():
    """HWiNFO64を起動（管理者権限で）"""
    hwinfo_path = find_hwinfo()
    
    if not hwinfo_path:
        print("[ERROR] HWiNFO64.exeが見つかりません")
        print("[INFO] 以下のパスを確認してください:")
        print("  - C:\\Program Files\\HWiNFO64\\HWiNFO64.EXE")
        print("  - C:\\Program Files (x86)\\HWiNFO64\\HWiNFO64.EXE")
        return False
    
    try:
        # HWiNFO64を管理者権限で起動
        if sys.platform == 'win32':
            # PowerShellを使用して管理者権限で起動
            # ShellExecuteExを使用してUAC昇格を試みる
            try:
                import ctypes
                # ShellExecuteWを使用（管理者権限で起動を試みる）
                # runasを使う
                result = subprocess.run(
                    ['powershell', '-Command', 
                     f'Start-Process -FilePath "{hwinfo_path}" -Verb RunAs'],
                    capture_output=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print(f"[INFO] HWiNFO64を起動しました: {hwinfo_path}")
                    return True
            except Exception:
                # フォールバック: 通常の方法で起動を試みる
                pass
            
            # 通常の方法でも試み（既に管理者権限の場合）
            try:
                subprocess.Popen(
                    [hwinfo_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                print(f"[INFO] HWiNFO64を起動しました: {hwinfo_path}")
                return True
            except Exception as e2:
                # それでも失敗する場合は、cmd経由でrunasを試み
                try:
                    subprocess.Popen(
                        ['cmd', '/c', f'start "" "{hwinfo_path}"'],
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    print(f"[INFO] HWiNFO64を起動しました: {hwinfo_path}")
                    return True
                except Exception:
                    raise e2
        else:
            # 非Windows環境
            subprocess.Popen(
                [hwinfo_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"[INFO] HWiNFO64を起動しました: {hwinfo_path}")
            return True
    except Exception as e:
        print(f"[ERROR] HWiNFO64の起動に失敗: {e}")
        print("[INFO] 手動でHWiNFO64を管理者として起動してください")
        return False

def restart_hwinfo():
    """HWiNFO64を再起動"""
    print("[INFO] HWiNFO64を再起動します...")
    kill_hwinfo()
    return start_hwinfo()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'restart':
        # 再起動のみ実行
        restart_hwinfo()
    else:
        # テスト実行
        print("HWiNFO64自動再起動スクリプト（テストモード）")
        print("=" * 50)
        hwinfo_path = find_hwinfo()
        if hwinfo_path:
            print(f"[OK] HWiNFO64が見つかりました: {hwinfo_path}")
            print("\n再起動する場合は以下を実行してください:")
            print(f"  python {__file__} restart")
        else:
            print("[ERROR] HWiNFO64.exeが見つかりません")

