#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リモートサーバー起動スクリプト
"""

import sys
import subprocess
from pathlib import Path

def main():
    """リモートサーバーを起動"""
    script_dir = Path(__file__).resolve().parent
    server_script = script_dir / 'remote_server.py'
    
    if not server_script.exists():
        print(f"エラー: {server_script} が見つかりません")
        return
    
    print("リモートサーバーを起動しています...")
    print("ブラウザで http://localhost:5000 にアクセスしてください")
    print("別PCからは http://<このPCのIPアドレス>:5000 にアクセスしてください")
    print()
    print("Ctrl+C で停止")
    print()
    
    try:
        subprocess.run([sys.executable, str(server_script)], check=True)
    except KeyboardInterrupt:
        print("\nサーバーを停止しました")
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == '__main__':
    main()




