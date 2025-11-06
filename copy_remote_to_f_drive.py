#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fドライブに遠隔操作関連ファイルをコピーするスクリプト
"""

import os
import shutil
from pathlib import Path

# ソースとターゲットパス
SOURCE_DIR = Path(r"C:\Users\tomoh\WasherInspection")
TARGET_DIR = Path(r"F:\cs\washerinspection")

def copy_file(src, dst):
    """ファイルをコピー"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
        print(f"  [コピー] {src.name}")
        return True
    except Exception as e:
        print(f"  [エラー] {src.name}: {e}")
        return False

def main():
    print("=" * 60)
    print("遠隔操作関連ファイルをFドライブにコピーします")
    print("=" * 60)
    print()
    
    # 遠隔操作関連のスクリプト
    remote_scripts = [
        "scripts/remote_server.py",
        "scripts/start_remote_server.py",
        "scripts/remote_server_tunnel.py",
        "scripts/auto_setup_remote_access.py",
        "scripts/remote_transfer.py",
        "scripts/configure_remote_transfer.py",
        "scripts/configure_tunnel.py"
    ]
    
    print("[1/3] 遠隔操作スクリプトをコピー中...")
    scripts_dst = TARGET_DIR / "scripts"
    scripts_dst.mkdir(parents=True, exist_ok=True)
    
    for script_path in remote_scripts:
        src_file = SOURCE_DIR / script_path
        if src_file.exists():
            dst_file = scripts_dst / Path(script_path).name
            copy_file(src_file, dst_file)
        else:
            print(f"  [警告] {script_path} が見つかりません")
    
    # バッチファイル
    print("\n[2/3] バッチファイルをコピー中...")
    batch_files = [
        "start_remote_server.bat"
    ]
    
    for batch_file in batch_files:
        src_file = SOURCE_DIR / batch_file
        if src_file.exists():
            copy_file(src_file, TARGET_DIR / batch_file)
        else:
            print(f"  [警告] {batch_file} が見つかりません")
    
    # READMEファイル
    print("\n[3/3] READMEファイルをコピー中...")
    readme_files = [
        "README_remote_control.md",
        "README_remote_transfer.md",
        "README_remote_tunnel.md"
    ]
    
    for readme_file in readme_files:
        src_file = SOURCE_DIR / readme_file
        if src_file.exists():
            copy_file(src_file, TARGET_DIR / readme_file)
        else:
            print(f"  [警告] {readme_file} が見つかりません")
    
    print("\n" + "=" * 60)
    print("コピー完了！")
    print("=" * 60)
    print("\n使い方は README_遠隔操作機能.md を参照してください。")

if __name__ == "__main__":
    main()




