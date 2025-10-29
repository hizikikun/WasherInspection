#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub関連ファイル整理スクリプト
使われていないファイルをold/に移動
"""

import os
import shutil
from pathlib import Path

def organize_github_files():
    """GitHub関連ファイルを整理"""
    base_dir = Path(__file__).parent
    github_tools_dir = base_dir / 'github_tools'
    old_dir = base_dir / 'old' / 'github_unused'
    
    print(f"[整理] 作業ディレクトリ: {base_dir}")
    
    # old/github_unused/ディレクトリ作成
    old_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [OK] old/github_unused/ ディレクトリ作成")
    
    # ファイルの依存関係を確認
    # メインファイル: github_sync.py（統合GitHubシステム）
    # これが使うファイル: github_autocommit.py, auto_sync.py
    
    # 未使用と判断されるファイル（重複・古い機能）
    unused_files = {
        'sync.py': 'github_sync.py と機能重複（IntegratedAutoSyncは未使用）',
        'cursor_extension.py': 'GUIツール（実際には使われていない可能性）',
        'github_integration.py': '古い統合システム（github_sync.py に置き換え済み）',
    }
    
    # 使用中ファイル（メインファイルが参照）
    used_files = {
        'github_sync.py': '統合GitHubシステム（メイン）',
        'github_autocommit.py': '自動コミットシステム（github_sync.pyから使用）',
        'auto_sync.py': '自動同期システム（github_sync.pyから使用）',
        'cursor_integration.py': 'Cursor統合（実際に使用中か確認が必要）',
        'token_setup.py': 'トークン設定（設定時に使用）',
    }
    
    moved_count = 0
    
    print("\n[整理] 未使用ファイルをold/github_unused/に移動...")
    for filename, reason in unused_files.items():
        src = github_tools_dir / filename
        dst = old_dir / filename
        
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"  [OK] {filename}")
            print(f"       -> old/github_unused/ ({reason})")
            moved_count += 1
        else:
            print(f"  [NOT FOUND] {filename}")
    
    print(f"\n[整理] {moved_count}個のファイルを移動しました")
    
    print("\n[整理] 残されたファイル（使用中）:")
    for filename, desc in used_files.items():
        src = github_tools_dir / filename
        if src.exists():
            print(f"  [ACTIVE] {filename} - {desc}")
    
    # 最終的なファイル一覧
    print("\n[整理] github_tools/ の最終構成:")
    remaining_files = list(github_tools_dir.glob('*.py'))
    for f in sorted(remaining_files):
        print(f"  - {f.name}")

if __name__ == "__main__":
    try:
        organize_github_files()
    except Exception as e:
        print(f"\n[エラー] {e}")
        import traceback
        traceback.print_exc()

