#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buildディレクトリ内の文字化けファイル名を修正
"""

import os
import sys
from pathlib import Path

if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def fix_build_files():
    """buildディレクトリ内の文字化けファイル名を修正"""
    build_dir = Path('build/ResinWasherInspection')
    backup_dir = Path('backup/backup_20251029_230637/build/ResinWasherInspection')
    
    replacements = {
        '樹ｹ脂ッシステムススｼ読解渊をｷをｹステΒ': 'ResinWasherInspection'
    }
    
    fixed_count = 0
    
    for target_dir in [build_dir, backup_dir]:
        if not target_dir.exists():
            continue
        
        print(f"処理中: {target_dir}")
        for file_path in target_dir.iterdir():
            if not file_path.is_file():
                continue
            
            old_name = file_path.name
            new_name = old_name
            
            # 文字化けパターンを置換
            for old, new in replacements.items():
                if old in new_name:
                    new_name = new_name.replace(old, new)
            
            if new_name != old_name:
                new_path = file_path.parent / new_name
                try:
                    if new_path.exists():
                        # 既に存在する場合は削除
                        file_path.unlink()
                        print(f"  削除（既存）: {old_name}")
                    else:
                        file_path.rename(new_path)
                        print(f"  リネーム: {old_name} -> {new_name}")
                    fixed_count += 1
                except Exception as e:
                    print(f"  エラー: {old_name} -> {e}")
    
    print(f"\n修正完了: {fixed_count}件")
    return fixed_count

if __name__ == "__main__":
    fix_build_files()

