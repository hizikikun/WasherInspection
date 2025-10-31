#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ファイル内容内の文字化けをクリーンアップ
（マッピング定義は除外）
"""

import os
import sys
from pathlib import Path

if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    os.environ['PYTHONIOENCODING'] = 'utf-8'

CONTENT_REPLACEMENTS = [
    ("樹ｹ脂ッシステムススｼ読解渊をｷをｹステΒ", "ResinWasherInspection"),
    ("樹脂ワッシャー検査システム", "ResinWasherInspection"),
    ("樹脂ワチEャー検査シスチE", "ResinWasherInspection"),
]

def clean_content(file_path):
    """ファイル内容の文字化けを修正（マッピング定義は除く）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # マッピング定義ファイルはスキップ
        if 'MOJIBAKE_MAPPINGS' in content or 'CONTENT_REPLACEMENTS' in content or 'exact_fixes' in content:
            return False
        
        original = content
        for old, new in CONTENT_REPLACEMENTS:
            if old in content:
                content = content.replace(old, new)
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    except Exception:
        return False

def main():
    """メイン処理"""
    target_files = [
        Path('.specstory/history/2025-10-22_04-43Z-githubへの自動転送システムの修正.md'),
    ]
    
    fixed_count = 0
    for file_path in target_files:
        if file_path.exists():
            if clean_content(file_path):
                print(f"修正: {file_path}")
                fixed_count += 1
    
    print(f"\n修正完了: {fixed_count}件")

if __name__ == "__main__":
    main()

