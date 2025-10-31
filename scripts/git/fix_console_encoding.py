#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
すべてのPythonスクリプトのコンソール出力文字化けを修正
"""

import os
import sys
from pathlib import Path
import re

# UTF-8エンコーディング設定（安全な方法）
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def add_encoding_setup(content):
    """スクリプトにエンコーディング設定を追加"""
    # 既にエンコーディング設定があるかチェック
    if 'io.TextIOWrapper' in content or 'codecs.getwriter' in content:
        return content
    
    # shebangの後にエンコーディング設定を追加
    encoding_setup = '''# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
'''
    
    # sysがインポートされているかチェック
    if 'import sys' not in content:
        # 最初のimport文の後に追加
        if content.startswith('#!/usr/bin/env python3'):
            lines = content.split('\n')
            insert_pos = 2  # shebangとcoding行の後
            if len(lines) > insert_pos and 'import' in lines[insert_pos]:
                lines.insert(insert_pos, 'import sys')
            else:
                lines.insert(insert_pos, 'import sys')
            content = '\n'.join(lines)
    
    # sysのインポートの後にエンコーディング設定を追加
    if 'import sys' in content and 'io.TextIOWrapper' not in content:
        # import sysの直後に追加
        content = re.sub(
            r'(import sys\n)',
            r'\1' + encoding_setup,
            content,
            count=1
        )
    
    return content

def fix_all_scripts():
    """すべてのPythonスクリプトを修正"""
    scripts_dir = Path('scripts')
    
    # 修正対象ファイル
    target_files = list(scripts_dir.glob('*.py'))
    
    print("=" * 60)
    print("すべてのPythonスクリプトの文字化け修正")
    print("=" * 60)
    print()
    
    fixed_count = 0
    for file_path in target_files:
        try:
            # UTF-8で読み込み
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # エンコーディング設定を追加
            new_content = add_encoding_setup(content)
            
            if new_content != content:
                # UTF-8で保存
                with open(file_path, 'w', encoding='utf-8', errors='replace') as f:
                    f.write(new_content)
                print(f"  Fixed: {file_path.name}")
                fixed_count += 1
        except Exception as e:
            print(f"  Error fixing {file_path.name}: {e}")
    
    print()
    print(f"修正完了: {fixed_count} ファイル")
    print("=" * 60)

if __name__ == '__main__':
    fix_all_scripts()

