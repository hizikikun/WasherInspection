#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
すべての文字化けを修正し、ファイル整理を行うスクリプト
"""

import os
import sys
import shutil
from pathlib import Path

# UTF-8エンコーディング設定（安全な方法）
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def fix_file_encoding(file_path):
    """ファイルのエンコーディングを修正"""
    try:
        # 複数のエンコーディングで試み
        content = None
        for encoding in ['utf-8', 'cp932', 'shift-jis', 'windows-1252']:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
                break
            except Exception:
                continue
        
        if content:
            # UTF-8で保存
            with open(file_path, 'w', encoding='utf-8', errors='replace') as f:
                f.write(content)
            return True
    except Exception:
        pass
    return False

def organize_hwinfo_files():
    """HWiNFO関連ファイルを整理"""
    scripts_dir = Path('scripts')
    
    # 統合ムべきファイル
    files_to_organize = {
        # メインのhwinfo_reader.pyを優先、他の改善版は削除または統合
        'hwinfo_reader_improved.py': 'hwinfo_reader.py',  # 改善版をメインに統合
        'hwinfo_reader_fixed.py': None,  # 削除（不要）
        
        # テストファイルの整理
        'test_hwinfo_integration.py': 'hwinfo_test.py',
        'test_hwinfo_detailed.py': 'hwinfo_debug.py',
        'debug_hwinfo_all_readings.py': 'hwinfo_debug_all.py',
        'check_hwinfo_status.py': 'hwinfo_status.py',
        
        # バッチファイルの整理
        'setup_hwinfo_scheduler_admin.bat': 'hwinfo_setup_admin.bat',
        'setup_hwinfo_scheduler.bat': 'hwinfo_setup.bat',
        'setup_hwinfo_scheduler.ps1': 'hwinfo_setup.ps1',
        'setup_hwinfo_scheduler_v2.ps1': None,  # 削除（v2は不要）
        'restart_hwinfo.bat': 'hwinfo_restart.bat',
    }
    
    print("HWiNFO関連ファイルの整理...")
    for old_name, new_name in files_to_organize.items():
        old_path = scripts_dir / old_name
        if old_path.exists():
            if new_name:
                new_path = scripts_dir / new_name
                if new_path.exists() and old_name != 'hwinfo_reader_improved.py':
                    # 既に存在する場合は統合
                    print(f"  Skip: {old_name} (already exists: {new_name})")
                else:
                    try:
                        if old_name == 'hwinfo_reader_improved.py':
                            # 改善版の内容をメインに統合
                            with open(old_path, 'r', encoding='utf-8') as f:
                                improved_content = f.read()
                            with open(scripts_dir / 'hwinfo_reader.py', 'w', encoding='utf-8') as f:
                                f.write(improved_content)
                            print(f"  Merged: {old_name} -> hwinfo_reader.py")
                            old_path.unlink()
                        else:
                            old_path.rename(new_path)
                            print(f"  Renamed: {old_name} -> {new_name}")
                            fix_file_encoding(new_path)
                    except Exception as e:
                        print(f"  Error: {old_name} -> {e}")
            else:
                # 削除
                try:
                    old_path.unlink()
                    print(f"  Deleted: {old_name}")
                except Exception as e:
                    print(f"  Error deleting {old_name}: {e}")

def fix_console_output():
    """すべてのPythonスクリプトのコンソール出力を修正"""
    scripts_dir = Path('scripts')
    
    # 修正対象ファイル（HWiNFO関連）
    target_files = [
        'hwinfo_reader.py',
        'hwinfo_auto_restart.py',
        'hwinfo_test.py',
        'hwinfo_debug.py',
        'hwinfo_status.py',
        'check_hwinfo_status.py',
    ]
    
    print("\nコンソール出力の文字化け修正...")
    for filename in target_files:
        file_path = scripts_dir / filename
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                # エンコーディング設定を追加（まだない場合）
                if '# -*- coding: utf-8 -*-' not in content[:100]:
                    content = '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n' + content
                
                # UTF-8で保存
                with open(file_path, 'w', encoding='utf-8', errors='replace') as f:
                    f.write(content)
                
                print(f"  Fixed: {filename}")
            except Exception as e:
                print(f"  Error fixing {filename}: {e}")

def main():
    print("=" * 60)
    print("文字化け修正とファイル整理")
    print("=" * 60)
    print()
    
    # HWiNFO関連ファイルの整理
    organize_hwinfo_files()
    
    # コンソール出力の修正
    fix_console_output()
    
    print()
    print("=" * 60)
    print("完了")
    print("=" * 60)

if __name__ == '__main__':
    main()

