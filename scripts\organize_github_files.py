#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub関連ファイルの整理と文字化け修正
"""

import os
import sys
import shutil
from pathlib import Path
import re

# UTF-8 encoding for console output (Windows)
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
        for encoding in ['utf-8', 'cp932', 'shift-jis', 'windows-1252', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
                # UTF-8で保存
                with open(file_path, 'w', encoding='utf-8', errors='replace') as f:
                    f.write(content)
                return True
            except Exception:
                continue
    except Exception as e:
        print(f"  Error fixing {file_path}: {e}")
    return False

def organize_github_scripts():
    """GitHub関連スクリプトを整理"""
    scripts_git_dir = Path('scripts/git')
    github_tools_dir = Path('github_tools')
    
    # 重複や不要なファイルをチェック
    files_to_check = list(scripts_git_dir.glob('*.py'))
    
    print("GitHub関連スクリプトの整理...")
    fixed_count = 0
    
    for file_path in files_to_check:
        # 文字化け修正
        if fix_file_encoding(file_path):
            fixed_count += 1
            print(f"  Fixed encoding: {file_path.name}")
    
    # github_tools内のファイルも修正
    if github_tools_dir.exists():
        for file_path in github_tools_dir.glob('*.py'):
            if fix_file_encoding(file_path):
                fixed_count += 1
                print(f"  Fixed encoding: {file_path.name}")
    
    print(f"✓ 文字化け修正: {fixed_count}個")

def fix_all_python_files():
    """すべてのPythonファイルの文字化けを修正"""
    target_dirs = [
        'scripts/git',
        'scripts/hwinfo',
        'scripts/training',
        'scripts/utils',
        'scripts/config',
        'github_tools',
        'tools',
    ]
    
    print("\n全Pythonファイルの文字化け修正...")
    total_fixed = 0
    
    for target_dir in target_dirs:
        dir_path = Path(target_dir)
        if not dir_path.exists():
            continue
        
        for file_path in dir_path.glob('*.py'):
            try:
                # エンコーディング設定の確認
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                # エンコーディング設定がない場合は追加
                if '# -*- coding: utf-8 -*-' not in content[:500]:
                    if content.startswith('#!/usr/bin/env python3'):
                        content = content.replace('#!/usr/bin/env python3\n', 
                                                  '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n', 1)
                    elif content.startswith('#!/usr/bin/env python'):
                        content = content.replace('#!/usr/bin/env python\n', 
                                                  '#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n', 1)
                    else:
                        content = '# -*- coding: utf-8 -*-\n' + content
                
                # UTF-8で保存
                with open(file_path, 'w', encoding='utf-8', errors='replace') as f:
                    f.write(content)
                total_fixed += 1
            except Exception as e:
                print(f"  Error fixing {file_path}: {e}")
    
    print(f"✓ 修正完了: {total_fixed}個")

def prepare_git_commit():
    """Gitコミットの準備"""
    print("\nGitコミットの準備...")
    
    # .gitignoreの確認
    gitignore_path = Path('.gitignore')
    if not gitignore_path.exists():
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
*.h5
*.csv
*.log
logs/
models/
temp/
backup/
dist/
build/
"""
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        print("  Created .gitignore")
    
    print("✓ Git準備完了")

def main():
    print("=" * 60)
    print("GitHub関連ファイルの整理と文字化け修正")
    print("=" * 60)
    print()
    
    organize_github_scripts()
    fix_all_python_files()
    prepare_git_commit()
    
    print()
    print("=" * 60)
    print("整理完了")
    print("=" * 60)
    print()
    print("次のステップ:")
    print("  1. git add .")
    print("  2. git commit -m 'ファイル整理と文字化け修正'")
    print("  3. git push")

if __name__ == '__main__':
    main()

