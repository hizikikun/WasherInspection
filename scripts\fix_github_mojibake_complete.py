#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHubの既存ファイルの文字化けをすべて修正
コメント、文字列、ファイル名すべてをチェック
"""

import os
import sys
import subprocess
import re
from pathlib import Path

# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def fix_file_comments_and_strings(file_path):
    """ファイル内のコメントと文字列の文字化けを修正"""
    try:
        # UTF-8で読み込み
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        original_content = content
        
        # よくある文字化けパターンの修正
        fixes = {
            # Shift-JIS → UTF-8の誤変換
            'データ': 'データ',
            'ファイル': 'ファイル',
            'エラー': 'エラー',
            'の': 'の',
            'が': 'が',
            'を': 'を',
            '設定': '設定',
            'ファイル': 'ファイル',
            'システム': 'システム',
            'システム': 'システム',
            'データ': 'データ',
            'ファイル': 'ファイル',
            # 半角カナの誤変換
            '': '',  # 文脈依存なので慎重に
            '': '',  # 文脈依存なので慎重に
        }
        
        # パターンマッチングでより安全に修正
        # 文字化け文字がコメントや文字列内にある場合のみ修正
        for pattern, replacement in fixes.items():
            if pattern in content:
                # コメント内または文字列内にあるかチェック
                lines = content.split('\n')
                fixed_lines = []
                for line in lines:
                    # コメント行または文字列内のパターンを修正
                    if '#' in line or '"' in line or "'" in line:
                        if pattern in line:
                            line = line.replace(pattern, replacement)
                    fixed_lines.append(line)
                content = '\n'.join(fixed_lines)
        
        # 変更があった場合のみ保存
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8', errors='replace') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"  Error fixing {file_path}: {e}")
        return False

def fix_all_github_files():
    """GitHubリポジトリ内のすべてのファイルを修正"""
    print("=" * 60)
    print("GitHubファイルの文字化け修正")
    print("=" * 60)
    print()
    
    target_dirs = [
        'scripts',
        'github_tools',
        'tools',
        'inspectors',
        'trainers',
        'utilities',
        'dashboard',
        'docs',
    ]
    
    total_files = 0
    fixed_files = 0
    
    for target_dir in target_dirs:
        dir_path = Path(target_dir)
        if not dir_path.exists():
            continue
        
        print(f"処理中: {target_dir}/")
        
        # Pythonファイル
        for file_path in dir_path.rglob('*.py'):
            total_files += 1
            if fix_file_comments_and_strings(file_path):
                fixed_files += 1
                print(f"  [修正] {file_path.name}")
        
        # Markdownファイル
        for file_path in dir_path.rglob('*.md'):
            total_files += 1
            if fix_file_comments_and_strings(file_path):
                fixed_files += 1
                print(f"  [修正] {file_path.name}")
    
    print()
    print("=" * 60)
    print("修正結果")
    print("=" * 60)
    print(f"処理ファイル数: {total_files}")
    print(f"修正ファイル数: {fixed_files}")
    print()

def commit_and_push():
    """修正をコミット・プッシュ"""
    print("\nGitコミット・プッシュ...")
    
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        # git add
        result = subprocess.run(['git', 'add', '.'], capture_output=True, text=True, 
                              encoding='utf-8', errors='replace', env=env)
        if result.returncode != 0:
            print(f"  git add エラー: {result.stderr}")
            return False
        
        # git commit
        result = subprocess.run(['git', 'commit', '-m', 'GitHubファイルの文字化け修正完了'], 
                              capture_output=True, text=True,
                              encoding='utf-8', errors='replace', env=env)
        if result.returncode != 0:
            if result.stdout and 'nothing to commit' in result.stdout:
                print("  変更がありません（既にコミット済み）")
                return True
            if result.stderr:
                print(f"  git commit エラー: {result.stderr}")
            return False
        
        if result.stdout:
            print(f"  コミット成功: {result.stdout.strip()}")
        else:
            print("  コミット成功")
        
        # git push
        result = subprocess.run(['git', 'push'], capture_output=True, text=True,
                              encoding='utf-8', errors='replace', env=env)
        if result.returncode != 0:
            if result.stderr:
                print(f"  git push エラー: {result.stderr}")
            print("  手動で実行してください: git push")
            return False
        
        print("  プッシュ成功")
        return True
    except Exception as e:
        print(f"  エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    fix_all_github_files()
    
    print()
    print("GitHubにコミット・プッシュします...")
    commit_and_push()

if __name__ == '__main__':
    main()

