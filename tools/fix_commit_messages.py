#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix garbled commit messages in Git history
This script will correct the encoding of past commit messages
"""

import subprocess
import sys
import os
import re

# Set UTF-8 encoding
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LANG'] = 'ja_JP.UTF-8'
    os.environ['LC_ALL'] = 'ja_JP.UTF-8'

# Mapping of garbled messages to correct messages
COMMIT_FIXES = {
    # Pattern matching and correction
    '情報: PROJECT_STRUCTURE.md': '修正: PROJECT_STRUCTURE.md',
    'の主要ファイル説明セクションを更新': 'の主要ファイル説明セクションを更新',
    '譖ｴがｰ: docs/PROJECT_STRUCTURE.md': '更新: docs/PROJECT_STRUCTURE.md',
    'のGitHubドキュメントを更新': 'のGitHubドキュメントを更新',
    '譖ｴがｰ: プロジェクト': '更新: プロジェクト',
    'ファイル整理と整理': 'ファイル整理と整理',
    '整理: ファイル': '整理: ファイル',
}

def fix_commit_message(garbled_msg):
    """Attempt to fix a garbled commit message"""
    # Try to detect common patterns and fix them
    # This is a simplified version - full fix would require understanding the original encoding
    
    # Common garbled patterns and their fixes
    fixes = {
        '情報': '修正',
        '譖ｴがｰ': '更新',
        '謨ｴ逅テ': '整理',
        'の': 'の',
        'のｧ': 'を',
        'のｨ': 'に',
        'を': '',  # Common garbled character, often removed
        'ス': '',  # Common garbled character
        '報': '',  # Common garbled character
        '譏': '',  # Common garbled character
    }
    
    fixed = garbled_msg
    for garbled, correct in fixes.items():
        fixed = fixed.replace(garbled, correct)
    
    return fixed

def get_commit_list(count=10):
    """Get list of recent commits"""
    env = os.environ.copy()
    env['LANG'] = 'ja_JP.UTF-8'
    env['LC_ALL'] = 'ja_JP.UTF-8'
    
    result = subprocess.run(
        ['git', 'log', f'-{count}', '--format=%H|%s'],
        env=env,
        encoding='utf-8',
        capture_output=True,
        text=True
    )
    
    commits = []
    for line in result.stdout.strip().split('\n'):
        if '|' in line:
            sha, msg = line.split('|', 1)
            commits.append((sha.strip(), msg.strip()))
    
    return commits

def main():
    print("=" * 60)
    print("Gitコミットメッセージの文字化け修正ツール")
    print("=" * 60)
    print("\n注意: このツールは過去のコミットメッセージを確認します")
    print("実際の修正は手動で行う必要があります\n")
    
    commits = get_commit_list(15)
    
    print(f"最近のコミット ({len(commits)}件):")
    print("-" * 60)
    
    garbled_count = 0
    for i, (sha, msg) in enumerate(commits, 1):
        # Check if message appears garbled (contains common garbled patterns)
        is_garbled = any(char in msg for char in ['情', '報', 'を', 'ス', '譏', 'の'])
        
        if is_garbled:
            garbled_count += 1
            print(f"\n[{i}] {sha[:8]} - 文字化け検出")
            print(f"    現在: {msg}")
            
            # Attempt to fix
            fixed = fix_commit_message(msg)
            if fixed != msg:
                print(f"    修正候補: {fixed}")
        else:
            print(f"[{i}] {sha[:8]} - {msg}")
    
    print("\n" + "=" * 60)
    print(f"文字化け検出: {garbled_count}件")
    print("=" * 60)
    
    if garbled_count > 0:
        print("\n重要:")
        print("過去のコミットメッセージを修正するには、以下を実行してください:")
        print("1. git rebase -i <最初のコミットの親>")
        print("2. 各コミットで 'reword' を指定")
        print("3. コミットメッセージを正しい文字列に変更")
        print("4. git push --force-with-lease origin main")
        print("\n警告: 履歴を書き換えるため、既に共有されている場合は注意が必要です")

if __name__ == "__main__":
    main()


