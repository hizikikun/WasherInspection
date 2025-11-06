#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
すべてのコミットメッセージの文字化けを修正
"""

import subprocess
import sys
import os
import re

# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def check_mojibake(text):
    """文字化けをチェック"""
    # よくある文字化け文字
    mojibake_chars = ['縺', '繧', '繝', '豁', '菫', '譁', '螳', '險', '險', '荳', '隱', '譖', '譖']
    return any(char in text for char in mojibake_chars)

def get_all_commits_with_mojibake():
    """文字化けしているコミットをすべて取得"""
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    result = subprocess.run(
        ['git', 'log', '--format=%H|%s', '--all'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    commits = []
    for line in result.stdout.strip().split('\n'):
        if '|' in line:
            hash_val, message = line.split('|', 1)
            if check_mojibake(message):
                commits.append({
                    'hash': hash_val,
                    'message': message
                })
    
    return commits

def suggest_fixed_message(message):
    """修正されたメッセージを提案"""
    # パターンマッチングで修正
    patterns = {
        r'chore:\s*Git\s*UTF-8[^\x00-\x7F]+': 'chore: Git UTF-8エンコーディングとセキュリティ設定',
        r'菫ｮ豁｣:.*PROJECT_STRUCTURE': '更新: PROJECT_STRUCTURE.mdのGitHubコミット修正',
        r'譖ｴ譁ｰ:.*PROJECT_STRUCTURE': '修正: docs/PROJECT_STRUCTURE.mdのGitHubコミット修正',
        r'譖ｴ譁ｰ:.*gitignore': '修正: .gitignoreにトークンとシークレット追加',
        r'謨ｴ逅・.*整理': '整理: ファイルとフォルダの整理',
    }
    
    for pattern, replacement in patterns.items():
        if re.search(pattern, message):
            return replacement
    
    # 文字化けだけを削除して、意味のある部分を残す
    if 'PROJECT_STRUCTURE' in message or 'PROJECT_STRUCTURE' in message.upper():
        return '更新: PROJECT_STRUCTURE.md修正'
    elif 'gitignore' in message.lower():
        return '修正: .gitignore更新'
    elif 'Git' in message or 'UTF' in message:
        return 'chore: Git UTF-8エンコーディング設定'
    elif '整理' in message or '整理' in message:
        return '整理: ファイル整理'
    else:
        return 'コミットメッセージ修正'

def main():
    print("=" * 60)
    print("コミットメッセージの文字化けチェック")
    print("=" * 60)
    print()
    
    commits = get_all_commits_with_mojibake()
    
    if not commits:
        print("✓ 文字化けしているコミットは見つかりませんでした")
        return
    
    print(f"見つかった文字化けコミット: {len(commits)}個")
    print()
    
    for i, commit in enumerate(commits[:10], 1):  # 最初の10個だけ表示
        print(f"[{i}] {commit['hash'][:8]}")
        print(f"  元: {commit['message'][:60]}")
        fixed = suggest_fixed_message(commit['message'])
        print(f"  修正後: {fixed}")
        print()
    
    if len(commits) > 10:
        print(f"  ... 他 {len(commits) - 10}個のコミット")
        print()
    
    print("=" * 60)
    print("注意")
    print("=" * 60)
    print()
    print("コミット履歴の文字化けを修正するには、以下の方法があります：")
    print()
    print("1. 新しいコミットで履歴を上書き（推奨）")
    print("   - git reset --soft origin/main")
    print("   - 新しいコミットを作成")
    print()
    print("2. 個別に修正（時間がかかります）")
    print("   - git rebase -i <最初のコミット>")
    print("   - 各コミットのメッセージを修正")
    print()
    print("既にリモートにプッシュされている場合は、force pushが必要です：")
    print("   git push --force-with-lease origin main")

if __name__ == '__main__':
    main()
