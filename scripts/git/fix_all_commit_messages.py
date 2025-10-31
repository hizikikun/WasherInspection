#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
過去のすべてのコミットメッセージを修正するスクリプト
文字化けしたコミットメッセージを正しいUTF-8メッセージに修正
"""

import sys
import os
import subprocess
import re

if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    import git
except ImportError:
    print("エラー: GitPythonが必要です。'pip install GitPython'を実行してください。")
    sys.exit(1)

# 文字化けしたメッセージから正しいメッセージへのマッピング
COMMIT_MESSAGE_FIXES = {
    # SHA: (old_garbled, new_correct)
    # 実際のSHAは実行時に特定
}

def detect_garbled_message(msg):
    """メッセージが文字化けしているか検出"""
    garbled_patterns = [
        '情', '報', 'を', 'ス', 'の', '譏', '謨', '譖', '邨', 
        '実', 'ス', 'ス', '翫', '蜿', '中', '出', '検'
    ]
    return any(pattern in msg for pattern in garbled_patterns)

def fix_message(garbled_msg):
    """文字化けしたメッセージを修正（パターンマッチング）"""
    # 既知のパターンから修正を試みる
    fixes = {
        '情報': '修正',
        '譖ｴがｰ': '更新',
        '謨ｴ逅テ': '整理',
        'の': 'の',
        'のｧ': 'を',
        'のｨ': 'に',
        'のｫ': 'に',
        'を｡を､スｫ': 'ファイル',
        'ス｡スｳス亥テ': 'プロジェクト',
        'ステぅスｬをｯ': 'cursor',
        'ステテスｫ': 'ドキュメント',
        'をｯスｪス励ヨ': 'ignore',
        'ス舌ャをｯを｢': 'ディレクトリ',
        '情報': '修正',
    }
    
    fixed = garbled_msg
    for garbled, correct in fixes.items():
        fixed = fixed.replace(garbled, correct)
    
    # 特定のパターンを検出して修正
    if 'PROJECT_STRUCTURE.md' in garbled_msg:
        if '情報' in garbled_msg or '修正' in fixed:
            fixed = '修正: PROJECT_STRUCTURE.mdの主要ファイル説明セクションを更新'
    elif 'docs/PROJECT_STRUCTURE.md' in garbled_msg:
        if '譖ｴがｰ' in garbled_msg or '更新' in fixed:
            fixed = '更新: docs/PROJECT_STRUCTURE.mdのGitHubドキュメントを更新'
    elif '.gitignore' in garbled_msg:
        fixed = '更新: .gitignoreにディレクトリとignoreファイルを追加'
    elif 'ス輔ぃを､スｫ' in garbled_msg or 'ファイル' in fixed:
        if '整理' in fixed or '謨ｴ逅テ' in garbled_msg:
            fixed = '整理: ファイル整理とリネーム - ultra_, final_, advanced_などのプレフィックスを削除'
    
    return fixed

def get_all_commits():
    """すべてのコミットを取得"""
    repo = git.Repo('.')
    commits = []
    
    for commit in repo.iter_commits('--all'):
        commits.append({
            'sha': commit.hexsha,
            'message': commit.message.strip(),
            'short_sha': commit.hexsha[:8]
        })
    
    return commits

def rewrite_commit_messages():
    """すべてのコミットメッセージを修正"""
    repo = git.Repo('.')
    commits = get_all_commits()
    
    print("=" * 60)
    print("コミットメッセージ修正スクリプト")
    print("=" * 60)
    print(f"\n総コミット数: {len(commits)}")
    
    # 文字化けしたコミットを特定
    garbled_commits = []
    for commit in commits:
        if detect_garbled_message(commit['message']):
            garbled_commits.append(commit)
    
    print(f"文字化け検出: {len(garbled_commits)}件\n")
    
    if len(garbled_commits) == 0:
        print("文字化けしたコミットは見つかりませんでした。")
        return
    
    # 修正候補を表示
    print("修正対象コミット:")
    print("-" * 60)
    fixes = {}
    for commit in garbled_commits:
        fixed_msg = fix_message(commit['message'])
        if fixed_msg != commit['message']:
            fixes[commit['sha']] = fixed_msg
            print(f"\n[{commit['short_sha']}]")
            print(f"  現在: {commit['message'][:80]}...")
            print(f"  修正: {fixed_msg[:80]}...")
    
    if not fixes:
        print("修正可能なコミットが見つかりませんでした。")
        return
    
    print("\n" + "=" * 60)
    print("注意: この操作はGit履歴を書き換えまム")
    print("=" * 60)
    
    # 確認
    response = input("\n修正を実行しますか？ (yes/no): ")
    if response.lower() != 'yes':
        print("キャンセルされました。")
        return
    
    # rebase用のスクリプトを生成
    oldest_garbled_sha = garbled_commits[-1]['sha']  # 最も古い文字化けコミット
    
    print(f"\n最も古い文字化けコミット: {oldest_garbled_sha[:8]}")
    print("git rebase -i を実行して、各コミットで 'reword' を指定してください。")
    print("\nまたは、以下のPythonスクリプトで自動修正できまム:")
    print("=" * 60)
    
    # 自動修正スクリプトを生成
    auto_fix_script = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import git
import sys

repo = git.Repo('.')
fixes = {"""
    
    for sha, fixed_msg in fixes.items():
        # エスケープ
        fixed_msg_escaped = fixed_msg.replace("'", "\\'").replace("\n", "\\n")
        auto_fix_script += f'\n    "{sha}": "{fixed_msg_escaped}",'
    
    auto_fix_script += """
}

# 逆順で処理（古いコミットから）
for commit in list(repo.iter_commits('HEAD')):
    if commit.hexsha in fixes:
        old_msg = commit.message
        new_msg = fixes[commit.hexsha]
        print(f"修正: {commit.hexsha[:8]} - {old_msg[:50]} -> {new_msg[:50]}")
        
        # 注意: GitPythonでは直接コミットメッセージを変更できません
        # git filter-branch または git rebase が必要です
"""
    
    print(auto_fix_script)
    print("=" * 60)
    
    # 実際には git filter-branch または git rebase を使う必要がある
    print("\n実際の修正方法:")
    print("1. git filter-branch -f --msg-filter 'python fix_msg.py' HEAD")
    print("2. または、手動で git rebase -i <最初のコミットの親>")

if __name__ == "__main__":
    rewrite_commit_messages()


