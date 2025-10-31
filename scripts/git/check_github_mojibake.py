#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHubリポジトリ内の文字化けを確認するスクリプト
"""

import sys
import os
import subprocess

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

def check_github_commits():
    """GitHubコミットメッセージの文字化けを確認"""
    print("=" * 60)
    print("GitHubコミットメッセージの文字化けチェック")
    print("=" * 60)
    print()
    
    repo = git.Repo('.')
    
    # 文字化けパターン
    mojibake_patterns = [
        'を', 'ス', 'の', '情', '報', '譖', '謨', '邨', '実', '翫', 
        '蜿', '中', '出', '検', '讒', '謾', '謗', '鬆', '霎', '槭',
        '蜑', '企', '勁', '蜻', '崎', '丞', '援', '髢', '騾'
    ]
    
    print("[1] ローカルコミットのチェック...")
    local_issues = []
    commits = list(repo.iter_commits('HEAD', max_count=50))
    
    for commit in commits:
        message = commit.message.strip()
        has_mojibake = any(pattern in message for pattern in mojibake_patterns)
        
        if has_mojibake:
            local_issues.append({
                'sha': commit.hexsha[:7],
                'message': message[:80],
                'date': commit.committed_datetime.strftime('%Y-%m-%d')
            })
            print(f"  [文字化け] {commit.hexsha[:7]}: {message[:60]}...")
    
    print(f"  発見: {len(local_issues)}件")
    
    # リモートブランチとの比較
    print("\n[2] リモートブランチの確認...")
    try:
        origin = repo.remote('origin')
        for ref in origin.refs:
            branch_name = ref.name.replace('origin/', '')
            print(f"  ブランチ: {branch_name}")
            
            # リモートブランチのコミットを確認
            try:
                remote_commits = list(repo.iter_commits(f'origin/{branch_name}', max_count=10))
                remote_mojibake_count = 0
                for commit in remote_commits:
                    message = commit.message.strip()
                    if any(pattern in message for pattern in mojibake_patterns):
                        remote_mojibake_count += 1
                
                if remote_mojibake_count > 0:
                    print(f"    → 文字化けコミット: {remote_mojibake_count}件")
            except Exception as e:
                print(f"    → エラー: {e}")
    except Exception as e:
        print(f"  リモートリポジトリの確認に失敗: {e}")
    
    # サマリー
    print("\n" + "=" * 60)
    print("チェック結果サマリー")
    print("=" * 60)
    print(f"  ローカルコミットの文字化け: {len(local_issues)}件")
    
    if len(local_issues) > 0:
        print("\n文字化けしているコミット:")
        for issue in local_issues[:10]:  # 最大10件表示
            print(f"    {issue['sha']} ({issue['date']}): {issue['message'][:50]}...")
    
    return local_issues

def check_uncommitted_changes():
    """未コミットの変更を確認"""
    print("\n[3] 未コミットの変更を確認...")
    repo = git.Repo('.')
    
    if repo.is_dirty():
        print("  未コミットの変更があります")
        print("  修正したファイル:")
        
        # ステージングされていない変更
        for item in repo.index.diff(None):
            if item.change_type in ['M', 'A']:
                print(f"    {item.a_path}")
        
        # ステージングされた変更
        if len(repo.index.diff('HEAD')) > 0:
            print("  ステージング済み:")
            for item in repo.index.diff('HEAD'):
                print(f"    {item.a_path}")
    else:
        print("  未コミットの変更はありません")

if __name__ == "__main__":
    issues = check_github_commits()
    check_uncommitted_changes()
    
    print("\n" + "=" * 60)
    if len(issues) == 0:
        print("✓ GitHubリポジトリに文字化けは見つかりませんでした！")
    else:
        print(f"⚠ {len(issues)}件の文字化けコミットが見つかりました")
        print("\n過去のコミットメッセージを修正するには:")
        print("  python scripts/fix_all_github_commits.py")
        print("  または")
        print("  git filter-branch を使用（注意: 履歴を書き換えまム）")
    print("=" * 60)

