#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
すべてのコミットメッセージを修正して新しい履歴を作成
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

# 修正マッピング
COMMIT_FIXES = {
    'ec67b4b3d0c911e54cd40d640d90527872128e6c': '修正: PROJECT_STRUCTURE.mdの主要ファイル説明セクションを更新',
    '2a1e872facac538ad6ed2bb02e51d046083da0ca': '更新: docs/PROJECT_STRUCTURE.mdのGitHubドキュメントを更新',
    '44b40594462e95b3b0dd239a30cc782a8417005e': '更新: プロジェクトファイル整理とリネーム',
    '7f220d64a2f0d408961804441132f81b8a5a7ca3': '更新: .gitignoreにディレクトリとignoreファイルを追加',
    'ceb8209a0fdcbf241bfad20c29d57dda58d20fee': '整理: ファイル整理とリネーム - ultra_, final_, advanced_プレフィックス削除',
    '08200b6456f1a2f40493a29784d9765efd49d066': 'fix: safe_git_commit.pyをGitPythonを使用するように修正',
    '248815ef1e6fd38a10720a35c9b349ca618fe7af': 'chore: Git UTF-8設定ツールとコミットメッセージ修正ツールを追加',
    '862abb8ad7a60de61e2a4a7aa4d3de2d9731656d': 'chore: Git UTF-8設定ツールとコミットメッセージ修正ツールを追加',
}

def rebuild_history():
    """履歴を再構築"""
    repo = git.Repo('.')
    
    print("=" * 60)
    print("コミット履歴の再構築")
    print("=" * 60)
    
    # すべてのコミットを取得（古い順）
    all_commits = list(repo.iter_commits('--all'))
    all_commits.reverse()
    
    print(f"\n総コミット数: {len(all_commits)}")
    print(f"修正対象: {len(COMMIT_FIXES)}件\n")
    
    # 修正が必要なコミットを表示
    need_fix = []
    for commit in all_commits:
        if commit.hexsha in COMMIT_FIXES:
            need_fix.append(commit)
            old_msg = commit.message.strip()
            new_msg = COMMIT_FIXES[commit.hexsha]
            print(f"[{commit.hexsha[:8]}] {old_msg[:50]}...")
            print(f"        -> {new_msg}")
    
    print("\n" + "=" * 60)
    print("修正方法:")
    print("1. 各コミットをチェックアウト")
    print("2. メッセージを修正してコミット")
    print("3. 新しい履歴を作成")
    print("\nこれは複雑な操作です。")
    print("代わりに、以下のコマンドで手動修正してください:\n")
    
    # 最も古い修正対象コミットの親を取得
    oldest = need_fix[0]
    parent = oldest.parents[0] if oldest.parents else None
    
    if parent:
        print(f"git rebase -i {parent.hexsha[:8]}")
        print("\nエディタで各コミットの 'pick' を 'reword' に変更してください。")
        print("その後、各コミットメッセージを修正してください。")
    
    print("\nまたは、以下のコマンドで一括修正を試行します:")
    print("(注意: この操作は履歴を書き換えまム)")
    
    response = input("\n一括修正を試行しますか？ (yes/no): ")
    if response.lower() != 'yes':
        print("キャンセルされました。")
        return
    
    # 各コミットを個別に修正
    # これは複雑なので、git rebase -i の自動化を試みる
    print("\n各コミットを個別に修正していまム...")
    
    # 実際には、git filter-branchがうまく動かないので、
    # 直接gitコマンドでrebaseを実行
    
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LANG'] = 'ja_JP.UTF-8'
    
    # rebase todoファイルを作成
    if parent:
        rebase_commands = []
        for commit in all_commits:
            if commit.hexsha in COMMIT_FIXES:
                rebase_commands.append(f"reword {commit.hexsha[:8]} {COMMIT_FIXES[commit.hexsha]}")
            elif commit.hexsha == parent.hexsha:
                continue
            else:
                rebase_commands.append(f"pick {commit.hexsha[:8]}")
        
        print("\n修正コマンドリスト:")
        for cmd in rebase_commands[:10]:  # 最初の10個だけ表示
            print(cmd)
        if len(rebase_commands) > 10:
            print(f"... 他 {len(rebase_commands) - 10} 件")
        
        print("\n注意: この方法は複雑です。")
        print("手動で git rebase -i を実行することを推奨します。")

if __name__ == "__main__":
    rebuild_history()


