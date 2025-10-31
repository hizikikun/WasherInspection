#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
すべてのコミットメッセージを修正
GitPythonを使用して直接コミットを作り直ム
"""

import sys
import os
import shutil

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

# 修正マッピング（SHA: 新しいメッセージ）
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

def fix_commits_with_rebase():
    """rebaseを使ってコミットメッセージを修正"""
    repo = git.Repo('.')
    
    print("=" * 60)
    print("コミットメッセージ修正")
    print("=" * 60)
    print(f"\n修正対象: {len(COMMIT_FIXES)}件\n")
    
    # rebase用のスクリプトを生成
    oldest_sha = list(COMMIT_FIXES.keys())[-1]  # 最も古いコミット
    
    # 親コミットを取得
    old_commit = repo.commit(oldest_sha)
    parent_sha = old_commit.parents[0].hexsha if old_commit.parents else None
    
    if not parent_sha:
        print("エラー: 親コミットが見つかりません")
        return
    
    print(f"最も古い修正対象コミット: {oldest_sha[:8]}")
    print(f"親コミット: {parent_sha[:8]}\n")
    
    # rebaseスクリプトを作成
    rebase_script_content = f"""# Rebase script for fixing commit messages
# Start rebase from parent of oldest garbled commit
"""
    
    # 逆順でrebaseコマンドを生成
    commits_list = list(COMMIT_FIXES.keys())
    commits_list.reverse()
    
    print("以下のコマンドを実行して、各コミットで 'reword' を指定してください:")
    print(f"\ngit rebase -i {parent_sha}")
    print("\n各コミットで:")
    for sha in commits_list:
        old_msg = repo.commit(sha).message.strip()
        new_msg = COMMIT_FIXES[sha]
        print(f"\n  [{sha[:8]}]")
        print(f"    現在: {old_msg[:60]}...")
        print(f"    修正: {new_msg}")
    
    print("\n" + "=" * 60)
    print("または、自動修正スクリプトを使用しますか？")
    print("(注意: これはgit rebaseを自動実行します)")
    
    # 実際にはgit rebaseを自動実行するのは難しいので、
    # filter-branchを使う方法に変更
    print("\n自動修正を行うには、git filter-branchを使用します:")
    print("(この処理は時間がかかる可能性があります)")
    
    # filter-branch用のスクリプトパスを絶対パスにする
    fix_script_path = os.path.abspath('scripts/fix_commits.py')
    
    cmd = f'$env:FILTER_BRANCH_SQUELCH_WARNING=1; git filter-branch -f --msg-filter "python {fix_script_path}" -- --all'
    
    print(f"\n実行コマンド:")
    print(cmd)
    
    response = input("\n実行しますか？ (yes/no): ")
    if response.lower() == 'yes':
        import subprocess
        env = os.environ.copy()
        env['FILTER_BRANCH_SQUELCH_WARNING'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            ['git', 'filter-branch', '-f', '--msg-filter', f'python {fix_script_path}', '--', '--all'],
            env=env,
            cwd='.',
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            print("\n✓ コミットメッセージの修正が完了しました！")
            print("\n次のステップ:")
            print("1. git log で確認")
            print("2. git push --force-with-lease origin main")
        else:
            print("\n✗ エラーが発生しました:")
            print(result.stderr)
    else:
        print("キャンセルされました。")

if __name__ == "__main__":
    fix_commits_with_rebase()


