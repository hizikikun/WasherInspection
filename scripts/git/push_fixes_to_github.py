#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文字化け修正をGitHubにプッシュするスクリプト
"""

import sys
import os
import subprocess

if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def push_fixes():
    """文字化け修正をコミットしてプッシュ"""
    print("=" * 60)
    print("GitHubへの文字化け修正のプッシュ")
    print("=" * 60)
    print()
    
    # 現在の変更をステージング
    print("[1] 変更をステージング...")
    try:
        result = subprocess.run(
            ['git', 'add', '-A'],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode != 0:
            print(f"エラー: {result.stderr}")
            return False
        print("  完了")
    except Exception as e:
        print(f"エラー: {e}")
        return False
    
    # コミットメッセージ
    commit_message = """fix: ワークスペースとGitHubリポジトリの文字化けを修正

- 文字化けファイル名テディレクトリ名を修正テ削除
- GitHub APIへのリクエストをUTF-8対応に修正
- コミットメッセージ生成処理を改善
- 文字化けチェックスクリプトを追加
- ドキュメントを追加（MOJIBAKE_FIX_SUMMARY.md, GITHUB_ENCODING_FIX.md）

修正内容:
- buildディレクトリ内の文字化けファイルを削除
- github_tools/github_autocommit.py をUTF-8対応に修正
- github_tools/cursor_integration.py をUTF-8対応に修正
- github_tools/auto_sync.py をUTF-8対応に修正
"""
    
    # コミット作成
    print("\n[2] コミットを作成...")
    try:
        # GitPythonを使用してUTF-8保証でコミット
        import git
        repo = git.Repo('.')
        
        # ステージング済みファイルを確認
        if len(repo.index.diff('HEAD')) == 0 and len(repo.untracked_files) == 0:
            print("  コミットする変更がありません")
            return True
        
        # コミット作成
        commit = repo.index.commit(commit_message)
        print(f"  コミット成功: {commit.hexsha[:7]}")
        print(f"  メッセージ: {commit_message.split(chr(10))[0]}")
    except ImportError:
        # GitPythonがない場合は通常のgitコマンドを使用
        print("  GitPythonを使用できません。通常のgitコマンドを使用します...")
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['GIT_COMMITTER_NAME'] = 'WasherInspection Bot'
        env['GIT_COMMITTER_EMAIL'] = 'bot@washerinspection.local'
        
        # コミットメッセージを一時ファイルに書き込み
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write(commit_message)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ['git', 'commit', '-F', temp_file],
                env=env,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            if result.returncode != 0:
                print(f"エラー: {result.stderr}")
                return False
            print("  コミット成功")
        finally:
            os.unlink(temp_file)
    except Exception as e:
        print(f"エラー: {e}")
        return False
    
    # プッシュ
    print("\n[3] GitHubにプッシュ...")
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            ['git', 'push', 'origin', 'main'],
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode == 0:
            print("  プッシュ成功")
            return True
        else:
            print(f"  エラー: {result.stderr}")
            print("  手動でプッシュしてください: git push origin main")
            return False
    except Exception as e:
        print(f"エラー: {e}")
        return False

if __name__ == "__main__":
    success = push_fixes()
    sys.exit(0 if success else 1)

