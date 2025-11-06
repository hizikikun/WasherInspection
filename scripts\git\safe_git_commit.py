#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全なGitコミットスクリプト - UTF-8エンコーディングを保証
GitPythonを使用して直接コミットを作成
"""

import sys
import os

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

def safe_git_commit(message, files=None):
    """
    UTF-8で安全にGitコミットを実行（GitPython使用）
    """
    try:
        # Open repository
        repo = git.Repo('.')
        
        # Add files if specified
        if files:
            for file in files:
                repo.index.add([file])
        else:
            repo.index.add(['*'])
        
        # Create commit directly using GitPython (UTF-8 guaranteed)
        # GitPython handles UTF-8 encoding internally
        commit = repo.index.commit(message)
        
        print(f"コミット成功: {message}")
        print(f"コミットSHA: {commit.hexsha[:8]}")
        return True
        
    except Exception as e:
        print(f"コミット失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python safe_git_commit.py <コミットメッセージ> [ファイル1] [ファイル2] ...")
        sys.exit(1)
    
    message = sys.argv[1]
    files = sys.argv[2:] if len(sys.argv) > 2 else None
    
    success = safe_git_commit(message, files)
    sys.exit(0 if success else 1)

