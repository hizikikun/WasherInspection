#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整理済みファイルをGitHubにアップロード
"""

import subprocess
import sys
from pathlib import Path

# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def run_command(cmd, check=True):
    """コマンドを実行"""
    try:
        print(f"実行中: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if check and result.returncode != 0:
            print(f"エラー: 終了コード {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"エラー: {e}")
        return False

def main():
    print("=" * 60)
    print("GitHubへのアップロード")
    print("=" * 60)
    print()
    
    # Git status確認
    print("[1/4] Git状態確認...")
    if not run_command(['git', 'status', '--short'], check=False):
        print("Gitリポジトリが見つかりません")
        return
    
    # 変更を追加
    print("\n[2/4] 変更をステージング...")
    if not run_command(['git', 'add', '.']):
        print("エラー: git add に失敗しました")
        return
    
    # コミット
    print("\n[3/4] コミット作成...")
    commit_message = "ファイル整理と文字化け修正完了"
    if not run_command(['git', 'commit', '-m', commit_message]):
        print("エラー: git commit に失敗しました")
        return
    
    # プッシュ
    print("\n[4/4] リモートにプッシュ...")
    if not run_command(['git', 'push']):
        print("エラー: git push に失敗しました")
        print("手動で実行してください: git push")
        return
    
    print()
    print("=" * 60)
    print("アップロード完了")
    print("=" * 60)

if __name__ == '__main__':
    main()

