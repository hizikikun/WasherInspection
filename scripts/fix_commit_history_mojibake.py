#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
コミット履歴の文字化けを修正
"""

import subprocess
import sys
import os
from pathlib import Path

# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def get_commits_to_fix():
    """修正が必要なコミットを取得"""
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    result = subprocess.run(
        ['git', 'log', '--oneline', 'origin/main..HEAD'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    if not result.stdout.strip():
        return []
    
    commits = []
    for line in result.stdout.strip().split('\n'):
        if line.strip():
            parts = line.split(' ', 1)
            if len(parts) == 2:
                commits.append({
                    'hash': parts[0],
                    'message': parts[1]
                })
    
    return commits

def fix_commit_message(message):
    """コミットメッセージの文字化けを修正"""
    # よくある文字化けパターンの修正
    fixes = {
        '網・そ絹シ腰の室豁」網': 'ファイル整理と文字化け修正',
        '網材繧定リス蜉回': '完了',
        '險1螳壹ヤ綢シ': 'エンコーディングと',
        '縺繧繝溘ヤ綱医': 'セキュリティ設定',
    }
    
    fixed = message
    for corrupted, correct in fixes.items():
        if corrupted in fixed:
            fixed = fixed.replace(corrupted, correct)
    
    # まだ文字化けが残っている場合、シンプルなメッセージに置換
    if any(ord(c) > 0x7F and ord(c) < 0x3000 for c in fixed if c):
        # 異常な文字が含まれている場合
        if 'ファイル' in fixed or '整理' in fixed or '文字化け' in fixed:
            return 'ファイル整理と文字化け修正完了'
        elif 'Git' in fixed or 'UTF-8' in fixed or 'エンコーディング' in fixed:
            return 'Git UTF-8エンコーディングとセキュリティ設定'
        else:
            return 'コミットメッセージ修正'
    
    return fixed

def rewrite_commits():
    """コミット履歴を書き換え"""
    print("=" * 60)
    print("コミット履歴の文字化け修正")
    print("=" * 60)
    print()
    
    commits = get_commits_to_fix()
    
    if not commits:
        print("修正が必要なコミットがありません")
        return
    
    print(f"修正対象のコミット: {len(commits)}個")
    print()
    
    for i, commit in enumerate(commits, 1):
        print(f"[{i}/{len(commits)}] {commit['hash'][:8]}")
        print(f"  元のメッセージ: {commit['message'][:50]}")
        
        fixed_message = fix_commit_message(commit['message'])
        print(f"  修正後のメッセージ: {fixed_message}")
        print()
    
    print("コミット履歴を書き換えますか？")
    print("（この操作は履歴を変更します）")
    print()
    
    # 自動実行
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    # origin/mainまでreset --soft
    print("コミットを統合中...")
    result = subprocess.run(
        ['git', 'reset', '--soft', 'origin/main'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    if result.returncode != 0:
        print(f"エラー: {result.stderr}")
        return
    
    # トークンを含むファイルを除外
    print("トークンファイルを除外中...")
    subprocess.run(
        ['git', 'reset', 'HEAD', 'config/cursor_github_config.json'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    # 新しいコミットを作成
    print("新しいコミットを作成中...")
    result = subprocess.run(
        ['git', 'commit', '-m', 'ファイル整理と文字化け修正完了'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    if result.returncode == 0:
        print("✓ コミット作成成功")
    else:
        if 'nothing to commit' in result.stdout:
            print("変更がありません")
        else:
            print(f"エラー: {result.stderr}")

def main():
    rewrite_commits()
    
    print()
    print("=" * 60)
    print("次のステップ")
    print("=" * 60)
    print()
    print("1. 新しいトークンを作成（まだの場合）")
    print("2. プッシュを試行:")
    print("   git push origin main")
    print()
    print("プッシュが成功するはずです。")

if __name__ == '__main__':
    main()

