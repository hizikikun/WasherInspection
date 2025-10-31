#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
コミットメッセージの文字化けを修正してプッシュ
"""

import subprocess
import sys
import os

# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def main():
    print("=" * 60)
    print("コミットメッセージ文字化け修正とプッシュ")
    print("=" * 60)
    print()
    
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    # 現在の状態を確認
    print("1. 現在の状態を確認...")
    result = subprocess.run(
        ['git', 'status', '--short'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    if result.stdout.strip():
        print("   ステージング済みの変更があります")
    else:
        print("   ステージング済みの変更はありません")
    
    # origin/mainとの差分を確認
    result = subprocess.run(
        ['git', 'log', '--oneline', 'origin/main..HEAD'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    if result.stdout.strip():
        print(f"   ローカルに {len(result.stdout.strip().split(chr(10)))} 個の未プッシュコミットがあります")
        print()
        print("2. origin/mainまでリセット（変更は保持）...")
        result = subprocess.run(
            ['git', 'reset', '--soft', 'origin/main'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )
        
        if result.returncode != 0:
            print(f"   エラー: {result.stderr}")
            return
        
        print("   ✓ リセット完了")
    else:
        print("   未プッシュのコミットはありません")
        print()
    
    # トークンファイルを除外
    print()
    print("3. トークンファイルを除外...")
    subprocess.run(
        ['git', 'reset', 'HEAD', 'config/cursor_github_config.json'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    print("   ✓ 除外完了")
    
    # すべての変更をステージング
    print()
    print("4. すべての変更をステージング...")
    result = subprocess.run(
        ['git', 'add', '.'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    if result.returncode != 0:
        print(f"   エラー: {result.stderr}")
        return
    
    print("   ✓ ステージング完了")
    
    # 新しいコミットを作成（UTF-8で正しいメッセージ）
    print()
    print("5. 新しいコミットを作成（文字化けなし）...")
    commit_msg = "ファイル整理と文字化け修正完了"
    result = subprocess.run(
        ['git', 'commit', '-m', commit_msg],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    if result.returncode != 0:
        if 'nothing to commit' in result.stdout:
            print("   変更がありません（既にコミット済み）")
        else:
            print(f"   エラー: {result.stderr}")
            return
    else:
        print(f"   ✓ コミット作成: {commit_msg}")
    
    # プッシュ
    print()
    print("6. プッシュ（force-with-lease）...")
    print("   注意: 既にリモートに文字化けコミットがある場合、force pushが必要です")
    print()
    
    result = subprocess.run(
        ['git', 'push', 'origin', 'main'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    if result.returncode == 0:
        print("   ✓ プッシュ成功")
    else:
        if 'Updates were rejected' in result.stderr or 'force' in result.stderr.lower():
            print("   force pushが必要です")
            print()
            print("   次のコマンドを実行してください:")
            print("   git push --force-with-lease origin main")
        else:
            print(f"   エラー: {result.stderr}")
    
    print()
    print("=" * 60)
    print("完了")
    print("=" * 60)

if __name__ == '__main__':
    main()

