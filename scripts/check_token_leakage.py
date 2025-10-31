#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
トークン漏洩の確認と緊急対応スクリプト
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

def check_commit_history():
    """コミット履歴にトークンが含まれているか確認"""
    print("=" * 60)
    print("コミット履歴の確認")
    print("=" * 60)
    print()
    
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    # config/cursor_github_config.json の履歴を確認
    result = subprocess.run(
        ['git', 'log', '--all', '--full-history', '--pretty=format:%H', '--', 'config/cursor_github_config.json'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    if result.stdout.strip():
        commits = result.stdout.strip().split('\n')
        print(f"⚠️  警告: {len(commits)}個のコミットにトークンが含まれています")
        print()
        print("含まれているコミット:")
        for commit in commits[:5]:  # 最初の5つを表示
            print(f"  {commit[:8]}")
        if len(commits) > 5:
            print(f"  ... 他 {len(commits) - 5}個")
        print()
        return True
    else:
        print("✓ コミット履歴には見つかりませんでした")
        print()
        return False

def check_remote_status():
    """リモートリポジトリの状態を確認"""
    print("=" * 60)
    print("リモートリポジトリの状態確認")
    print("=" * 60)
    print()
    
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    # プッシュ状態を確認
    result = subprocess.run(
        ['git', 'status'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    if 'ahead' in result.stdout:
        print("⚠️  警告: 未プッシュのコミットがあります")
        print("  これらにもトークンが含まれている可能性があります")
    elif 'up to date' in result.stdout.lower():
        print("✓ リモートと同期されています")
    else:
        print(result.stdout)
    
    print()

def show_emergency_steps():
    """緊急対応手順を表示"""
    print("=" * 60)
    print("🚨 緊急対応手順")
    print("=" * 60)
    print()
    print("1. すぐにトークンを無効化:")
    print("   https://github.com/settings/tokens")
    print("   該当トークンを削除してください")
    print()
    print("2. 新しいトークンを作成（必要に応じて）")
    print()
    print("3. ローカル設定を更新:")
    print("   config/cursor_github_config.json を編集")
    print("   github_token を新しいトークンに変更")
    print()
    print("4. コミット履歴から削除を検討:")
    print("   - BFG Repo-Cleaner を使用")
    print("   - または git filter-branch を使用")
    print()
    print("5. リポジトリの公開設定を確認:")
    print("   https://github.com/hizikikun/WasherInspection/settings")
    print("   できればPrivateに設定")
    print()

def main():
    print("=" * 60)
    print("トークン漏洩の確認")
    print("=" * 60)
    print()
    
    # コミット履歴を確認
    has_leakage = check_commit_history()
    
    # リモート状態を確認
    check_remote_status()
    
    # 緊急対応手順を表示
    if has_leakage:
        show_emergency_steps()
        print()
        print("=" * 60)
        print("⚠️  重要: すぐにトークンを無効化してください！")
        print("=" * 60)
    else:
        print("=" * 60)
        print("✓ 現在のローカルリポジトリには問題は見つかりませんでした")
        print("=" * 60)
        print()
        print("ただし、既にプッシュされている場合は、")
        print("リモートリポジトリにトークンが含まれている可能性があります。")
        print()
        print("念のため、トークンの無効化を推奨します。")

if __name__ == '__main__':
    main()

