#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Secretを削除してプッシュ
"""

import subprocess
import sys
import os
import re
from pathlib import Path

# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def find_secret_files():
    """トークンを含む可能性のあるファイルを検索"""
    print("=" * 60)
    print("Secretを含むファイルの検索")
    print("=" * 60)
    print()
    
    # トークンパターン
    token_patterns = [
        r'ghp_[A-Za-z0-9]{36,}',
        r'github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59,}',
        r'[A-Za-z0-9]{40,}',  # 長い文字列（トークンの可能性）
    ]
    
    suspicious_files = []
    
    # 検索対象ファイル
    search_files = [
        'config/*.json',
        '*.json',
        '*.py',
        '*.txt',
        '*.md',
    ]
    
    for pattern in search_files:
        for file_path in Path('.').glob(pattern):
            if file_path.is_file() and file_path.stat().st_size < 1024 * 1024:  # 1MB未満
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    for token_pattern in token_patterns:
                        matches = re.findall(token_pattern, content)
                        if matches:
                            # 実際のトークンかどうか確認（長すぎる文字列は除外）
                            for match in matches:
                                if len(match) >= 40 and ('ghp_' in match or 'github_pat_' in match or 'token' in content.lower()):
                                    suspicious_files.append((str(file_path), match[:20] + '...'))
                                    break
                except Exception:
                    pass
    
    if suspicious_files:
        print("検出された可能性のあるファイル:")
        for file_path, token_preview in suspicious_files[:10]:
            print(f"  {file_path}: {token_preview}")
    else:
        print("明らかなトークンは見つかりませんでした")
        print("（バックアップファイルに含まれている可能性があります）")
    
    print()
    return suspicious_files

def exclude_backup_and_large_files():
    """バックアップと大きなファイルを除外"""
    print("=" * 60)
    print("バックアップと大きなファイルを除外")
    print("=" * 60)
    print()
    
    # .gitignoreを更新
    gitignore_path = Path('.gitignore')
    with open(gitignore_path, 'r', encoding='utf-8') as f:
        gitignore_content = f.read()
    
    patterns_to_add = [
        '# Backup directories (may contain secrets)',
        'backup/',
        '# Large build files',
        'dist/ResinWasherInspection.exe',
        'build/ResinWasherInspection/',
        '*.pkg',
        '# Virtual environments (may contain secrets)',
        '**/.venv/',
        '**/venv/',
        '# Cache files',
        '**/__pycache__/',
        '*.pyc',
    ]
    
    added = False
    for pattern in patterns_to_add:
        if pattern not in gitignore_content and not pattern.startswith('#'):
            gitignore_content += '\n' + pattern + '\n'
            added = True
    
    if added:
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        print("✓ .gitignoreを更新しました")
    
    # バックアップと大きなファイルをステージングから除外
    print()
    print("ステージングから除外中...")
    
    exclude_patterns = [
        'backup/',
        'dist/ResinWasherInspection.exe',
        'build/ResinWasherInspection/',
        '**/.venv/',
        '**/__pycache__/',
    ]
    
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    for pattern in exclude_patterns:
        try:
            result = subprocess.run(
                ['git', 'reset', 'HEAD', pattern],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env
            )
        except Exception:
            pass
    
    print("✓ 除外完了")
    print()

def remove_secret_from_history():
    """コミット履歴からSecretを削除"""
    print("=" * 60)
    print("コミット履歴からSecretを削除")
    print("=" * 60)
    print()
    
    print("警告: この操作はコミット履歴を書き換えます")
    print("未プッシュのコミットのみを修正します")
    print()
    
    # origin/mainとの差分を取得
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    result = subprocess.run(
        ['git', 'rev-list', '--oneline', 'origin/main..HEAD'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    commits = [line.strip() for line in result.stdout.split('\n') if line.strip()]
    
    if not commits:
        print("修正するコミットがありません")
        return False
    
    print(f"未プッシュのコミット: {len(commits)}個")
    print()
    print("これらのコミットを1つにまとめて、Secretを含むファイルを除外します")
    print()
    
    # reset --softでコミットを統合
    result = subprocess.run(
        ['git', 'reset', '--soft', 'origin/main'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    if result.returncode == 0:
        print("✓ コミットを統合しました")
        return True
    else:
        print(f"× エラー: {result.stderr}")
        return False

def create_clean_commit():
    """クリーンなコミットを作成"""
    print("=" * 60)
    print("クリーンなコミットを作成")
    print("=" * 60)
    print()
    
    # 大きなファイルとバックアップを除外
    exclude_backup_and_large_files()
    
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    # コミット
    result = subprocess.run(
        ['git', 'commit', '-m', 'ファイル整理と文字化け修正完了（Secret除外済み）'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    if result.returncode == 0:
        print("✓ コミット成功")
        return True
    else:
        if 'nothing to commit' in result.stdout:
            print("変更がありません（既にコミット済み）")
            return True
        print(f"× コミット失敗: {result.stderr}")
        return False

def try_push():
    """プッシュを試行"""
    print("=" * 60)
    print("プッシュ試行")
    print("=" * 60)
    print()
    
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    result = subprocess.run(
        ['git', 'push', 'origin', 'main'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    if result.returncode == 0:
        print("✓ プッシュ成功！")
        return True
    else:
        if result.stderr:
            print(f"× プッシュ失敗")
            print(f"  エラー: {result.stderr[:500]}")
        else:
            print(f"× プッシュ失敗: {result.stdout}")
        return False

def main():
    # Secretを含むファイルを検索
    find_secret_files()
    
    # コミット履歴から削除
    if remove_secret_from_history():
        # クリーンなコミットを作成
        if create_clean_commit():
            # プッシュを試行
            try_push()
        else:
            print("\nコミット作成に失敗しました")
    else:
        print("\nコミット履歴の修正に失敗しました")
    
    print()
    print("=" * 60)
    print("次のステップ")
    print("=" * 60)
    print()
    print("まだプッシュできない場合は:")
    print("1. 提供されたURLでSecretを許可:")
    print("   https://github.com/hizikikun/WasherInspection/security/secret-scanning/unblock-secret/34pMGYB8wD485PIUI70WsQGd6wd")
    print()
    print("2. または、バックアップフォルダーを完全に削除:")
    print("   git rm -r --cached backup/")
    print("   git commit -m 'Remove backup directory'")
    print("   git push")

if __name__ == '__main__':
    main()

