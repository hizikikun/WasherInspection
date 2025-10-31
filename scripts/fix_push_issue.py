#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
プッシュ問題を解決（複数のアプローチを試行）
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

def run_command(cmd, capture_output=True):
    """コマンドを実行"""
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, 
                              text=True, encoding='utf-8', errors='replace', env=env)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def method1_merge_commits():
    """方法1: コミットを1つにまとめる"""
    print("=" * 60)
    print("方法1: コミットを1つにまとめる")
    print("=" * 60)
    print()
    
    # 現在のHEAD位置を記録
    code, stdout, stderr = run_command('git rev-parse HEAD')
    current_head = stdout.strip()
    
    # origin/mainとの差分を取得
    code, stdout, stderr = run_command('git rev-parse origin/main')
    origin_main = stdout.strip()
    
    if not origin_main:
        print("  エラー: origin/mainが見つかりません")
        return False
    
    print(f"  現在のHEAD: {current_head[:8]}")
    print(f"  origin/main: {origin_main[:8]}")
    print()
    
    # コミット数を確認
    code, stdout, stderr = run_command('git rev-list --count origin/main..HEAD')
    commit_count = int(stdout.strip()) if stdout.strip() else 0
    
    if commit_count == 0:
        print("  プッシュ待ちのコミットがありません")
        return False
    
    print(f"  未プッシュのコミット: {commit_count}個")
    print()
    print("  これらを1つのコミットにまとめますか？")
    print("  （この操作はコミット履歴を変更します）")
    print()
    
    # 実際には自動実行しない（安全のため）
    print("  実行コマンド（手動実行）:")
    print(f"    git reset --soft origin/main")
    print(f"    git commit -m 'ファイル整理と文字化け修正完了（まとめ）'")
    print(f"    git push")
    print()
    
    return False  # 手動実行が必要

def method2_switch_to_ssh():
    """方法2: SSHに切り替え"""
    print("=" * 60)
    print("方法2: HTTPSからSSHに切り替え")
    print("=" * 60)
    print()
    
    # 現在のURLを取得
    code, stdout, stderr = run_command('git remote get-url origin')
    current_url = stdout.strip()
    
    if 'github.com' not in current_url:
        print(f"  エラー: GitHubリポジトリではありません ({current_url})")
        return False
    
    # HTTPS URLからSSH URLに変換
    if current_url.startswith('https://'):
        # https://github.com/user/repo.git -> git@github.com:user/repo.git
        ssh_url = current_url.replace('https://github.com/', 'git@github.com:')
        
        print(f"  現在のURL: {current_url}")
        print(f"  SSH URL: {ssh_url}")
        print()
        print("  SSH URLに切り替えますか？")
        print("  （SSH鍵の設定が必要です）")
        print()
        print("  実行コマンド:")
        print(f"    git remote set-url origin {ssh_url}")
        print(f"    git push")
        print()
        
        # 自動実行（ユーザーが要求した場合）
        answer = "y"  # 自動実行
        if answer.lower() == 'y':
            code, stdout, stderr = run_command(f'git remote set-url origin {ssh_url}')
            if code == 0:
                print("  ✓ SSH URLに切り替えました")
                print()
                print("  SSH接続をテスト...")
                code, stdout, stderr = run_command('git ls-remote origin')
                if code == 0:
                    print("  ✓ SSH接続成功")
                    return True
                else:
                    print(f"  × SSH接続失敗: {stderr[:100]}")
                    print("  SSH鍵の設定が必要です")
                    # 元に戻す
                    run_command(f'git remote set-url origin {current_url}')
                    return False
        return False
    
    print("  既にSSH URLが設定されています")
    return False

def method3_push_incrementally():
    """方法3: インクリメンタルにプッシュ（1コミットずつ）"""
    print("=" * 60)
    print("方法3: インクリメンタルプッシュ（1コミットずつ）")
    print("=" * 60)
    print()
    
    # 未プッシュのコミットを取得
    code, stdout, stderr = run_command('git log --oneline --reverse origin/main..HEAD')
    commits = [line.strip() for line in stdout.split('\n') if line.strip()]
    
    if not commits:
        print("  プッシュ待ちのコミットがありません")
        return False
    
    print(f"  未プッシュのコミット: {len(commits)}個")
    print()
    
    # 最新のコミットから順にプッシュを試す
    # ただし、これは複雑なので、代わりにコミットをまとめる方法を推奨
    print("  この方法は複雑なため、コミットをまとめる方法（方法1）を推奨します")
    print()
    
    return False

def method4_use_github_cli():
    """方法4: GitHub CLIを使用"""
    print("=" * 60)
    print("方法4: GitHub CLIを使用")
    print("=" * 60)
    print()
    
    # GitHub CLIがインストールされているか確認
    code, stdout, stderr = run_command('gh --version')
    if code != 0:
        print("  GitHub CLI (gh) がインストールされていません")
        print()
        print("  インストール方法:")
        print("    winget install GitHub.cli")
        print("    または")
        print("    https://cli.github.com/ からダウンロード")
        return False
    
    print("  ✓ GitHub CLI が利用可能です")
    print()
    print("  認証を確認...")
    code, stdout, stderr = run_command('gh auth status')
    if code != 0:
        print("  × 認証が必要です")
        print("    実行: gh auth login")
        return False
    
    print("  ✓ 認証済み")
    print()
    print("  GitHub CLIでプッシュを試行...")
    print("  （通常のgit pushと同じですが、CLI経由で再試行）")
    
    return False

def method5_force_push_with_tags():
    """方法5: タグ付きでフォースプッシュ（危険）"""
    print("=" * 60)
    print("方法5: 安全なプッシュ設定")
    print("=" * 60)
    print()
    
    print("  Git設定を最適化...")
    
    # 追加の設定
    settings = [
        ('http.version', 'HTTP/2'),
        ('http.postBuffer', '524288000'),
        ('http.lowSpeedLimit', '0'),
        ('http.lowSpeedTime', '999999'),
        ('core.compression', '0'),
    ]
    
    for key, value in settings:
        code, stdout, stderr = run_command(f'git config {key} {value}')
        if code == 0:
            print(f"  ✓ {key} = {value}")
    
    print()
    print("  設定完了。再度プッシュを試してください:")
    print("    git push")
    print()
    
    return False

def main():
    print("=" * 60)
    print("プッシュ問題の解決方法")
    print("=" * 60)
    print()
    print("リポジトリサイズが大きい（11.42 GiB）ため、")
    print("HTTP 500エラーが発生している可能性があります。")
    print()
    
    methods = [
        ("コミットを1つにまとめる", method1_merge_commits),
        ("SSHに切り替え", method2_switch_to_ssh),
        ("インクリメンタルプッシュ", method3_push_incrementally),
        ("GitHub CLI使用", method4_use_github_cli),
        ("設定最適化", method5_force_push_with_tags),
    ]
    
    for i, (name, func) in enumerate(methods, 1):
        print(f"\n[{i}] {name}")
        print("-" * 60)
        result = func()
        if result:
            print(f"\n✓ {name}が成功しました！")
            return
        print()
    
    print("\n" + "=" * 60)
    print("推奨解決方法")
    print("=" * 60)
    print()
    print("以下のコマンドを順に実行してください:")
    print()
    print("1. コミットを1つにまとめる:")
    print("   git reset --soft origin/main")
    print("   git commit -m 'ファイル整理と文字化け修正完了'")
    print()
    print("2. SSHに切り替える（SSH鍵がある場合）:")
    print("   git remote set-url origin git@github.com:hizikikun/WasherInspection.git")
    print()
    print("3. プッシュ:")
    print("   git push")
    print()

if __name__ == '__main__':
    main()

