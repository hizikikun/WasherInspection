#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
プッシュ問題の診断と解決
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

def diagnose_push_issue():
    """プッシュ問題を診断"""
    print("=" * 60)
    print("プッシュ問題の診断")
    print("=" * 60)
    print()
    
    # 1. リモート情報を確認
    print("[1] リモートリポジトリ情報:")
    code, stdout, stderr = run_command('git remote -v')
    print(stdout)
    print()
    
    # 2. Git設定を確認
    print("[2] Git設定:")
    code, stdout, stderr = run_command('git config --list')
    config_lines = stdout.split('\n')
    relevant_configs = [line for line in config_lines if any(keyword in line.lower() 
                       for keyword in ['remote', 'url', 'credential', 'http', 'push'])]
    for config in relevant_configs[:10]:
        print(f"  {config}")
    print()
    
    # 3. ブランチ情報
    print("[3] ブランチ情報:")
    code, stdout, stderr = run_command('git branch -vv')
    print(stdout)
    print()
    
    # 4. コミット数
    print("[4] プッシュ待ちのコミット数:")
    code, stdout, stderr = run_command('git log --oneline origin/main..HEAD')
    commit_count = len([line for line in stdout.split('\n') if line.strip()])
    print(f"  未プッシュのコミット: {commit_count}個")
    if stdout.strip():
        print("  コミット一覧:")
        for line in stdout.split('\n')[:5]:
            if line.strip():
                print(f"    {line}")
    print()
    
    # 5. リポジトリサイズ
    print("[5] リポジトリサイズ:")
    code, stdout, stderr = run_command('git count-objects -vH')
    print(stdout)
    print()
    
    # 6. ネットワークテスト
    print("[6] GitHub接続テスト:")
    remote_url = None
    code, stdout, stderr = run_command('git remote get-url origin')
    remote_url = stdout.strip()
    if remote_url:
        print(f"  リモートURL: {remote_url}")
        if 'github.com' in remote_url:
            print("  GitHub URL確認: OK")
        else:
            print(f"  警告: GitHub以外のリポジトリ ({remote_url})")
    print()
    
    return remote_url, commit_count

def try_alternative_push_methods(remote_url, commit_count):
    """代替プッシュ方法を試す"""
    print("=" * 60)
    print("代替プッシュ方法を試行")
    print("=" * 60)
    print()
    
    methods = []
    
    # 方法1: HTTPバッファサイズを増やす
    if commit_count > 5:
        print("[方法1] HTTPバッファサイズを増やす...")
        code, stdout, stderr = run_command('git config http.postBuffer 524288000')
        if code == 0:
            methods.append(('http.postBuffer', '設定完了'))
            print("  ✓ HTTPバッファサイズを500MBに設定")
        print()
    
    # 方法2: タイムアウトを延長
    print("[方法2] タイムアウトを延長...")
    code, stdout, stderr = run_command('git config http.lowSpeedLimit 0')
    code, stdout, stderr = run_command('git config http.lowSpeedTime 999999')
    if code == 0:
        methods.append(('timeout', '設定完了'))
        print("  ✓ タイムアウトを延長")
    print()
    
    # 方法3: 圧縮レベルを下げる（転送速度優先）
    print("[方法3] 圧縮設定を調整...")
    code, stdout, stderr = run_command('git config core.compression 0')
    if code == 0:
        methods.append(('compression', '設定完了'))
        print("  ✓ 圧縮を無効化（転送速度優先）")
    print()
    
    # 方法4: 小さなバッチでプッシュ（最新のコミットのみ）
    if commit_count > 1:
        print(f"[方法4] 最新の1コミットのみプッシュを試行...")
        print("  （大きなコミットを分割）")
        methods.append(('batch_push', '試行可能'))
    print()
    
    return methods

def try_push_with_retry():
    """リトライ付きでプッシュを試行"""
    print("=" * 60)
    print("プッシュ試行（リトライ付き）")
    print("=" * 60)
    print()
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        print(f"試行 {attempt}/{max_retries}...")
        code, stdout, stderr = run_command('git push', capture_output=False)
        
        if code == 0:
            print("\n✓ プッシュ成功！")
            return True
        
        print(f"\n× プッシュ失敗 (試行 {attempt})")
        if stderr:
            print(f"  エラー: {stderr[:200]}")
        
        if attempt < max_retries:
            print(f"  5秒後に再試行...")
            import time
            time.sleep(5)
    
    return False

def try_batch_push():
    """バッチでプッシュ（最新のコミットから順に）"""
    print("\n" + "=" * 60)
    print("バッチプッシュ（最新のコミットから順に）")
    print("=" * 60)
    print()
    
    # 最新のコミットを取得
    code, stdout, stderr = run_command('git log --oneline origin/main..HEAD')
    commits = [line.strip() for line in stdout.split('\n') if line.strip()]
    
    if not commits:
        print("プッシュ待ちのコミットがありません")
        return False
    
    print(f"プッシュ待ちのコミット数: {len(commits)}")
    print()
    
    # 最新の1つだけプッシュを試す
    print("最新のコミットのみプッシュを試行...")
    code, stdout, stderr = run_command('git push origin HEAD~{0}:main'.format(len(commits)-1))
    
    if code == 0:
        print("✓ 部分プッシュ成功")
        return True
    else:
        print("× 部分プッシュ失敗")
        print(f"  エラー: {stderr[:200]}")
        return False

def main():
    # 診断
    remote_url, commit_count = diagnose_push_issue()
    
    # 代替方法を試す
    methods = try_alternative_push_methods(remote_url, commit_count)
    
    # プッシュ試行
    success = try_push_with_retry()
    
    if not success:
        print("\n" + "=" * 60)
        print("プッシュ失敗 - 代替案")
        print("=" * 60)
        print()
        print("以下の方法を試してください:")
        print()
        print("1. GitHub CLIを使用:")
        print("   gh auth login")
        print("   gh repo sync")
        print()
        print("2. SSHでプッシュ（HTTPSからSSHに変更）:")
        print("   git remote set-url origin git@github.com:USER/REPO.git")
        print("   git push")
        print()
        print("3. 手動でブラウザからプッシュ")
        print()
        print("4. 小さなコミットに分割:")
        print("   git reset --soft HEAD~N  # Nはコミット数")
        print("   git commit -m '...'")
        print("   git push")

if __name__ == '__main__':
    main()

