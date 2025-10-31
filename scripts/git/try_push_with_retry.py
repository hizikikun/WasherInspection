#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リトライ機能付きGitHubプッシュスクリプト
"""

import sys
import os
import time
import subprocess

if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def push_with_retry(max_retries=3, delay=5):
    """リトライ機能付きでプッシュ"""
    print("=" * 60)
    print("GitHubへのプッシュ（リトライ機能付き）")
    print("=" * 60)
    print()
    
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    for attempt in range(1, max_retries + 1):
        print(f"[試行 {attempt}/{max_retries}] プッシュ中...")
        
        try:
            result = subprocess.run(
                ['git', 'push', 'origin', 'main'],
                env=env,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=60
            )
            
            if result.returncode == 0:
                print("✓ プッシュ成功！")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                print(f"× プッシュ失敗 (コード: {result.returncode})")
                if result.stderr:
                    print(result.stderr)
                
                # "Everything up-to-date"の場合は成功とみなム
                if "Everything up-to-date" in result.stderr or "Everything up-to-date" in result.stdout:
                    print("→ リモートは既に最新の状態です")
                    return True
                
                # HTTP 500エラーの場合はリトライ
                if "HTTP 500" in result.stderr or "500" in result.stderr:
                    if attempt < max_retries:
                        print(f"→ GitHub側の一時的なエラーの可能性があります")
                        print(f"→ {delay}秒後に再試行します...")
                        time.sleep(delay)
                        continue
                    else:
                        print("→ 最大リトライ回数に達しました")
                        print("\n手動でプッシュしてください:")
                        print("  git push origin main")
                        return False
                else:
                    # その他のエラーは即座に返ム
                    print("\n手動でプッシュしてください:")
                    print("  git push origin main")
                    return False
                    
        except subprocess.TimeoutExpired:
            print(f"× タイムアウト（60秒）")
            if attempt < max_retries:
                print(f"→ {delay}秒後に再試行します...")
                time.sleep(delay)
            else:
                print("→ 最大リトライ回数に達しました")
                return False
        except Exception as e:
            print(f"× エラー: {e}")
            if attempt < max_retries:
                print(f"→ {delay}秒後に再試行します...")
                time.sleep(delay)
            else:
                return False
    
    return False

if __name__ == "__main__":
    success = push_with_retry()
    sys.exit(0 if success else 1)

