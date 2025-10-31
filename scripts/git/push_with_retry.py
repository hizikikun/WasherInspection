#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Push with Retry
リトライ機能付きGitHubプッシュスクリプト
"""

import subprocess
import time
import sys
# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def push_with_retry(max_retries=5, delay=10):
    """リトライ機能付きでGitHubにプッシュ"""
    print("=" * 80)
    print("GitHubプッシュ（リトライ機能付き）")
    print("=" * 80)
    
    for attempt in range(max_retries):
        print(f"\nプッシュ試行 {attempt + 1}/{max_retries}")
        
        try:
            # プッシュ実行
            result = subprocess.run(
                ["git", "push", "origin", "main"], 
                capture_output=True, 
                text=True, 
                timeout=300  # 5分タイムアウト
            )
            
            if result.returncode == 0:
                print("✅ プッシュ成功!")
                print(result.stdout)
                return True
            else:
                print(f"❌ プッシュ失敗 (試行 {attempt + 1})")
                print(f"エラー: {result.stderr}")
                
                if attempt < max_retries - 1:
                    print(f"⏳ {delay}秒後に再試行...")
                    time.sleep(delay)
                    delay *= 2  # 指数バックオフ
                else:
                    print("❌ 最大試行回数に達しました")
                    return False
                    
        except subprocess.TimeoutExpired:
            print(f"⏰ タイムアウト (試行 {attempt + 1})")
            if attempt < max_retries - 1:
                print(f"⏳ {delay}秒後に再試行...")
                time.sleep(delay)
                delay *= 2
            else:
                print("❌ タイムアウトで最大試行回数に達しました")
                return False
                
        except Exception as e:
            print(f"❌ 予期しないエラー: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ {delay}秒後に再試行...")
                time.sleep(delay)
                delay *= 2
            else:
                print("❌ エラーで最大試行回数に達しました")
                return False
    
    return False

def check_git_status():
    """Git状況を確認"""
    print("Git状況を確認中...")
    
    try:
        # ステータス確認
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if result.returncode == 0:
            if result.stdout.strip():
                print("⚠️ 未コミットの変更があります:")
                print(result.stdout)
                return False
            else:
                print("✅ すべての変更がコミット済みです")
                return True
        else:
            print(f"❌ Git状況確認エラー: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Git状況確認エラー: {e}")
        return False

def main():
    """メイン実行関数"""
    print("GitHubプッシュ（リトライ機能付き）を開始します")
    
    # Git状況確認
    if not check_git_status():
        print("\n⚠️ 未コミットの変更があります。先にコミットしてください。")
        return
    
    # プッシュ実行
    success = push_with_retry()
    
    if success:
        print("\n🎉 GitHubプッシュが完了しました!")
        print("https://github.com/[ユーザー名]/WasherInspection で確認できまム")
    else:
        print("\n❌ GitHubプッシュに失敗しました")
        print("手動でプッシュしてください: git push origin main")
        print("または、GitHubのサーバー状況を確認してください")

if __name__ == "__main__":
    main()