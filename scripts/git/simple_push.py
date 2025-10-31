#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Push
シンプルなGitHubプッシュスクリプト
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

def push_to_github():
    """GitHubにプッシュ"""
    print("GitHubにプッシュ中...")
    
    max_retries = 3
    delay = 10
    
    for attempt in range(max_retries):
        print(f"試行 {attempt + 1}/{max_retries}")
        
        try:
            result = subprocess.run(
                ["git", "push", "origin", "main"], 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            
            if result.returncode == 0:
                print("プッシュ成功!")
                return True
            else:
                print(f"プッシュ失敗: {result.stderr}")
                if attempt < max_retries - 1:
                    print(f"{delay}秒後に再試行...")
                    time.sleep(delay)
                    delay *= 2
                
        except Exception as e:
            print(f"エラー: {e}")
            if attempt < max_retries - 1:
                print(f"{delay}秒後に再試行...")
                time.sleep(delay)
                delay *= 2
    
    return False

def main():
    """メイン実行関数"""
    print("GitHubプッシュを開始します")
    
    success = push_to_github()
    
    if success:
        print("GitHubプッシュが完了しました!")
    else:
        print("GitHubプッシュに失敗しました")
        print("手動でプッシュしてください: git push origin main")

if __name__ == "__main__":
    main()
