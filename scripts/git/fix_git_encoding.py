#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gitエンコーディング設定スクリプト
このスクリプトを実行すると、GitのUTF-8設定が正しく行われまム
"""

import subprocess
import sys
import os

if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def setup_git_encoding():
    """Set up Git encoding for UTF-8"""
    print("Git UTF-8エンコーディング設定中...")
    
    configs = [
        ('--global', 'i18n.commitencoding', 'utf-8'),
        ('--global', 'i18n.logoutputencoding', 'utf-8'),
        ('--global', 'core.quotepath', 'false'),
        ('--local', 'i18n.commitencoding', 'utf-8'),
        ('--local', 'i18n.logoutputencoding', 'utf-8'),
        ('--local', 'core.quotepath', 'false'),
    ]
    
    for scope, key, value in configs:
        try:
            subprocess.run(
                ['git', 'config', scope, key, value],
                check=True,
                capture_output=True
            )
            print(f"  ✓ {scope} {key} = {value}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to set {scope} {key}: {e}")
    
    print("\n設定完了!")
    print("\n今後のコミットメッセージはUTF-8で正しく保存されまム。")
    print("注意: 過去のコミットメッセージは修正されません。")

if __name__ == "__main__":
    setup_git_encoding()


