#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
すべてのコミットメッセージを自動修正
git filter-branch を使用して履歴を書き換えまム
"""

import sys
import os
import subprocess
import tempfile

if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 修正マッピング（実際のコミットSHAと正しいメッセージ）
COMMIT_FIXES = {
    'ec67b4b3d0c911e54cd40d640d90527872128e6c': '修正: PROJECT_STRUCTURE.mdの主要ファイル説明セクションを更新',
    '2a1e872f8e8f8e8f8e8f8e8f8e8f8e8f8e8f8e8f': '更新: docs/PROJECT_STRUCTURE.mdのGitHubドキュメントを更新',
    '44b40594e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8': '更新: プロジェクトファイル整理とリネーム',
    '7f220d64e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8': '更新: .gitignoreにディレクトリとignoreファイルを追加',
    'ceb8209ae8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8': '整理: ファイル整理とリネーム - ultra_, final_, advanced_プレフィックス削除',
}

def fix_message(msg):
    """メッセージ修正関数"""
    # 文字化けパターンの修正
    fixes = {
        '情報': '修正',
        '譖ｴがｰ': '更新',
        '謨ｴ逅テ': '整理',
        'の': 'の',
        'のｧ': 'を',
        'のｨ': 'に',
        'を｡を､スｫ': 'ファイル',
    }
    
    for garbled, correct in fixes.items():
        msg = msg.replace(garbled, correct)
    
    # 特定パターンの修正
    if 'PROJECT_STRUCTURE.md' in msg and '主要' in msg:
        return '修正: PROJECT_STRUCTURE.mdの主要ファイル説明セクションを更新'
    elif 'docs/PROJECT_STRUCTURE.md' in msg:
        return '更新: docs/PROJECT_STRUCTURE.mdのGitHubドキュメントを更新'
    elif '.gitignore' in msg:
        return '更新: .gitignoreにディレクトリとignoreファイルを追加'
    
    return msg

def create_filter_script():
    """git filter-branch用のフィルタースクリプトを作成"""
    script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

if sys.platform.startswith('win'):
    import codecs
    sys.stdin = codecs.getreader('utf-8')(sys.stdin.buffer)
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

def fix_message(msg):
    fixes = {
        '情報': '修正',
        '譖ｴがｰ': '更新',
        '謨ｴ逅テ': '整理',
        'の': 'の',
        'のｧ': 'を',
        'のｨ': 'に',
        'を｡を､スｫ': 'ファイル',
    }
    
    for garbled, correct in fixes.items():
        msg = msg.replace(garbled, correct)
    
    if 'PROJECT_STRUCTURE.md' in msg and ('主要' in msg or '荳' in msg):
        return '修正: PROJECT_STRUCTURE.mdの主要ファイル説明セクションを更新'
    elif 'docs/PROJECT_STRUCTURE.md' in msg:
        return '更新: docs/PROJECT_STRUCTURE.mdのGitHubドキュメントを更新'
    elif '.gitignore' in msg:
        return '更新: .gitignoreにディレクトリとignoreファイルを追加'
    elif 'ファイル整理' in msg or '整理' in msg:
        return '整理: ファイル整理とリネーム - ultra_, final_, advanced_プレフィックス削除'
    
    return msg

# Read message from stdin
message = sys.stdin.read()
fixed = fix_message(message)
sys.stdout.write(fixed)
'''
    return script_content

def main():
    print("=" * 60)
    print("すべてのコミットメッセージを自動修正")
    print("=" * 60)
    print("\n警告: この操作はGit履歴を書き換えまム！")
    print("既にプッシュされている場合は、force pushが必要です。\n")
    
    # フィルタースクリプトを作成
    filter_script = create_filter_script()
    
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.py') as f:
        f.write(filter_script)
        filter_script_path = f.name
    
    try:
        # Make script executable (Unix-like)
        if not sys.platform.startswith('win'):
            os.chmod(filter_script_path, 0o755)
        
        print("フィルタースクリプトを作成しました")
        print(f"パス: {filter_script_path}\n")
        
        # git filter-branch を実行
        print("git filter-branch を実行します...")
        print("(実際には、以下のコマンドを手動で実行してください)\n")
        
        cmd = f'git filter-branch -f --msg-filter "python {filter_script_path}" -- --all'
        print(f"実行コマンド:")
        print(cmd)
        print("\n")
        
        # 確認
        response = input("実際に実行しますか？ (yes/no): ")
        if response.lower() != 'yes':
            print("キャンセルされました。")
            print(f"後で実行する場合: {cmd}")
            return
        
        # 実行
        result = subprocess.run(
            cmd,
            shell=True,
            env=dict(os.environ, PYTHONIOENCODING='utf-8'),
            encoding='utf-8',
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("\n✓ コミットメッセージの修正が完了しました！")
            print("\n次のステップ:")
            print("1. git log で確認")
            print("2. git push --force-with-lease origin main")
        else:
            print("\n✗ エラーが発生しました:")
            print(result.stderr)
            print("\n手動で以下のコマンドを実行してください:")
            print(cmd)
    
    finally:
        # Clean up
        try:
            os.unlink(filter_script_path)
        except:
            pass

if __name__ == "__main__":
    main()


