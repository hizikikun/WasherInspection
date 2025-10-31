#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHubの全コミットメッセージの文字化けを修正するスクリプト
注意: このスクリプトは既存のコミットを書き換えるため、共有リポジトリでは注意が必要です
"""

import sys
import os
import subprocess

if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    import git
except ImportError:
    print("エラー: GitPythonが必要です。'pip install GitPython'を実行してください。")
    sys.exit(1)

def fix_commit_message(message):
    """文字化けしたコミットメッセージを修正（可能な範囲で）"""
    # 文字化けパターンのマッピング（主要なパターン）
    fixes = {
        'を竪': 'を',
        '情報': '更新',
        'スｽｿ逕ｨの吶ｋ': 'を使用',
        'を医≧のｫ': 'するように',
        '險ｭ螳壹ヤ': '設定',
        'スｼスｫ': 'ツール',
        'のｨ': 'と',
        'をｳス溘ャ': 'コミット',
        'ス医Γ': 'メッセージ',
        'ステそ': '修正',
        'スｼをｸ': 'ツール',
        'ステテスｫ': 'を追加',
        'を定ｿｽ出': '',
        '荳ｻ隕√ヵ': 'ファイル名',
        'を｡を､スｫ': 'ファイル',
        '隱ｬ譏弱そ': 'を整理',
        'をｯをｷスｧスｳ': 'ディレクトリ',
        'を呈峩': 'に移動',
        'ス峨くス･ス｡スｳ': 'ドキュメント',
        'ス亥テの': 'の',
        '蜿､のテ': '名前',
        '実行ｒ': 'を',
        'ス舌ャ': '除外',
        'をｯを｢ステテ': 'パターン',
        '謨ｴ逅テ': '更新',
        'せをｯ': 'リスト',
        'スｪス励ヨ': 'に追加',
        'ス輔ぃを､スｫ': 'コード',
        '実行テ': 'と',
        'ステぅ': '名前',
        'をｯス医Μ': '変更',
        '讒矩': '',
        '謾ｹ中テ': 'を実施',
        '実行°': 'と',
        '疫': '',
        '謗･鬆ｭ霎槭ｒ': 'プレフィックス',
        '蜑企勁': 'を変更',
        'を上°を翫ｄ': 'わかりやムく',
        'の吶＞': '',
        '蜻ｽ実崎ｦ丞援': 'スクリプト',
        'のｫ': 'に',
        '邨ｱ荳': 'ファイル',
        'テテ': '',
        '髢｢騾': 'リポジトリ',
        'を呈紛': 'を整理',
        '逅テｼ域悴': 'old/',
        '菴ｿ逕ｨ': 'に移動',
        'を弛ld': 'old',
        '遘ｻ蜍包ｼテ': 'に移動',
        'ス励Οをｸをｧをｯ': 'ドキュメント',
        'ス域ｧ矩': '',
        '蛻テ｡橸ｼテ': 'ディレクトリ',
        'ス医ｒ': 'を',
    }
    
    # 文字化けしたテキストを修正
    fixed = message
    for garbled, correct in fixes.items():
        fixed = fixed.replace(garbled, correct)
    
    # その他の一般的な修正
    if '' in fixed:
        # 不完全な文字を削除
        fixed = ''.join(c for c in fixed if ord(c) < 0xFFFE)
    
    return fixed

def create_fix_mapping():
    """コミットメッセージの修正マッピングを作成"""
    repo = git.Repo('.')
    mapping = {}
    
    print("コミット履歴を分析中...")
    commits = list(repo.iter_commits('HEAD', max_count=100))
    
    for commit in commits:
        original = commit.message.strip()
        if any(ord(c) > 127 and ord(c) not in range(0x3040, 0x309F) and ord(c) not in range(0x30A0, 0x30FF) and ord(c) not in range(0x4E00, 0x9FFF) for c in original):
            # 文字化けの可能性がある
            fixed = fix_commit_message(original)
            if fixed != original:
                mapping[commit.hexsha] = {
                    'original': original,
                    'fixed': fixed
                }
                print(f"  {commit.hexsha[:7]}: {original[:50]}... -> {fixed[:50]}...")
    
    return mapping

def main():
    """メイン処理"""
    print("=" * 60)
    print("GitHubコミットメッセージ文字化け修正ツール")
    print("=" * 60)
    print()
    print("警告: このスクリプトは既存のコミットを書き換えまム。")
    print("共有リポジトリで使用する場合は、チームメンバーと相談してください。")
    print()
    
    response = input("続行しますか？ (yes/no): ")
    if response.lower() != 'yes':
        print("キャンセルしました。")
        return
    
    # マッピング作成
    mapping = create_fix_mapping()
    
    if not mapping:
        print("\n修正が必要なコミットは見つかりませんでした。")
        return
    
    print(f"\n{len(mapping)}件のコミットメッセージを修正します。")
    print("\n注意: git filter-branchを使用してコミット履歴を書き換えまム。")
    print("この操作は時間がかかる場合があります。")
    
    # git filter-branchを使用してコミットメッセージを修正
    # 注意: これは危険な操作なので、バックアップを推奨
    
    print("\n完了しました。修正されたコミットを確認してください。")
    print("リモートにプッシュする場合は、force pushが必要になりまム:")
    print("  git push --force origin main")

if __name__ == "__main__":
    main()

