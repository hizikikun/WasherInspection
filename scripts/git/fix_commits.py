#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
すべてのコミットメッセージを修正するフィルタースクリプト
git filter-branch の --msg-filter で使用
"""

import sys
import re

# UTF-8 stdin/stdout設定
if sys.platform.startswith('win'):
    import codecs
    sys.stdin = codecs.getreader('utf-8')(sys.stdin.buffer)
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

def fix_message(msg):
    """コミットメッセージを修正"""
    
    # 完全一致の修正マッピング
    exact_fixes = {
        '情報: PROJECT_STRUCTURE.mdの荳ｻ隕√ヵを｡を､スｫ隱ｬ譏弱そをｯをｷスｧスｳを呈峩がｰ': 
            '修正: PROJECT_STRUCTURE.mdの主要ファイル説明セクションを更新',
        
        '譖ｴがｰ: docs/PROJECT_STRUCTURE.mdのGitHubステテスｫ荳隕ｧをゆｿ報': 
            '更新: docs/PROJECT_STRUCTURE.mdのGitHubドキュメントを更新',
        
        '譖ｴがｰ: ス峨くス･ス｡スｳス亥テの蜿､のテヵを｡を､スｫ実行ｒ情報': 
            '更新: プロジェクトファイル整理とリネーム',
        
        '譖ｴがｰ: .gitignoreのｫス舌ャをｯを｢ステテのｨ謨ｴ逅テせをｯスｪス励ヨを定ｿｽ出': 
            '更新: .gitignoreにディレクトリとignoreファイルを追加',
        
        'fix: safe_git_commit.pyを竪itPythonを剃ｽｿ逕ｨの吶ｋを医≧のｫ情報': 
            'fix: safe_git_commit.pyをGitPythonを使用するように修正',
        
        'chore: Git UTF-8險ｭ螳壹ヤスｼスｫのｨをｳス溘ャス医Γステそスｼをｸ情報ステテスｫを定ｿｽ出': 
            'chore: Git UTF-8設定ツールとコミットメッセージ修正ツールを追加',
    }
    
    # 完全一致チェック
    msg_stripped = msg.strip()
    if msg_stripped in exact_fixes:
        return exact_fixes[msg_stripped] + '\n'
    
    # パターンマッチング修正
    # 長いメッセージ（ファイル整理関連）
    if '謨ｴ逅テ' in msg and 'ス輔ぃを､スｫ' in msg:
        if 'ultra_' in msg or 'final_' in msg or 'advanced_' in msg:
            return '整理: ファイル整理とリネーム - ultra_, final_, advanced_プレフィックス削除\n'
    
    # 部分文字列の置換
    fixes = {
        '情報': '修正',
        '譖ｴがｰ': '更新',
        '謨ｴ逅テ': '整理',
        'の': 'の',
        'のｧ': 'を',
        'のｨ': 'に',
        'のｫ': 'に',
        'を｡を､スｫ': 'ファイル',
        'ス｡スｳス亥テ': 'プロジェクト',
        'ステぅスｬをｯ': 'cursor',
        'ステテスｫ': 'ドキュメント',
        'をｯスｪス励ヨ': 'ignore',
        'ス舌ャをｯを｢': 'ディレクトリ',
        '荳ｻ隕√ヵ': '主要',
        '隱ｬ': '説明',
        '実行': 'セクション',
        '弱そ': 'を',
        '呈峩': '更新',
    }
    
    fixed = msg
    for garbled, correct in fixes.items():
        fixed = fixed.replace(garbled, correct)
    
    # 特定パターンの最終修正
    if 'PROJECT_STRUCTURE.md' in fixed and '主要' in fixed:
        fixed = '修正: PROJECT_STRUCTURE.mdの主要ファイル説明セクションを更新'
    
    return fixed

# Read from stdin, fix, write to stdout
if __name__ == '__main__':
    message = sys.stdin.read()
    fixed = fix_message(message)
    sys.stdout.write(fixed)


