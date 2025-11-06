#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.specstoryフォルダ内の文字化けファイル名を修正するスクリプト
"""

import os
import sys
from pathlib import Path

if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 文字化けパターンと正しい名前のマッピング
SPECSTORY_MAPPINGS = {
    # ファイル名の文字化けパターン（部分一致で修正）
    'git-hubAgȂ': 'GitHub文字化け修正',
    'R[h̏SҌ': 'コードの整理と再構成',
    'aiwK̔rR[h쐬': 'AI学習データの比較コード作成',
    '\tgEFẢ@l': 'システムの最適化方法を検討',
    'githubւ̎]VXȅC': 'GitHubへの自動同期システムの修正',
    'R[hs': 'コード整理と実行',
    'OόVXẻƃXCh쐬': '外部通信システムの改善とクラス作成',
}

def fix_specstory_files():
    """.specstoryフォルダ内の文字化けファイルを修正"""
    specstory_dir = Path('.specstory/history')
    
    if not specstory_dir.exists():
        print(".specstory/historyフォルダが見つかりません")
        return
    
    print("=" * 60)
    print(".specstoryフォルダ内の文字化けファイル修正")
    print("=" * 60)
    
    fixed_count = 0
    
    for file_path in specstory_dir.glob('*.md'):
        original_name = file_path.name
        
        # 文字化けパターンをチェック
        fixed_name = original_name
        for garbled, correct in SPECSTORY_MAPPINGS.items():
            if garbled in fixed_name:
                # 日時の部分を保持して、文字化け部分だけを置換
                # ファイル名の形式: YYYY-MM-DD_HH-MMZ-文字化け部分.md
                parts = fixed_name.split('-')
                if len(parts) >= 3:
                    date_part = '-'.join(parts[:2])  # 日時部分
                    garbled_part = '-'.join(parts[2:])  # 残りの部分
                    if garbled in garbled_part:
                        new_name = f"{date_part}-{correct}.md"
                        new_path = file_path.parent / new_name
                        
                        try:
                            if not new_path.exists():
                                file_path.rename(new_path)
                                print(f"  リネーム: {original_name}")
                                print(f"        -> {new_name}")
                                fixed_count += 1
                            else:
                                print(f"  スキップ: {original_name} (既に正しい名前のファイルが存在)")
                                file_path.unlink()  # 重複を削除
                        except Exception as e:
                            print(f"  エラー: {original_name} のリネームに失敗: {e}")
    
    print(f"\n修正完了: {fixed_count}件")

if __name__ == "__main__":
    fix_specstory_files()

