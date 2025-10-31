#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
検出された文字化けをすべて修正
"""

import os
import sys
import re
from pathlib import Path
# import chardet  # オプション: pip install chardet

# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def detect_encoding(file_path):
    """ファイルのエンコーディングを検出"""
    # chardetが利用できない場合はNoneを返す
    return None, 0

def read_file_safely(file_path):
    """ファイルを安全に読み込む（複数のエンコーディングを試す）"""
    encodings = ['utf-8', 'cp932', 'shift-jis', 'euc-jp', 'windows-1252', 'latin-1']
    
    # まずchardetで検出
    detected_encoding, confidence = detect_encoding(file_path)
    if detected_encoding and confidence > 0.7:
        if detected_encoding.lower() not in ['ascii']:
            encodings.insert(0, detected_encoding)
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            return content, encoding
        except Exception:
            continue
    
    # それでも失敗した場合はUTF-8で強制読み込み
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return content, 'utf-8'
    except Exception:
        return None, None

def fix_mojibake_text(text):
    """文字化けテキストを修正"""
    # よくある文字化けパターンの置換
    fixes = {
        # Shift-JIS → UTF-8の誤変換
        'データ': 'データ',
        '繧｡繧､繝': 'デー',
        'ファイル': 'ファイル',
        'エラー': 'エラー',
        'の': 'の',
        'が': 'が',
        'を': 'を',
        '縺': '',
        '設定': '設定',
        'ファイル': 'ファイル',
        'システム': 'システム',
        'システム': 'システム',
        'データ': 'データ',
        'ファイル': 'ファイル',
    }
    
    fixed_text = text
    replacements = []
    
    for pattern, replacement in fixes.items():
        if pattern in fixed_text:
            count = fixed_text.count(pattern)
            fixed_text = fixed_text.replace(pattern, replacement)
            if count > 0:
                replacements.append(f"{pattern} → {replacement} ({count}回)")
    
    return fixed_text, replacements

def fix_file(file_path):
    """ファイルの文字化けを修正"""
    try:
        content, encoding = read_file_safely(file_path)
        if content is None:
            return False, "読み込み失敗"
        
        # 文字化け修正
        fixed_content, replacements = fix_mojibake_text(content)
        
        # 変更がない場合はスキップ
        if fixed_content == content:
            return False, "変更なし"
        
        # UTF-8で保存
        with open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            f.write(fixed_content)
        
        return True, replacements
    except Exception as e:
        return False, str(e)

def fix_all_files():
    """すべてのファイルを修正"""
    print("=" * 60)
    print("文字化け修正")
    print("=" * 60)
    print()
    
    target_dirs = [
        'scripts',
        'github_tools',
        'tools',
        'inspectors',
        'trainers',
        'utilities',
        'dashboard',
        'docs',
    ]
    
    total_files = 0
    fixed_files = 0
    
    for target_dir in target_dirs:
        dir_path = Path(target_dir)
        if not dir_path.exists():
            continue
        
        print(f"処理中: {target_dir}/")
        
        # Pythonファイル
        for file_path in dir_path.rglob('*.py'):
            total_files += 1
            success, result = fix_file(file_path)
            if success:
                fixed_files += 1
                if isinstance(result, list):
                    print(f"  [修正] {file_path.name}")
                    for rep in result[:2]:  # 最初の2件のみ表示
                        print(f"    {rep}")
                else:
                    print(f"  [修正] {file_path.name}: {result}")
        
        # Markdownファイル
        for file_path in dir_path.rglob('*.md'):
            total_files += 1
            success, result = fix_file(file_path)
            if success:
                fixed_files += 1
                print(f"  [修正] {file_path.name}")
        
        # テキストファイル
        for file_path in dir_path.rglob('*.txt'):
            total_files += 1
            success, result = fix_file(file_path)
            if success:
                fixed_files += 1
                print(f"  [修正] {file_path.name}")
    
    print()
    print("=" * 60)
    print("修正結果")
    print("=" * 60)
    print(f"処理ファイル数: {total_files}")
    print(f"修正ファイル数: {fixed_files}")
    print()

def main():
    fix_all_files()
    
    print("=" * 60)
    print("修正完了")
    print("=" * 60)
    print()
    print("次のステップ:")
    print("  1. git add .")
    print("  2. git commit -m '文字化け修正'")
    print("  3. git push")

if __name__ == '__main__':
    main()

