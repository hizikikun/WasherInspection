#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リポジトリ内のすべてのファイルとコメントの文字化けをチェックして修正
"""

import os
import sys
import re
from pathlib import Path

# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# 文字化けパターン（日本語の文字化け）
MOJIBAKE_PATTERNS = {
    # 「す」の文字化け
    '試み': '試み',
    '試み': '試み',
    
    # 「すべて」「全て」の文字化け
    'すべて': 'すべて',
    'すべて': 'すべて',
    
    # 「と」の文字化け
    'と整理': 'と整理',
    'と修正': 'と修正',
    
    # 「です」「ます」の文字化け
    'です': 'です',
    'あります': 'あります',
    'します': 'します',
    'です。': 'です。',
    'あります。': 'あります。',
    
    # 「する」の文字化け
    'する': 'する',
    'する': 'する',
    
    # 「の」の文字化け
    'の': 'の',
    
    # その他の一般的な文字化け
    'ファイル': 'ファイル',
    'の': 'の',
    'が': 'が',
    'を': 'を',
}

# チェック対象の拡張子
TARGET_EXTENSIONS = ['.py', '.md', '.txt', '.bat', '.ps1', '.sh', '.json', '.yml', '.yaml']

# 除外ディレクトリ
EXCLUDE_DIRS = {
    '__pycache__', '.git', 'node_modules', 'venv', 'env', '.venv',
    'dist', 'build', 'backup', '.pytest_cache', '.mypy_cache',
    'models', 'logs', 'temp', 'cs_AItraining_data', 'feedback_data',
}

def detect_mojibake(text):
    """文字化けを検出"""
    issues = []
    for pattern, correct in MOJIBAKE_PATTERNS.items():
        if pattern in text:
            issues.append((pattern, correct))
    return issues

def fix_mojibake_in_content(content):
    """コンテンツ内の文字化けを修正"""
    fixed_content = content
    replacements = []
    
    for pattern, correct in MOJIBAKE_PATTERNS.items():
        if pattern in fixed_content:
            count = fixed_content.count(pattern)
            fixed_content = fixed_content.replace(pattern, correct)
            replacements.append((pattern, correct, count))
    
    return fixed_content, replacements

def fix_file_encoding_and_mojibake(file_path):
    """ファイルのエンコーディングと文字化けを修正"""
    try:
        # 複数のエンコーディングで試す
        content = None
        used_encoding = None
        
        for encoding in ['utf-8', 'cp932', 'shift-jis', 'windows-1252', 'latin-1', 'euc-jp']:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
                used_encoding = encoding
                break
            except Exception:
                continue
        
        if content is None:
            return False, "読み込み失敗"
        
        # 文字化けを検出・修正
        original_content = content
        fixed_content, replacements = fix_mojibake_in_content(content)
        
        # 修正があった場合のみ保存
        if fixed_content != original_content or used_encoding != 'utf-8':
            # UTF-8で保存
            with open(file_path, 'w', encoding='utf-8', errors='replace') as f:
                f.write(fixed_content)
            
            if replacements:
                return True, f"文字化け修正: {', '.join([f'{p}→{c}({n}回)' for p, c, n in replacements])}"
            else:
                return True, f"エンコーディング修正: {used_encoding} → utf-8"
        
        return False, None
    
    except Exception as e:
        return False, f"エラー: {e}"

def scan_directory(root_dir='.'):
    """ディレクトリをスキャンして文字化けを検出・修正"""
    root_path = Path(root_dir)
    total_files = 0
    fixed_files = 0
    issues_found = []
    
    print("=" * 60)
    print("リポジトリ内の文字化けチェックと修正")
    print("=" * 60)
    print()
    
    for file_path in root_path.rglob('*'):
        # 除外ディレクトリをスキップ
        if any(excluded in file_path.parts for excluded in EXCLUDE_DIRS):
            continue
        
        # ファイルのみ処理
        if not file_path.is_file():
            continue
        
        # 対象拡張子のみ処理
        if file_path.suffix not in TARGET_EXTENSIONS:
            continue
        
        total_files += 1
        
        try:
            fixed, message = fix_file_encoding_and_mojibake(file_path)
            if fixed:
                fixed_files += 1
                relative_path = file_path.relative_to(root_path)
                print(f"✓ 修正: {relative_path}")
                if message:
                    print(f"  {message}")
                    issues_found.append((relative_path, message))
        except Exception as e:
            relative_path = file_path.relative_to(root_path)
            print(f"✗ エラー: {relative_path} - {e}")
    
    print()
    print("=" * 60)
    print(f"スキャン完了: {total_files}ファイル")
    print(f"修正完了: {fixed_files}ファイル")
    print("=" * 60)
    
    if issues_found:
        print()
        print("修正された問題:")
        for path, msg in issues_found[:20]:  # 最初の20個まで表示
            print(f"  - {path}: {msg}")
        if len(issues_found) > 20:
            print(f"  ... 他 {len(issues_found) - 20}個")
    
    return fixed_files

def main():
    print("リポジトリ全体の文字化けチェックを開始します...")
    print()
    
    fixed_count = scan_directory('.')
    
    print()
    if fixed_count > 0:
        print(f"{fixed_count}個のファイルを修正しました。")
        print("変更を確認してからコミットしてください:")
        print("  git status")
        print("  git add .")
        print("  git commit -m 'リポジトリ内の文字化け修正'")
        print("  git push")
    else:
        print("文字化けは見つかりませんでした。")

if __name__ == '__main__':
    main()

