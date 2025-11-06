#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
すべてのファイルの文字化けをチェック
"""

import os
import sys
import re
from pathlib import Path
from collections import defaultdict

# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# 文字化けパターン（よくある文字化け）
MOJIBAKE_PATTERNS = [
    # 日本語の文字化けパターン
    r'繧[^\x00-\x7F]+',  # 繧で始まる文字化け
    r'縺[^\x00-\x7F]+',  # 縺で始まる文字化け
    r'譁[^\x00-\x7F]+',  # 譁で始まる文字化け
    r'蠖[^\x00-\x7F]+',  # 蠖で始まる文字化け
    r'・[^\x00-\x7F]+',  # ・で始まる文字化け（半角・全角）
    # 一般的な文字化け文字
    r'[讓閼ゅΡ繝す繝繝ｼシステムシステム]+',
    r'[データ蜷阪繧呈蟷蜉蝟]+',
    r'[ファイルエラーのがを]+',
    r'[設定のがを]+',
]

def detect_mojibake_in_text(text, file_path):
    """テキスト内の文字化けを検出"""
    issues = []
    lines = text.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        # 文字化けパターンのチェック
        for pattern in MOJIBAKE_PATTERNS:
            matches = re.finditer(pattern, line)
            for match in matches:
                issues.append({
                    'line': line_num,
                    'position': match.start(),
                    'text': match.group(),
                    'pattern': pattern,
                    'context': line[:100] if len(line) > 100 else line
                })
        
        # 一般的な文字化け文字の検出（連続する異常な文字）
        # Shift-JISからUTF-8への誤変換など
        if re.search(r'[\uFF00-\uFFEF]{3,}', line):  # 全角文字が3文字以上連続
            if any(char in line for char in ['繧', '縺', '譁', '蠖']):
                issues.append({
                    'line': line_num,
                    'position': 0,
                    'text': line[:50],
                    'pattern': 'suspicious_wide_char',
                    'context': line[:100] if len(line) > 100 else line
                })
    
    return issues

def check_file_encoding(file_path):
    """ファイルのエンコーディングをチェック"""
    encodings = ['utf-8', 'cp932', 'shift-jis', 'windows-1252', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            return content, encoding
        except Exception:
            continue
    
    return None, None

def fix_mojibake_in_content(content, issues):
    """文字化けを修正（可能な場合）"""
    fixed_content = content
    fixes_applied = []
    
    # よくある文字化けの置換
    common_fixes = {
        'データ': 'データ',
        'ファイル': 'ファイル',
        'エラー': 'エラー',
        'の': 'の',
        'が': 'が',
        'を': 'を',
        '設定': '設定',
    }
    
    for original, fixed in common_fixes.items():
        if original in fixed_content:
            fixed_content = fixed_content.replace(original, fixed)
            fixes_applied.append(f"{original} → {fixed}")
    
    return fixed_content, fixes_applied

def scan_all_files():
    """すべてのファイルをスキャン"""
    print("=" * 60)
    print("全ファイルの文字化けチェック")
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
    
    all_issues = defaultdict(list)
    total_files = 0
    files_with_issues = 0
    
    for target_dir in target_dirs:
        dir_path = Path(target_dir)
        if not dir_path.exists():
            continue
        
        print(f"スキャン中: {target_dir}/")
        
        # Pythonファイル
        for file_path in dir_path.rglob('*.py'):
            total_files += 1
            try:
                content, encoding = check_file_encoding(file_path)
                if content is None:
                    print(f"  [ERROR] {file_path}: 読み込み失敗")
                    continue
                
                issues = detect_mojibake_in_text(content, file_path)
                if issues:
                    files_with_issues += 1
                    all_issues[str(file_path)] = {
                        'encoding': encoding,
                        'issues': issues
                    }
                    print(f"  [問題] {file_path.name}: {len(issues)}件の問題")
            except Exception as e:
                print(f"  [ERROR] {file_path}: {e}")
        
        # Markdownファイル
        for file_path in dir_path.rglob('*.md'):
            total_files += 1
            try:
                content, encoding = check_file_encoding(file_path)
                if content is None:
                    continue
                
                issues = detect_mojibake_in_text(content, file_path)
                if issues:
                    files_with_issues += 1
                    all_issues[str(file_path)] = {
                        'encoding': encoding,
                        'issues': issues
                    }
                    print(f"  [問題] {file_path.name}: {len(issues)}件の問題")
            except Exception:
                pass
    
    print()
    print("=" * 60)
    print("チェック結果")
    print("=" * 60)
    print(f"総ファイル数: {total_files}")
    print(f"問題のあるファイル: {files_with_issues}")
    print()
    
    if all_issues:
        print("問題のあるファイルの詳細:")
        print()
        for file_path, data in sorted(all_issues.items()):
            print(f"  {file_path} (encoding: {data['encoding']})")
            for issue in data['issues'][:3]:  # 最初の3件のみ表示
                print(f"    行 {issue['line']}: {issue['text'][:50]}")
            if len(data['issues']) > 3:
                print(f"    ... 他 {len(data['issues']) - 3}件")
            print()
    
    return all_issues

def main():
    issues = scan_all_files()
    
    if issues:
        print("=" * 60)
        print("修正が必要です")
        print("=" * 60)
        print()
        print("修正スクリプトを実行してください:")
        print("  python scripts/fix_all_mojibake_found.py")
    else:
        print("=" * 60)
        print("文字化けは見つかりませんでした")
        print("=" * 60)

if __name__ == '__main__':
    main()

