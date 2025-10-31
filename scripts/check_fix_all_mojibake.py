#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全ファイルの文字化けをチェックして修正
コメント、文字列リテラル内の文字化けも検出と修正
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

# 文字化けパターン（Shift-JISの文字化け例）
MOJIBAKE_PATTERNS = [
    # よくある文字化けパターン
    (r'の', 'の'),  # 「の」が「の」に
    (r'を', 'を'),  # 「を」が「を」に
    (r'が', 'が'),  # 「が」が「が」に
    (r'読', '読'),  # 「読」が「読」に
    (r'解', '解'),  # 「解」が「解」に
    (r'測', '測'),  # 「測」が「測」に
    (r'樹', '樹'),  # 「樹」が「樹」に
    (r'脂', '脂'),  # 「脂」が「脂」に
    (r'ッ', 'ッ'),  # 「ッ」が「ッ」に
    (r'シ', 'シ'),   # 「シ」が「シ」に
    (r'ス', 'ス'),  # 「ス」が「ス」に
    (r'テ', 'テ'),  # 「テ」が「テ」に
    (r'ム', 'ム'),  # 「ム」が「ム」に
    (r'ス', 'ト'), # 「ト」が「ス」に
    (r'スｼ', ''),
    (r'をｷ', 'ワ'),
    (r'をｹ', 'ッ'),
    (r'ステΒ', 'シャー'),
    (r'樹ｹ', '樹'),
    (r'脂ッ', '脂ワ'),
    (r'情', '情'),
    (r'報', '報'),
    (r'の', 'の'),
    (r'のｧ', 'と'),
    (r'のｨ', 'を'),
    (r'の', 'の'),
    (r'を｡', 'テ'),
    (r'を､', 'ス'),
    (r'スｫ', 'ト'),
    (r'実', '実'),
    (r'行', '行'),
    (r'を呈', '確認'),
    (r'検', '検'),
    (r'出', '出'),
    (r'中', '中'),
]

# 文字化けが疑われる文字列パターン
SUSPICIOUS_PATTERNS = [
    r'の[^\s\w]',  # 「の」で始まる文字列
    r'を[^\s\w]',  # 「を」で始まる文字列
    r'が[^\s\w]',  # 「が」で始まる文字列
    r'読[^\s\w]',  # 「読」で始まる文字列
]

def detect_mojibake(content):
    """文字化けを検出"""
    issues = []
    
    # 文字化けパターンをチェック
    for pattern, _ in MOJIBAKE_PATTERNS:
        matches = re.finditer(pattern, content)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            context = content[max(0, match.start()-20):min(len(content), match.end()+20)]
            issues.append({
                'line': line_num,
                'pattern': pattern,
                'match': match.group(),
                'context': context.replace('\n', ' ')
            })
    
    # 疑わしいパターンをチェック
    for pattern in SUSPICIOUS_PATTERNS:
        matches = re.finditer(pattern, content)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            context = content[max(0, match.start()-20):min(len(content), match.end()+20)]
            issues.append({
                'line': line_num,
                'pattern': pattern,
                'match': match.group(),
                'context': context.replace('\n', ' ')
            })
    
    return issues

def fix_mojibake(content):
    """文字化けを修正"""
    fixed_content = content
    
    # パターンごとに置換
    for pattern, replacement in MOJIBAKE_PATTERNS:
        fixed_content = re.sub(pattern, replacement, fixed_content)
    
    return fixed_content

def check_file(file_path):
    """ファイルをチェック"""
    try:
        # 複数のエンコーディングで読み込み
        content = None
        encoding_used = None
        
        for encoding in ['utf-8', 'cp932', 'shift-jis', 'windows-1252', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
                encoding_used = encoding
                break
            except Exception:
                continue
        
        if content is None:
            return None, None, None
        
        # 文字化け検出
        issues = detect_mojibake(content)
        
        # 文字化け修正
        fixed_content = fix_mojibake(content) if issues else content
        
        return content, fixed_content, issues, encoding_used
    
    except Exception as e:
        print(f"  Error checking {file_path}: {e}")
        return None, None, None, None

def main():
    print("=" * 60)
    print("全ファイルの文字化けチェックと修正")
    print("=" * 60)
    print()
    
    # チェック対象ファイル
    target_extensions = ['.py', '.md', '.txt', '.json', '.bat', '.ps1']
    skip_dirs = {'.git', '__pycache__', 'node_modules', 'venv', '.venv', 'dist', 'build'}
    
    root = Path('.')
    checked_files = []
    fixed_files = []
    total_issues = 0
    
    # ファイルをスキャン
    print("[1/3] ファイルスキャン中...")
    for ext in target_extensions:
        for file_path in root.rglob(f'*{ext}'):
            # スキップディレクトリをチェック
            if any(skip in file_path.parts for skip in skip_dirs):
                continue
            
            checked_files.append(file_path)
    
    print(f"  発見: {len(checked_files)}個のファイル")
    print()
    
    # 各ファイルをチェック
    print("[2/3] 文字化けチェック中...")
    file_issues = defaultdict(list)
    
    for file_path in checked_files:
        content, fixed_content, issues, encoding = check_file(file_path)
        
        if issues:
            file_issues[file_path] = {
                'issues': issues,
                'content': content,
                'fixed_content': fixed_content,
                'encoding': encoding
            }
            total_issues += len(issues)
            print(f"  [文字化け検出] {file_path} ({len(issues)}個)")
    
    print()
    print(f"  合計: {total_issues}個の文字化けを検出")
    print()
    
    # 文字化けを修正
    print("[3/3] 文字化け修正中...")
    for file_path, data in file_issues.items():
        try:
            # 修正済み内容を保存
            with open(file_path, 'w', encoding='utf-8', errors='replace') as f:
                f.write(data['fixed_content'])
            fixed_files.append(file_path)
            print(f"  修正: {file_path}")
        except Exception as e:
            print(f"  エラー: {file_path} - {e}")
    
    print()
    print("=" * 60)
    print("完了")
    print("=" * 60)
    print()
    print(f"チェックファイル数: {len(checked_files)}個")
    print(f"文字化け検出: {total_issues}個")
    print(f"修正ファイル数: {len(fixed_files)}個")
    
    if fixed_files:
        print()
        print("修正されたファイル:")
        for f in fixed_files:
            print(f"  - {f}")
        print()
        print("次のステップ:")
        print("  1. git add .")
        print("  2. git commit -m '文字化け修正'")
        print("  3. git push")

if __name__ == '__main__':
    main()

