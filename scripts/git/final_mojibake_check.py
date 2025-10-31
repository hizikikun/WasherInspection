#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最終的な文字化けチェックスクリプト
ワークスペース全体を再確認し、報告する
"""

import os
import sys
from pathlib import Path

if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 文字化けパターン
MOJIBAKE_PATTERNS = [
    '樹ｹ脂ッシ', 'ステムススｼ', '読解渊', 'をｷをｹステΒ',
    '情報', '譖ｴがｰ', '謨ｴ逅テ', 'の', 'のｧ', 'のｨ',
    'を｡を､スｫ', '実行', 'を呈', '検', '出', '中'
]

def check_files():
    """ファイル名の文字化けをチェック"""
    workspace = Path('.')
    issues = []
    
    print("=" * 60)
    print("文字化け最終チェック")
    print("=" * 60)
    print()
    
    # スキップするディレクトリ
    skip_dirs = {'.git', '__pycache__', 'dist', 'venv', '.venv', 'node_modules'}
    
    # ファイル名チェック
    print("[1] ファイル名の文字化けチェック...")
    file_count = 0
    for root, dirs, files in os.walk(workspace):
        root_path = Path(root)
        # スキップディレクトリを除外
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        if any(skip in root_path.parts for skip in skip_dirs):
            continue
        
        for filename in files:
            file_path = root_path / filename
            # ファイル名に文字化けパターンが含まれているかチェック
            has_mojibake = any(pattern in filename for pattern in MOJIBAKE_PATTERNS)
            
            if has_mojibake:
                issues.append(('file', str(file_path), filename))
                print(f"  [文字化け] {file_path}")
                file_count += 1
    
    print(f"  発見: {file_count}件")
    
    # ディレクトリ名チェック
    print("\n[2] ディレクトリ名の文字化けチェック...")
    dir_count = 0
    for root, dirs, files in os.walk(workspace):
        root_path = Path(root)
        if any(skip in root_path.parts for skip in skip_dirs):
            continue
        
        for dirname in dirs:
            dir_path = root_path / dirname
            has_mojibake = any(pattern in dirname for pattern in MOJIBAKE_PATTERNS)
            
            if has_mojibake:
                issues.append(('dir', str(dir_path), dirname))
                print(f"  [文字化け] {dir_path}")
                dir_count += 1
    
    print(f"  発見: {dir_count}件")
    
    # ファイル内容のチェック（テキストファイルのみ）
    print("\n[3] ファイル内容の文字化けチェック...")
    content_count = 0
    text_extensions = {'.py', '.md', '.txt', '.json', '.yml', '.yaml', '.spec', '.bat', '.ps1', '.sh'}
    
    for root, dirs, files in os.walk(workspace):
        root_path = Path(root)
        if any(skip in root_path.parts for skip in skip_dirs):
            continue
        
        for filename in files:
            file_path = root_path / filename
            if file_path.suffix.lower() not in text_extensions:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    has_mojibake = any(pattern in content for pattern in MOJIBAKE_PATTERNS)
                    
                    if has_mojibake:
                        # スクリプトファイル内のマッピング定義は除外
                        if 'MOJIBAKE_MAPPINGS' in content or 'CONTENT_REPLACEMENTS' in content:
                            continue
                        
                        issues.append(('content', str(file_path), None))
                        print(f"  [文字化け] {file_path}")
                        content_count += 1
            except (UnicodeDecodeError, PermissionError):
                continue
    
    print(f"  発見: {content_count}件")
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("チェック結果サマリー")
    print("=" * 60)
    print(f"  ファイル名: {file_count}件")
    print(f"  ディレクトリ名: {dir_count}件")
    print(f"  ファイル内容: {content_count}件")
    print(f"  合計: {len(issues)}件")
    
    if len(issues) == 0:
        print("\n✓ 文字化けは見つかりませんでした！")
    else:
        print(f"\n⚠ {len(issues)}件の文字化けが見つかりました")
        print("\n詳細:")
        for issue_type, path, name in issues:
            print(f"  [{issue_type}] {path}")
    
    return issues

if __name__ == "__main__":
    issues = check_files()
    sys.exit(0 if len(issues) == 0 else 1)

