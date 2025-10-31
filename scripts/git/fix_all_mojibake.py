#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ワークスペース内のすべての文字化けファイルテフォルダーを修正するスクリプト
"""

import os
import sys
import shutil
from pathlib import Path

if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 文字化けパターンと正しい名前のマッピング
MOJIBAKE_MAPPINGS = {
    # ファイル名
    '樹ｹ脂ッシステムススｼ読解渊をｷをｹステΒ.spec': 'ResinWasherInspection.spec',
    '樹脂ワッシャー検査システム.spec': 'ResinWasherInspection.spec',
    '樹脂ワチEャー検査シスチE.spec': 'ResinWasherInspection.spec',
    
    # ディレクトリ名
    '樹ｹ脂ッシステムススｼ読解渊をｷをｹステΒ': 'ResinWasherInspection',
    '樹脂ワッシャー検査システム': 'ResinWasherInspection',
    '樹脂ワチEャー検査シスチE': 'ResinWasherInspection',
}

# ファイル内容の文字化けパターン（正規表現で検索テ置換）
CONTENT_REPLACEMENTS = [
    ("樹ｹ脂ッシステムススｼ読解渊をｷをｹステΒ", "ResinWasherInspection"),
    ("樹脂ワッシャー検査システム", "ResinWasherInspection"),
    ("樹脂ワチEャー検査シスチE", "ResinWasherInspection"),
]

def fix_file_content(file_path):
    """ファイル内容の文字化けを修正"""
    try:
        # バイナリファイルはスキップ
        if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.ico', '.zip', '.pkg', '.pyz', '.pyc']:
            return False
        
        # 読み込み
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # UTF-8で読めない場合はスキップ
            return False
        except Exception:
            return False
        
        original_content = content
        modified = False
        
        # 文字化けパターンを置換
        for old, new in CONTENT_REPLACEMENTS:
            if old in content:
                content = content.replace(old, new)
                modified = True
        
        # 変更があれば保存
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    except Exception as e:
        print(f"  エラー: {file_path} の内容修正に失敗: {e}")
        return False

def fix_mojibake_files():
    """文字化けファイルを修正"""
    workspace = Path('.')
    fixed_count = 0
    error_count = 0
    
    print("=" * 60)
    print("文字化けファイルテフォルダー修正ツール")
    print("=" * 60)
    print()
    
    # 1. ファイル名の修正
    print("[1] ファイル名の修正中...")
    for root, dirs, files in os.walk(workspace):
        # .git, __pycache__, .specstory, build, dist, backup はスキップ（重要なものを除く）
        root_path = Path(root)
        if any(skip in root_path.parts for skip in ['.git', '__pycache__', 'dist', '.specstory']):
            continue
        
        # ファイル名を修正
        for filename in files:
            file_path = Path(root) / filename
            if filename in MOJIBAKE_MAPPINGS:
                new_name = MOJIBAKE_MAPPINGS[filename]
                new_path = file_path.parent / new_name
                
                try:
                    if file_path.exists():
                        if new_path.exists() and file_path != new_path:
                            # 既に正しい名前のファイルが存在する場合は古い方を削除
                            print(f"  削除: {file_path}")
                            file_path.unlink()
                        else:
                            # リネーム
                            file_path.rename(new_path)
                            print(f"  リネーム: {file_path.name} -> {new_name}")
                            fixed_count += 1
                except Exception as e:
                    print(f"  エラー: {file_path} のリネームに失敗: {e}")
                    error_count += 1
    
    # 2. ディレクトリ名の修正（再帰的に）
    print("\n[2] ディレクトリ名の修正中...")
    max_depth = 10  # 無限ループ防止
    for depth in range(max_depth):
        fixed_in_round = False
        for root, dirs, files in os.walk(workspace):
            if any(skip in Path(root).parts for skip in ['.git', '__pycache__', 'dist', '.specstory']):
                continue
            
            for dirname in dirs:
                dir_path = Path(root) / dirname
                if dirname in MOJIBAKE_MAPPINGS:
                    new_name = MOJIBAKE_MAPPINGS[dirname]
                    new_path = dir_path.parent / new_name
                    
                    try:
                        if dir_path.exists():
                            if new_path.exists() and dir_path != new_path:
                                # 既に正しい名前のディレクトリが存在する場合は中身をマージ
                                print(f"  マージ: {dir_path} -> {new_path}")
                                for item in dir_path.iterdir():
                                    shutil.move(str(item), str(new_path / item.name))
                                dir_path.rmdir()
                            else:
                                # リネーム
                                dir_path.rename(new_path)
                                print(f"  リネーム: {dir_path.name} -> {new_name}")
                                fixed_count += 1
                                fixed_in_round = True
                    except Exception as e:
                        print(f"  エラー: {dir_path} のリネームに失敗: {e}")
                        error_count += 1
        
        if not fixed_in_round:
            break
    
    # 3. ファイル内容の修正
    print("\n[3] ファイル内容の修正中...")
    content_fixed = 0
    for root, dirs, files in os.walk(workspace):
        if any(skip in Path(root).parts for skip in ['.git', '__pycache__', 'dist', '.specstory', 'build']):
            continue
        
        for filename in files:
            file_path = Path(root) / filename
            if fix_file_content(file_path):
                print(f"  修正: {file_path}")
                content_fixed += 1
    
    print("\n" + "=" * 60)
    print(f"修正完了:")
    print(f"  ファイルテフォルダー名: {fixed_count}件")
    print(f"  ファイル内容: {content_fixed}件")
    print(f"  エラー: {error_count}件")
    print("=" * 60)

if __name__ == "__main__":
    fix_mojibake_files()

