#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ファイル整理スクリプト
ワークディレクトリのファイルを整理します
"""

import os
import shutil
from pathlib import Path

def organize_files():
    """ファイルを整理する"""
    base_dir = Path(__file__).parent
    print(f"[整理] 作業ディレクトリ: {base_dir}")
    
    # 作成するディレクトリ
    dirs_to_create = {
        'scripts': 'AI学習スクリプト',
        'github_tools': 'GitHub統合ツール',
        'config': '設定ファイル',
        'batch_files': 'バッチファイル',
        'docs': 'ドキュメント'
    }
    
    # GitHub関連ファイル
    github_files = [
        'github_auto_commit_system.py',
        'integrated_github_system.py',
        'code_training_auto_sync.py',
        'cursor_github_integration.py',
        'github_washer_inspection_system.py',
        'auto_github_token_creator.py',
        'integrated_auto_sync.py',
        'cursor_github_extension.py',
        'setup_github_token.py'
    ]
    
    # 学習スクリプト
    script_files = [
        'clear_progress_sparse_modeling_four_class_ensemble.py',
        'advanced_deep_learning_with_spatial_modeling.py',
        'ultra_high_accuracy_ensemble.py',
        'four_class_ultra_high_accuracy_ensemble.py',
        'corrected_four_class_ultra_high_accuracy_ensemble.py',
        'sparse_modeling_four_class_ultra_high_accuracy_ensemble.py'
    ]
    
    # 設定ファイル
    config_files = [
        'github_config.json',
        'cursor_github_config.json',
        'auto_sync_config.json',
        'training_statistics.json',
        'class_names.json'
    ]
    
    # バッチファイル
    batch_files = [
        'start_github_auto_commit.bat',
        'start_auto_sync.bat',
        'sync_once.bat',
        'github_sync_once.bat'
    ]
    
    # ドキュメント
    doc_files = [
        'GITHUB_SETUP_GUIDE.md',
        'CODE_TRAINING_SYNC_GUIDE.md',
        'README.md',
        'PROJECT_STRUCTURE.md'
    ]
    
    # ディレクトリ作成
    print("\n[整理] ディレクトリ作成中...")
    for dir_name, desc in dirs_to_create.items():
        dir_path = base_dir / dir_name
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"  [OK] {dir_name}/ ({desc})")
        else:
            print(f"  - {dir_name}/ は既に存在します")
    
    # ファイル移動
    moved_count = 0
    
    print("\n[整理] GitHub関連ファイル移動中...")
    for filename in github_files:
        src = base_dir / filename
        dst = base_dir / 'github_tools' / filename
        if src.exists():
            if not dst.exists():
                shutil.move(str(src), str(dst))
                print(f"  [OK] {filename} -> github_tools/")
                moved_count += 1
            else:
                print(f"  - {filename} は既に移動済み")
        else:
            print(f"  × {filename} が見つかりません")
    
    print("\n[整理] 学習スクリプト移動中...")
    for filename in script_files:
        src = base_dir / filename
        dst = base_dir / 'scripts' / filename
        if src.exists():
            if not dst.exists():
                shutil.move(str(src), str(dst))
                print(f"  [OK] {filename} -> scripts/")
                moved_count += 1
            else:
                print(f"  - {filename} は既に移動済み")
        else:
            print(f"  × {filename} が見つかりません")
    
    print("\n[整理] 設定ファイル移動中...")
    for filename in config_files:
        src = base_dir / filename
        dst = base_dir / 'config' / filename
        if src.exists():
            if not dst.exists():
                shutil.move(str(src), str(dst))
                print(f"  [OK] {filename} -> config/")
                moved_count += 1
            else:
                print(f"  - {filename} は既に移動済み")
        else:
            print(f"  × {filename} が見つかりません")
    
    print("\n[整理] バッチファイル移動中...")
    for filename in batch_files:
        src = base_dir / filename
        dst = base_dir / 'batch_files' / filename
        if src.exists():
            if not dst.exists():
                shutil.move(str(src), str(dst))
                print(f"  [OK] {filename} -> batch_files/")
                moved_count += 1
            else:
                print(f"  - {filename} は既に移動済み")
        else:
            print(f"  × {filename} が見つかりません")
    
    print("\n[整理] ドキュメント移動中...")
    for filename in doc_files:
        src = base_dir / filename
        dst = base_dir / 'docs' / filename
        if src.exists():
            if not dst.exists():
                shutil.move(str(src), str(dst))
                print(f"  [OK] {filename} -> docs/")
                moved_count += 1
            else:
                print(f"  - {filename} は既に移動済み")
        else:
            print(f"  × {filename} が見つかりません")
    
    print(f"\n[整理] 完了！ {moved_count}個のファイルを移動しました")
    print("\n[整理] 整理後の構造:")
    print("  [DIR] scripts/          - AI学習スクリプト")
    print("  [DIR] github_tools/     - GitHub統合ツール")
    print("  [DIR] config/            - 設定ファイル")
    print("  [DIR] batch_files/       - バッチファイル")
    print("  [DIR] docs/              - ドキュメント")
    print("  [FILE] main.py            - メインシステム（そのまま）")
    print("  [FILE] improved_multi_camera_selection_step5.py - カメラシステム（そのまま）")

if __name__ == "__main__":
    try:
        organize_files()
    except Exception as e:
        print(f"\n[エラー] {e}")
        import traceback
        traceback.print_exc()
