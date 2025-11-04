#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ファイル名整理スクリプト
わかりにくいプレフィックス（ultra_, final_, advanced_など）を削除
"""

import os
import shutil
from pathlib import Path

def rename_files():
    """ファイル名を整理する"""
    base_dir = Path(__file__).parent
    print(f"[名前変更] 作業ディレクトリ: {base_dir}")
    
    # リネームマッピング（旧名 -> 新名）
    rename_mapping = {
        # scripts/ ディレクトリ
        'scripts/clear_progress_sparse_modeling_four_class_ensemble.py': 'scripts/train_4class_sparse_ensemble.py',
        'scripts/advanced_deep_learning_with_spatial_modeling.py': 'scripts/train_2class_with_augmentation.py',
        'scripts/ultra_high_accuracy_ensemble.py': 'scripts/train_2class_ensemble.py',
        'scripts/four_class_ultra_high_accuracy_ensemble.py': 'scripts/train_4class_ensemble.py',
        'scripts/corrected_four_class_ultra_high_accuracy_ensemble.py': 'scripts/train_4class_ensemble_v2.py',
        'scripts/sparse_modeling_four_class_ultra_high_accuracy_ensemble.py': 'scripts/train_4class_sparse_ensemble.py',
        
        # github_tools/ ディレクトリ
        'github_tools/integrated_github_system.py': 'github_tools/github_sync.py',
        'github_tools/github_auto_commit_system.py': 'github_tools/github_autocommit.py',
        'github_tools/code_training_auto_sync.py': 'github_tools/auto_sync.py',
        'github_tools/cursor_github_integration.py': 'github_tools/cursor_integration.py',
        'github_tools/github_washer_inspection_system.py': 'github_tools/github_integration.py',
        'github_tools/auto_github_token_creator.py': 'github_tools/token_setup.py',
        'github_tools/integrated_auto_sync.py': 'github_tools/sync.py',
        'github_tools/cursor_github_extension.py': 'github_tools/cursor_extension.py',
        'github_tools/setup_github_token.py': 'github_tools/token_setup.py',
        
        # ルートディレクトリ
        'improved_multi_camera_selection_step5.py': 'camera_inspection.py',
    }
    
    renamed_count = 0
    skipped_count = 0
    
    print("\n[名前変更] ファイル名変更開始...")
    for old_path, new_path in rename_mapping.items():
        old_file = base_dir / old_path
        new_file = base_dir / new_path
        
        if old_file.exists():
            if new_file.exists():
                print(f"  [SKIP] {old_path} -> {new_path} (既に存在)")
                skipped_count += 1
            else:
                # ディレクトリが存在することを確認
                new_file.parent.mkdir(parents=True, exist_ok=True)
                
                # ファイル移動
                shutil.move(str(old_file), str(new_file))
                print(f"  [OK] {old_path}")
                print(f"       -> {new_path}")
                renamed_count += 1
        else:
            # 既に移動済みか、存在しない
            if new_file.exists():
                print(f"  [SKIP] {old_path} (既に {new_path} として存在)")
                skipped_count += 1
            else:
                print(f"  [NOT FOUND] {old_path}")
    
    print(f"\n[名前変更] 完了！ {renamed_count}個のファイルを変更しました")
    if skipped_count > 0:
        print(f"          {skipped_count}個のファイルをスキップしました")
    
    # 新しい構造を表示
    print("\n[名前変更] 整理後のファイル構造:")
    print("\n  [scripts/]")
    print("    - train_4class_sparse_ensemble.py")
    print("    - train_2class_with_augmentation.py")
    print("    - train_2class_ensemble.py")
    print("    - train_4class_ensemble.py")
    
    print("\n  [github_tools/]")
    print("    - github_sync.py")
    print("    - github_autocommit.py")
    print("    - auto_sync.py")
    print("    - cursor_integration.py")
    
    print("\n  [ルート]")
    print("    - main.py")
    print("    - camera_inspection.py")
    
    # import文の修正が必要なファイルをチェック
    print("\n[注意] import文の修正が必要なファイルがあります:")
    print("  - github_tools/ 内のファイルの相互import")
    print("  - scripts/ 内のファイルの相互import")
    print("  これらのファイルのimport文を手動で修正してください")

if __name__ == "__main__":
    try:
        rename_files()
    except Exception as e:
        print(f"\n[エラー] {e}")
        import traceback
        traceback.print_exc()

