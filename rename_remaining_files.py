#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ルートディレクトリの残りのファイル名整理
"""

import os
import shutil
from pathlib import Path

def rename_remaining():
    """ルートディレクトリの残りのファイルをリネーム"""
    base_dir = Path(__file__).parent
    print(f"[名前変更] 作業ディレクトリ: {base_dir}")
    
    # ルートディレクトリのリネームマッピング
    rename_mapping = {
        'advanced_deep_learning_trainer.py': 'scripts/train_2class_basic.py',
        'advanced_focus_inspection.py': 'old/inspection_focus.py',
        'advanced_trainer.py': 'scripts/train_2class_v2.py',
        'camera2_ultra_color_inspection.py': 'old/camera2_inspection.py',
        'final_fixed_inspection.py': 'old/inspection_fixed.py',
        'final_perfect_inspection.py': 'old/inspection_perfect.py',
        'final_trainer.py': 'old/trainer_final.py',
        'ultra_high_accuracy_inspection.py': 'old/inspection_ultra.py',
        'ultra_high_accuracy_inspection_fixed.py': 'old/inspection_ultra_fixed.py',
        'ultra_high_accuracy_trainer.py': 'scripts/train_2class_v3.py',
        'ultra_simple_camera.py': 'old/camera_simple.py',
    }
    
    # old/ディレクトリ作成
    old_dir = base_dir / 'old'
    old_dir.mkdir(exist_ok=True)
    print(f"  [OK] old/ ディレクトリ作成")
    
    renamed_count = 0
    
    print("\n[名前変更] ルートディレクトリのファイル名変更開始...")
    for old_name, new_path in rename_mapping.items():
        old_file = base_dir / old_name
        new_file = base_dir / new_path
        
        if old_file.exists():
            # ディレクトリが存在することを確認
            new_file.parent.mkdir(parents=True, exist_ok=True)
            
            # ファイル移動
            shutil.move(str(old_file), str(new_file))
            print(f"  [OK] {old_name}")
            print(f"       -> {new_path}")
            renamed_count += 1
        else:
            print(f"  [NOT FOUND] {old_name}")
    
    print(f"\n[名前変更] 完了！ {renamed_count}個のファイルを変更しました")

if __name__ == "__main__":
    try:
        rename_remaining()
    except Exception as e:
        print(f"\n[エラー] {e}")
        import traceback
        traceback.print_exc()

