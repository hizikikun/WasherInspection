#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重複ファイル整理スクリプト
- メインファイルを特定し、重複しているファイルをoldフォルダに移動
"""

import os
import shutil
from pathlib import Path

# メインファイル（改良版）
MAIN_FILES = {
    'inspection': 'camera_inspection.py',  # メインの検査システム
    'trainer': 'scripts/train_4class_sparse_ensemble.py',  # メインの学習スクリプト
    'github_commit': 'auto-commit.ps1',  # メインの自動コミットスクリプト
}

# 重複ファイル（oldフォルダに移動）
DUPLICATE_FILES = {
    'inspection': [
        'all_cameras_inspection.py',
        'camera_selector_inspection.py',
        'fundamental_fix_inspection.py',
        'fixed_inspection.py',
        'perfect_focus_restored.py',
        'focus_enhanced_inspection.py',
        'high_accuracy_inspection.py',
        'perfect_ui_inspection.py',
        'perfect_ui_inspection_fixed.py',
        'real_image_inspection.py',
        'network_washer_inspection_system.py',
        'camera_fix_inspection.py',
        'camera_startup_fixed_inspection.py',
        'correct_camera_inspection.py',
        'auto_adjust_inspection.py',
        'auto_detect_hd_inspection.py',
        'auto_high_resolution_camera_step3.py',
        'camera2_basic_inspection.py',
        'camera2_color_inspection.py',
        'camera2_force_color_inspection.py',
        'camera2_hd_inspection.py',
        'camera2_manual_color_inspection.py',
        'camera2_minimal_inspection.py',
        'camera2_second_image_ui_inspection.py',
        'camera2_super_color_inspection.py',
        'hd_color_inspection.py',
        'hd_pro_webcam_inspection.py',
        'high_resolution_camera_step2.py',
        'minimal_camera_step1.py',
        'multi_camera_selection_step4.py',
        'robust_camera_inspection.py',
        'robust_color_camera.py',
        'working_camera_inspection.py',
        'working_camera_switch_inspection.py',
    ],
    'trainer': [
        'enhanced_trainer.py',
        'improved_trainer.py',
        'improved_accuracy_trainer.py',
        'high_accuracy_trainer.py',
        'class_balanced_trainer.py',
        'precision_fix_trainer.py',
        'ultimate_trainer.py',
        'resin_washer_trainer.py',
        'retrain_improved.py',
        'retrain_from_feedback.py',
    ],
    'github_commit': [
        'test_auto_commit.ps1',  # テスト用なので残す
    ],
}

def organize_files():
    """重複ファイルをoldフォルダに移動"""
    base_path = Path('.')
    old_path = base_path / 'old' / 'duplicate_files'
    old_path.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    
    for category, files in DUPLICATE_FILES.items():
        category_path = old_path / category
        category_path.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            file_path = base_path / file
            if file_path.exists():
                dest_path = category_path / file
                try:
                    shutil.move(str(file_path), str(dest_path))
                    print(f"移動: {file} → old/duplicate_files/{category}/")
                    moved_count += 1
                except Exception as e:
                    print(f"エラー: {file} の移動に失敗: {e}")
    
    print(f"\n合計 {moved_count} 個のファイルを移動しました。")
    print(f"\nメインファイル:")
    for category, main_file in MAIN_FILES.items():
        print(f"  {category}: {main_file}")

if __name__ == '__main__':
    print("重複ファイル整理スクリプト")
    print("=" * 60)
    print("\nこのスクリプトは以下のファイルをold/duplicate_files/に移動します:")
    for category, files in DUPLICATE_FILES.items():
        print(f"\n{category}:")
        for file in files[:5]:  # 最初の5つだけ表示
            print(f"  - {file}")
        if len(files) > 5:
            print(f"  ... 他 {len(files) - 5} 個")
    
    response = input("\n実行しますか? (y/n): ")
    if response.lower() == 'y':
        organize_files()
    else:
        print("キャンセルしました。")

