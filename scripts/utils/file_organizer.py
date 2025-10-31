#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
"""
File Organizer
ファイル整理とリネームスクリプト
"""

import os
import shutil
from pathlib import Path
import json

class FileOrganizer:
    def __init__(self):
        self.root_dir = Path(".")
        self.scripts_dir = Path("scripts")
        self.backup_dir = Path("backup/old_files")
        
        # ファイル名のマッピング（旧名 -> 新名）
        self.file_mappings = {
            # 学習システム
            "ultra_high_performance_trainer.py": "high_quality_trainer.py",
            "advanced_optimization_trainer.py": "optimized_trainer.py", 
            "simple_enhanced_trainer.py": "basic_trainer.py",
            "six_class_enhanced_trainer.py": "six_class_trainer.py",
            "six_class_inspection_system.py": "six_class_inspector.py",
            
            # 検査システム
            "camera2_super_color_inspection.py": "color_inspector.py",
            "camera2_hd_inspection.py": "hd_inspector.py",
            "camera2_basic_inspection.py": "basic_inspector.py",
            "camera2_force_color_inspection.py": "forced_color_inspector.py",
            "camera2_manual_color_inspection.py": "manual_color_inspector.py",
            "camera2_minimal_inspection.py": "minimal_inspector.py",
            "camera2_second_image_ui_inspection.py": "dual_image_inspector.py",
            
            # 学習関連
            "train_2class_basic.py": "binary_trainer.py",
            "train_2class_ensemble.py": "binary_ensemble_trainer.py",
            "train_2class_v2.py": "binary_trainer_v2.py",
            "train_2class_v3.py": "binary_trainer_v3.py",
            "train_2class_with_augmentation.py": "binary_augmented_trainer.py",
            "train_4class_ensemble.py": "four_class_trainer.py",
            "train_4class_ensemble_v2.py": "four_class_trainer_v2.py",
            "train_4class_sparse_ensemble.py": "four_class_sparse_trainer.py",
            "train_with_new_defects.py": "defect_retrainer.py",
            
            # その他
            "install_advanced_dependencies.py": "install_dependencies.py",
            "run_enhanced_training.py": "run_training.py",
            "generate_distortion_dent_samples.py": "generate_samples.py",
            "system_detector.py": "system_checker.py",
            
            # 検査システム（ルートディレクトリ）
            "all_cameras_inspection.py": "multi_camera_inspector.py",
            "auto_adjust_inspection.py": "auto_adjust_inspector.py",
            "auto_detect_hd_inspection.py": "auto_hd_inspector.py",
            "auto_high_resolution_camera_step3.py": "auto_hd_camera.py",
            "camera_fix_inspection.py": "camera_fix_inspector.py",
            "camera_inspection.py": "camera_inspector.py",
            "camera_selector_inspection.py": "camera_selector.py",
            "camera_startup_fixed_inspection.py": "camera_startup.py",
            "correct_camera_inspection.py": "correct_camera_inspector.py",
            "debug_realtime_inspection.py": "debug_realtime_inspector.py",
            "desktop_app_inspection.py": "desktop_inspector.py",
            "fixed_inspection.py": "fixed_inspector.py",
            "focus_enhanced_inspection.py": "focus_inspector.py",
            "fundamental_fix_inspection.py": "fundamental_fix_inspector.py",
            "guaranteed_camera.py": "reliable_camera.py",
            "hardware_level_inspection.py": "hardware_inspector.py",
            "hd_color_inspection.py": "hd_color_inspector.py",
            "hd_pro_webcam_inspection.py": "hd_webcam_inspector.py",
            "high_accuracy_inspection.py": "high_accuracy_inspector.py",
            "interactive_camera.py": "interactive_camera_inspector.py",
            "minimal_camera_step1.py": "minimal_camera.py",
            "multi_camera_selection_step4.py": "multi_camera_selector.py",
            "network_washer_inspection_system.py": "network_inspector.py",
            "original_ui_inspection.py": "original_ui_inspector.py",
            "perfect_focus_restored.py": "focus_restored.py",
            "perfect_ui_inspection_fixed.py": "ui_inspector_fixed.py",
            "perfect_ui_inspection.py": "ui_inspector.py",
            "persistent_camera.py": "persistent_camera_inspector.py",
            "real_image_inspection.py": "real_image_inspector.py",
            "real_working_camera_inspection.py": "working_camera_inspector.py",
            "realtime_inspection_fixed.py": "realtime_inspector_fixed.py",
            "realtime_inspection.py": "realtime_inspector.py",
            "robust_camera_inspection.py": "robust_camera_inspector.py",
            "robust_color_camera.py": "robust_color_inspector.py",
            "sensitive_defect_detection.py": "sensitive_defect_inspector.py",
            "ultimate_stable_detection.py": "stable_detector.py",
            "working_camera_inspection.py": "working_inspector.py",
            "working_camera_switch_inspection.py": "camera_switch_inspector.py",
            
            # 学習関連（ルートディレクトリ）
            "class_balanced_trainer.py": "balanced_trainer.py",
            "enhanced_trainer.py": "improved_trainer.py",
            "high_accuracy_trainer.py": "accurate_trainer.py",
            "improved_accuracy_trainer.py": "accuracy_trainer.py",
            "improved_trainer.py": "better_trainer.py",
            "precision_fix_trainer.py": "precision_trainer.py",
            "ultimate_trainer.py": "final_trainer.py",
        }
        
        # 削除対象ファイル（重複テ不要）
        self.files_to_remove = [
            # 重複した学習ファイル
            "train_2class_v2.py",
            "train_2class_v3.py", 
            "train_4class_ensemble_v2.py",
            
            # 古い検査ファイル
            "camera2_force_color_inspection.py",
            "camera2_manual_color_inspection.py",
            "camera2_minimal_inspection.py",
            
            # 重複した学習ファイル
            "improved_accuracy_trainer.py",
            "improved_trainer.py",
            "enhanced_trainer.py",
            
            # 古いモデルファイル（学習済みのものは残ム）
            "balanced_resin_washer_model.h5",
            "best_spatial_model.h5",
        ]
        
        # ディレクトリ構造の整理
        self.directory_structure = {
            "trainers": [
                "binary_trainer.py",
                "binary_ensemble_trainer.py", 
                "binary_augmented_trainer.py",
                "four_class_trainer.py",
                "four_class_sparse_trainer.py",
                "six_class_trainer.py",
                "high_quality_trainer.py",
                "optimized_trainer.py",
                "basic_trainer.py",
                "defect_retrainer.py",
                "balanced_trainer.py",
                "accurate_trainer.py",
                "accuracy_trainer.py",
                "better_trainer.py",
                "precision_trainer.py",
                "final_trainer.py",
            ],
            "inspectors": [
                "camera_inspector.py",
                "multi_camera_inspector.py",
                "auto_adjust_inspector.py",
                "auto_hd_inspector.py",
                "auto_hd_camera.py",
                "camera_fix_inspector.py",
                "camera_selector.py",
                "camera_startup.py",
                "correct_camera_inspector.py",
                "debug_realtime_inspector.py",
                "desktop_inspector.py",
                "fixed_inspector.py",
                "focus_inspector.py",
                "fundamental_fix_inspector.py",
                "reliable_camera.py",
                "hardware_inspector.py",
                "hd_color_inspector.py",
                "hd_webcam_inspector.py",
                "high_accuracy_inspector.py",
                "interactive_camera_inspector.py",
                "minimal_camera.py",
                "multi_camera_selector.py",
                "network_inspector.py",
                "original_ui_inspector.py",
                "focus_restored.py",
                "ui_inspector_fixed.py",
                "ui_inspector.py",
                "persistent_camera_inspector.py",
                "real_image_inspector.py",
                "working_camera_inspector.py",
                "realtime_inspector_fixed.py",
                "realtime_inspector.py",
                "robust_camera_inspector.py",
                "robust_color_inspector.py",
                "sensitive_defect_inspector.py",
                "stable_detector.py",
                "working_inspector.py",
                "camera_switch_inspector.py",
                "six_class_inspector.py",
            ],
            "utilities": [
                "install_dependencies.py",
                "run_training.py",
                "generate_samples.py",
                "system_checker.py",
                "check_cameras.py",
                "check_data_structure.py",
                "check_image_count.py",
                "classify_images.py",
                "improve_accuracy.py",
                "improve_model.py",
                "model_performance_analyzer.py",
                "test_model.py",
            ]
        }
    
    def create_backup(self):
        """バックアップディレクトリを作成"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"バックアップディレクトリを作成: {self.backup_dir}")
    
    def rename_files(self):
        """ファイルをリネーム"""
        print("=" * 80)
        print("ファイルリネーム開始")
        print("=" * 80)
        
        renamed_count = 0
        
        for old_name, new_name in self.file_mappings.items():
            old_path = self.root_dir / old_name
            new_path = self.root_dir / new_name
            
            if old_path.exists():
                try:
                    # バックアップを作成
                    backup_path = self.backup_dir / old_name
                    shutil.copy2(old_path, backup_path)
                    
                    # リネーム実行
                    old_path.rename(new_path)
                    print(f"リネーム: {old_name} -> {new_name}")
                    renamed_count += 1
                except Exception as e:
                    print(f"エラー: {old_name} のリネームに失敗: {e}")
            else:
                print(f"警告: {old_name} が見つかりません")
        
        print(f"\nリネーム完了: {renamed_count} ファイル")
    
    def remove_duplicate_files(self):
        """重複ファイルを削除"""
        print("\n" + "=" * 80)
        print("重複ファイル削除開始")
        print("=" * 80)
        
        removed_count = 0
        
        for file_name in self.files_to_remove:
            file_path = self.root_dir / file_name
            if file_path.exists():
                try:
                    # バックアップを作成
                    backup_path = self.backup_dir / file_name
                    shutil.copy2(file_path, backup_path)
                    
                    # ファイル削除
                    file_path.unlink()
                    print(f"削除: {file_name}")
                    removed_count += 1
                except Exception as e:
                    print(f"エラー: {file_name} の削除に失敗: {e}")
            else:
                print(f"警告: {file_name} が見つかりません")
        
        print(f"\n削除完了: {removed_count} ファイル")
    
    def organize_directories(self):
        """ディレクトリ構造を整理"""
        print("\n" + "=" * 80)
        print("ディレクトリ構造整理開始")
        print("=" * 80)
        
        # 新しいディレクトリを作成
        for dir_name in self.directory_structure.keys():
            new_dir = self.root_dir / dir_name
            new_dir.mkdir(exist_ok=True)
            print(f"ディレクトリ作成: {dir_name}")
        
        # ファイルを適切なディレクトリに移動
        for category, files in self.directory_structure.items():
            target_dir = self.root_dir / category
            moved_count = 0
            
            for file_name in files:
                source_path = self.root_dir / file_name
                target_path = target_dir / file_name
                
                if source_path.exists():
                    try:
                        # バックアップを作成
                        backup_path = self.backup_dir / file_name
                        shutil.copy2(source_path, backup_path)
                        
                        # ファイル移動
                        shutil.move(str(source_path), str(target_path))
                        print(f"移動: {file_name} -> {category}/")
                        moved_count += 1
                    except Exception as e:
                        print(f"エラー: {file_name} の移動に失敗: {e}")
            
            print(f"{category}: {moved_count} ファイル移動完了")
    
    def update_file_contents(self):
        """ファイル内容の表現を更新"""
        print("\n" + "=" * 80)
        print("ファイル内容更新開始")
        print("=" * 80)
        
        # 更新する表現のマッピング
        content_replacements = {
            "Ultra-High Performance": "高品質",
            "Advanced Optimization": "最適化",
            "Simple Enhanced": "基本",
            "Six-Class Enhanced": "6クラス",
            "Ultra-advanced": "高度な",
            "Ultra-high": "高品質",
            "Advanced": "高度な",
            "Enhanced": "改良",
            "Perfect": "完璧な",
            "Ultimate": "最終",
            "Super": "高機能",
            "Ultra": "高品質",
            "Sparse": "スパース",
            "Clear": "明確な",
            "Pro": "プロ",
            "Ultra-High": "高品質",
            "Advanced Spatial": "高度な空間",
            "Ultra-advanced": "高度な",
        }
        
        # 更新対象ファイル
        target_files = []
        for category, files in self.directory_structure.items():
            for file_name in files:
                file_path = self.root_dir / category / file_name
                if file_path.exists():
                    target_files.append(file_path)
        
        updated_count = 0
        for file_path in target_files:
            try:
                # ファイル読み込み
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 内容を更新
                original_content = content
                for old_text, new_text in content_replacements.items():
                    content = content.replace(old_text, new_text)
                
                # 変更があった場合のみ保存
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"更新: {file_path.name}")
                    updated_count += 1
                    
            except Exception as e:
                print(f"エラー: {file_path} の更新に失敗: {e}")
        
        print(f"\n内容更新完了: {updated_count} ファイル")
    
    def create_readme(self):
        """整理後のREADMEを作成"""
        print("\n" + "=" * 80)
        print("README作成開始")
        print("=" * 80)
        
        readme_content = """# ワッシャー不良品検出システム

## ディレクトリ構造

### trainers/ - 学習システム
- `six_class_trainer.py` - 6クラス学習システム（推奨）
- `four_class_trainer.py` - 4クラス学習システム
- `binary_trainer.py` - 2クラス学習システム
- `high_quality_trainer.py` - 高品質学習システム
- `optimized_trainer.py` - 最適化学習システム
- `basic_trainer.py` - 基本学習システム

### inspectors/ - 検査システム
- `six_class_inspector.py` - 6クラス検査システム（推奨）
- `camera_inspector.py` - カメラ検査システム
- `realtime_inspector.py` - リアルタイム検査システム
- `multi_camera_inspector.py` - 複数カメラ検査システム

### utilities/ - ユーティリティ
- `install_dependencies.py` - 依存関係インストール
- `run_training.py` - 学習実行スクリプト
- `generate_samples.py` - サンプルデータ生成
- `system_checker.py` - システムチェック

## 使用方法

### 学習の実行
```bash
# 6クラス学習（推奨）
python trainers/six_class_trainer.py

# 4クラス学習
python trainers/four_class_trainer.py
```

### 検査の実行
```bash
# 6クラス検査（推奨）
python inspectors/six_class_inspector.py --camera

# 単一画像検査
python inspectors/six_class_inspector.py image.jpg
```

## 検出可能な不良品
1. 良品 (good)
2. 黒点 (black_spot)
3. 欠け (chipping)
4. 傷 (scratch)
5. 歪み (distortion)
6. 凹み (dent)
"""
        
        readme_path = self.root_dir / "README_ORGANIZED.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"README作成完了: {readme_path}")
    
    def run_organization(self):
        """全体の整理を実行"""
        print("=" * 80)
        print("ファイル整理開始")
        print("=" * 80)
        
        try:
            # バックアップ作成
            self.create_backup()
            
            # ファイルリネーム
            self.rename_files()
            
            # 重複ファイル削除
            self.remove_duplicate_files()
            
            # ディレクトリ整理
            self.organize_directories()
            
            # ファイル内容更新
            self.update_file_contents()
            
            # README作成
            self.create_readme()
            
            print("\n" + "=" * 80)
            print("ファイル整理完了!")
            print("=" * 80)
            print("バックアップ: backup/old_files/")
            print("整理されたファイル: trainers/, inspectors/, utilities/")
            print("詳細: README_ORGANIZED.md")
            print("=" * 80)
            
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()

def main():
    organizer = FileOrganizer()
    organizer.run_organization()

if __name__ == "__main__":
    main()
