#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Organizer
GitHub用のファイル整理と文字化け修正スクリプト
"""

import os
import shutil
import subprocess
import sys
# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
from pathlib import Path
import json

class GitHubOrganizer:
    def __init__(self):
        self.root_dir = Path(".")
        self.backup_dir = Path("backup/github_organization")
        
    def create_backup(self):
        """バックアップディレクトリを作成"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"バックアップディレクトリを作成: {self.backup_dir}")
    
    def fix_encoding_issues(self):
        """文字化けを修正"""
        print("=" * 80)
        print("文字化け修正開始")
        print("=" * 80)
        
        # 文字化け修正対象ファイル
        files_to_fix = [
            "AUTO_COMMIT_README.md",
            "GITHUB_MOJIBAKE_REPORT.md",
            "docs/GITHUB_MOJIBAKE_STATUS.md",
            "test-file.txt",
            "test-japanese.ps1"
        ]
        
        for file_path in files_to_fix:
            if os.path.exists(file_path):
                try:
                    # バックアップ作成
                    backup_path = self.backup_dir / Path(file_path).name
                    shutil.copy2(file_path, backup_path)
                    
                    # ファイル読み込み（複数エンコーディングを試行）
                    content = None
                    for encoding in ['utf-8', 'cp932', 'shift_jis', 'euc-jp']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            print(f"読み込み成功: {file_path} ({encoding})")
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if content is None:
                        print(f"警告: {file_path} の読み込みに失敗")
                        continue
                    
                    # UTF-8で保存
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"修正完了: {file_path}")
                    
                except Exception as e:
                    print(f"エラー: {file_path} の修正に失敗: {e}")
    
    def add_new_files(self):
        """新しいファイルをGitに追加"""
        print("\n" + "=" * 80)
        print("新しいファイルをGitに追加")
        print("=" * 80)
        
        # 新しいディレクトリとファイルを追加
        new_items = [
            "trainers/",
            "inspectors/", 
            "utilities/",
            "README_ORGANIZED.md",
            "dashboard/"
        ]
        
        for item in new_items:
            if os.path.exists(item):
                try:
                    subprocess.run(["git", "add", item], check=True)
                    print(f"追加: {item}")
                except subprocess.CalledProcessError as e:
                    print(f"エラー: {item} の追加に失敗: {e}")
    
    def remove_deleted_files(self):
        """削除されたファイルをGitから削除"""
        print("\n" + "=" * 80)
        print("削除されたファイルをGitから削除")
        print("=" * 80)
        
        # 削除されたファイルのリスト
        deleted_files = [
            "all_cameras_inspection.py",
            "auto_adjust_inspection.py",
            "auto_detect_hd_inspection.py",
            "auto_high_resolution_camera_step3.py",
            "balanced_resin_washer_model.h5",
            "best_spatial_model.h5",
            "camera2_basic_inspection.py",
            "camera2_force_color_inspection.py",
            "camera2_hd_inspection.py",
            "camera2_manual_color_inspection.py",
            "camera2_minimal_inspection.py",
            "camera2_second_image_ui_inspection.py",
            "camera2_super_color_inspection.py",
            "camera_fix_inspection.py",
            "camera_inspection.py",
            "camera_selector_inspection.py",
            "camera_startup_fixed_inspection.py",
            "check_cameras.py",
            "check_data_structure.py",
            "check_image_count.py",
            "class_balanced_trainer.py",
            "classify_images.py",
            "correct_camera_inspection.py",
            "debug_realtime_inspection.py",
            "desktop_app_inspection.py",
            "enhanced_trainer.py",
            "fixed_inspection.py",
            "focus_enhanced_inspection.py",
            "fundamental_fix_inspection.py",
            "guaranteed_camera.py",
            "hardware_level_inspection.py",
            "hd_color_inspection.py",
            "hd_pro_webcam_inspection.py",
            "high_accuracy_inspection.py",
            "high_accuracy_trainer.py",
            "improve_accuracy.py",
            "improve_model.py",
            "improved_accuracy_trainer.py",
            "improved_trainer.py",
            "interactive_camera.py",
            "minimal_camera_step1.py",
            "model_performance_analyzer.py",
            "multi_camera_selection_step4.py",
            "network_washer_inspection_system.py",
            "original_ui_inspection.py",
            "perfect_focus_restored.py",
            "perfect_ui_inspection.py",
            "perfect_ui_inspection_fixed.py",
            "persistent_camera.py",
            "precision_fix_trainer.py",
            "real_image_inspection.py",
            "real_working_camera_inspection.py",
            "realtime_inspection.py",
            "realtime_inspection_fixed.py",
            "robust_camera_inspection.py",
            "robust_color_camera.py",
            "sensitive_defect_detection.py",
            "test_model.py",
            "ultimate_stable_detection.py",
            "ultimate_trainer.py",
            "working_camera_inspection.py",
            "working_camera_switch_inspection.py"
        ]
        
        for file_path in deleted_files:
            try:
                subprocess.run(["git", "rm", file_path], check=True)
                print(f"削除: {file_path}")
            except subprocess.CalledProcessError:
                # ファイルが既に削除されている場合は無視
                pass
    
    def commit_changes(self):
        """変更をコミット"""
        print("\n" + "=" * 80)
        print("変更をコミット")
        print("=" * 80)
        
        try:
            # すべての変更をステージング
            subprocess.run(["git", "add", "-A"], check=True)
            
            # コミットメッセージ
            commit_message = "ファイル整理とリネーム: 分かりやムい名前に変更、ディレクトリ構造を整理"
            
            # コミット実行
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            print("コミット完了")
            
        except subprocess.CalledProcessError as e:
            print(f"コミットエラー: {e}")
    
    def push_to_github(self):
        """GitHubにプッシュ"""
        print("\n" + "=" * 80)
        print("GitHubにプッシュ")
        print("=" * 80)
        
        try:
            # プッシュ実行
            subprocess.run(["git", "push", "origin", "main"], check=True)
            print("GitHubプッシュ完了")
            
        except subprocess.CalledProcessError as e:
            print(f"プッシュエラー: {e}")
            print("手動でプッシュしてください: git push origin main")
    
    def create_github_readme(self):
        """GitHub用のREADMEを作成"""
        print("\n" + "=" * 80)
        print("GitHub用README作成")
        print("=" * 80)
        
        readme_content = """# ワッシャー不良品検出システム

## 概要
AIを活用した樹脂製ワッシャーの不良品検出システムです。6種類の不良品（良品、黒点、欠け、傷、歪み、凹み）を高精度で分類できまム。

## 特徴
- **6クラス分類**: 良品、黒点、欠け、傷、歪み、凹みを検出
- **リアルタイム検査**: カメラからの即座な判定
- **高精度学習**: EfficientNetベースのアンサンブル学習
- **使いやムいUI**: 直感的な操作インターフェース

## ディレクトリ構造

### 📁 trainers/ - 学習システム
- `six_class_trainer.py` - 6クラス学習システム（推奨）
- `high_quality_trainer.py` - 高品質学習システム
- `optimized_trainer.py` - 最適化学習システム
- `basic_trainer.py` - 基本学習システム

### 📁 inspectors/ - 検査システム
- `six_class_inspector.py` - 6クラス検査システム（推奨）
- `camera_inspector.py` - カメラ検査システム
- `realtime_inspector.py` - リアルタイム検査システム
- `multi_camera_inspector.py` - 複数カメラ検査システム

### 📁 utilities/ - ユーティリティ
- `install_dependencies.py` - 依存関係インストール
- `run_training.py` - 学習実行スクリプト
- `generate_samples.py` - サンプルデータ生成
- `system_checker.py` - システムチェック

## セットアップ

### 1. 依存関係のインストール
```bash
python utilities/install_dependencies.py
```

### 2. サンプルデータの生成
```bash
python utilities/generate_samples.py
```

## 使用方法

### 学習の実行
```bash
# 6クラス学習（推奨）
python trainers/six_class_trainer.py

# 高品質学習
python trainers/high_quality_trainer.py
```

### 検査の実行
```bash
# 6クラス検査（推奨）
python inspectors/six_class_inspector.py --camera

# 単一画像検査
python inspectors/six_class_inspector.py image.jpg

# 一括検査
python inspectors/six_class_inspector.py --batch /path/to/images/
```

## 検出可能な不良品
1. **良品 (good)** - 正常なワッシャー
2. **黒点 (black_spot)** - 黒い点状の欠陥
3. **欠け (chipping)** - 破損テ欠損
4. **傷 (scratch)** - 表面の傷
5. **歪み (distortion)** - 形状の歪み
6. **凹み (dent)** - 表面の凹み

## 技術仕様
- **フレームワーク**: TensorFlow/Keras
- **モデル**: EfficientNet (B0, B1, B2)
- **学習方式**: アンサンブル学習
- **データ拡張**: 高度な画像変換
- **最適化**: AdamW, 学習率スケジューリング

## ライセンス
MIT License

## 貢献
プルリクエストやイシューの報告を歓迎します。

## 更新履歴
- v2.0: 6クラス対応、ファイル整理、UI改善
- v1.0: 基本4クラス検出システム
"""
        
        readme_path = self.root_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"GitHub用README作成完了: {readme_path}")
    
    def run_organization(self):
        """全体の整理を実行"""
        print("=" * 80)
        print("GitHub用ファイル整理開始")
        print("=" * 80)
        
        try:
            # バックアップ作成
            self.create_backup()
            
            # 文字化け修正
            self.fix_encoding_issues()
            
            # 新しいファイルを追加
            self.add_new_files()
            
            # 削除されたファイルを削除
            self.remove_deleted_files()
            
            # GitHub用README作成
            self.create_github_readme()
            
            # 変更をコミット
            self.commit_changes()
            
            # GitHubにプッシュ
            self.push_to_github()
            
            print("\n" + "=" * 80)
            print("GitHub整理完了!")
            print("=" * 80)
            print("バックアップ: backup/github_organization/")
            print("GitHubにプッシュ完了")
            print("=" * 80)
            
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()

def main():
    organizer = GitHubOrganizer()
    organizer.run_organization()

if __name__ == "__main__":
    main()
