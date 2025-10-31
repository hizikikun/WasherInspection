#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全ファイルとフォルダーの整理スクリプト
"""

import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def create_folders():
    """必要なフォルダーを作成"""
    folders = [
        'models',
        'models/ensemble',
        'models/sparse',
        'models/corrected',
        'logs/training',
        'logs/training/sparse',
        'logs/training/corrected',
        'temp',
        'docs/hwinfo',
        'scripts/training',
        'scripts/utils',
        'scripts/git',
        'scripts/hwinfo',
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"✓ フォルダー作成完了: {len(folders)}個")

def organize_models():
    """モデルファイル（.h5）を整理"""
    root = Path('.')
    models_dir = Path('models')
    
    model_patterns = {
        'models/sparse': [
            'sparse_ensemble_4class_model_*.h5',
            'sparse_best_4class_efficientnet*.h5',
            'clear_sparse_ensemble_4class_model_*.h5',
            'clear_sparse_best_4class_efficientnet*.h5',
            'retrain_sparse_ensemble_4class_model_*.h5',
        ],
        'models/corrected': [
            'corrected_ensemble_4class_model_*.h5',
            'corrected_best_4class_efficientnet*.h5',
        ],
        'models/ensemble': [
            'ensemble_4class_model_*.h5',
            'ensemble_model_*.h5',
            'best_4class_efficientnet*.h5',
            'best_efficientnet*.h5',
        ],
    }
    
    moved_count = 0
    for folder, patterns in model_patterns.items():
        target_dir = Path(folder)
        for pattern in patterns:
            for file_path in root.glob(pattern):
                try:
                    dest = target_dir / file_path.name
                    if dest.exists():
                        # 重複ファイルは日時を追加
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        dest = target_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
                    file_path.rename(dest)
                    moved_count += 1
                    print(f"  Moved: {file_path.name} -> {folder}/")
                except Exception as e:
                    print(f"  Error moving {file_path.name}: {e}")
    
    print(f"✓ モデルファイル整理: {moved_count}個")

def organize_logs():
    """ログファイル（.csv, .json）を整理"""
    root = Path('.')
    logs_dir = Path('logs/training')
    
    log_patterns = {
        'logs/training/sparse': [
            'sparse_training_log_*.csv',
            'clear_sparse_training_log_*.csv',
        ],
        'logs/training/corrected': [
            'corrected_training_log_*.csv',
        ],
        'logs/training': [
            'training_log_*.csv',
            '*_info.json',
        ],
    }
    
    moved_count = 0
    for folder, patterns in log_patterns.items():
        target_dir = Path(folder)
        for pattern in patterns:
            for file_path in root.glob(pattern):
                try:
                    dest = target_dir / file_path.name
                    if dest.exists():
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        dest = target_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
                    file_path.rename(dest)
                    moved_count += 1
                    print(f"  Moved: {file_path.name} -> {folder}/")
                except Exception as e:
                    print(f"  Error moving {file_path.name}: {e}")
    
    # logsフォルダー内の既存ファイルも整理
    if Path('logs').exists():
        for file_path in Path('logs').glob('*.json'):
            if file_path.name != 'training_status.json':
                try:
                    dest = logs_dir / file_path.name
                    if not dest.exists():
                        file_path.rename(dest)
                        moved_count += 1
                except Exception:
                    pass
    
    print(f"✓ ログファイル整理: {moved_count}個")

def organize_docs():
    """ドキュメントファイルを整理"""
    root = Path('.')
    docs_dir = Path('docs')
    
    doc_files = [
        'AUTO_COMMIT_README.md',
        'CURSOR_GITHUB_GUIDE.md',
        'GITHUB_AUTO_COMMIT_GUIDE.md',
        'GITHUB_MOJIBAKE_REPORT.md',
        'NETWORK_SETUP_GUIDE.md',
        'PROJECT_STRUCTURE.md',
        'README_ORGANIZED.md',
    ]
    
    # scripts内のHWiNFO関連ドキュメントも移動
    scripts_dir = Path('scripts')
    hwinfo_docs_dir = docs_dir / 'hwinfo'
    
    moved_count = 0
    for doc_file in doc_files:
        file_path = root / doc_file
        if file_path.exists():
            try:
                dest = docs_dir / file_path.name
                if not dest.exists():
                    file_path.rename(dest)
                    moved_count += 1
                    print(f"  Moved: {doc_file} -> docs/")
            except Exception as e:
                print(f"  Error moving {doc_file}: {e}")
    
    # scripts内のHWiNFOドキュメント
    hwinfo_doc_patterns = [
        'HWiNFO_*.md',
        'README_HWiNFO*.txt',
    ]
    
    for pattern in hwinfo_doc_patterns:
        for file_path in scripts_dir.glob(pattern):
            try:
                dest = hwinfo_docs_dir / file_path.name
                if not dest.exists():
                    file_path.rename(dest)
                    moved_count += 1
                    print(f"  Moved: {file_path.name} -> docs/hwinfo/")
            except Exception as e:
                print(f"  Error moving {file_path.name}: {e}")
    
    print(f"✓ ドキュメント整理: {moved_count}個")

def organize_scripts():
    """scriptsフォルダー内のファイルを分類整理"""
    scripts_dir = Path('scripts')
    
    categories = {
        'scripts/training': [
            'train_*.py',
            'advanced_optimization_trainer.py',
            'simple_enhanced_trainer.py',
            'ultra_high_performance_trainer.py',
        ],
        'scripts/utils': [
            'file_organizer.py',
            'github_organizer.py',
            'system_detector.py',
            'auto_recovery_watcher.py',
            'training_health_checker.py',
        ],
        'scripts/git': [
            '*git*.py',
            '*commit*.py',
            '*push*.py',
            '*mojibake*.py',
            '*fix*.py',
            'safe_git_commit.py',
        ],
        'scripts/hwinfo': [
            'hwinfo_*.py',
            'hwinfo_*.bat',
            'hwinfo_*.ps1',
        ],
    }
    
    moved_count = 0
    for category_dir, patterns in categories.items():
        target_dir = Path(category_dir)
        for pattern in patterns:
            for file_path in scripts_dir.glob(pattern):
                # メインファイルは移動しない
                if file_path.name in ['train_4class_sparse_ensemble.py']:
                    continue
                try:
                    dest = target_dir / file_path.name
                    if not dest.exists():
                        file_path.rename(dest)
                        moved_count += 1
                        print(f"  Moved: {file_path.name} -> {category_dir}/")
                except Exception as e:
                    print(f"  Error moving {file_path.name}: {e}")
    
    print(f"✓ スクリプト整理: {moved_count}個")

def remove_temp_files():
    """一時ファイルとデバッグファイルを削除または移動"""
    root = Path('.')
    temp_dir = Path('temp')
    
    temp_patterns = [
        'test_*.py',
        'test_*.txt',
        'test_*.ps1',
        'test_*.log',
        'debug_output.txt',
        'encoding-test.log',
        'commit_list.txt',
        '*.tmp',
        '*.bak',
        '*.old',
    ]
    
    removed_count = 0
    for pattern in temp_patterns:
        for file_path in root.glob(pattern):
            try:
                # 重要なファイルは除外
                if 'test_hwinfo' not in file_path.name:
                    file_path.unlink()
                    removed_count += 1
                    print(f"  Removed: {file_path.name}")
            except Exception as e:
                print(f"  Error removing {file_path.name}: {e}")
        
        # scripts内もチェック
        for file_path in Path('scripts').glob(pattern):
            if file_path.name.startswith('hwinfo_'):
                continue
            try:
                if file_path.suffix in ['.txt', '.log']:
                    file_path.unlink()
                    removed_count += 1
                    print(f"  Removed: {file_path.name}")
            except Exception as e:
                print(f"  Error removing {file_path.name}: {e}")
    
    print(f"✓ 一時ファイル削除: {removed_count}個")

def organize_root_files():
    """ルートディレクトリのファイルを整理"""
    root = Path('.')
    
    # カメラ関連スクリプトをinspectorsに移動
    camera_files = [
        'camera2_color_inspection.py',
        'color_camera_fix.py',
        'force_camera_window.py',
        'simple_camera.py',
        'resin_inspection_camera.py',
        'high_resolution_camera_step2.py',
    ]
    
    moved_count = 0
    inspectors_dir = Path('inspectors')
    if inspectors_dir.exists():
        for file_name in camera_files:
            file_path = root / file_name
            if file_path.exists():
                try:
                    dest = inspectors_dir / file_path.name
                    if not dest.exists():
                        file_path.rename(dest)
                        moved_count += 1
                        print(f"  Moved: {file_name} -> inspectors/")
                except Exception as e:
                    print(f"  Error moving {file_name}: {e}")
    
    # 画像ファイルをtempに移動
    image_files = list(root.glob('*.jpg')) + list(root.glob('*.png'))
    for file_path in image_files:
        try:
            dest = Path('temp') / file_path.name
            file_path.rename(dest)
            moved_count += 1
            print(f"  Moved: {file_path.name} -> temp/")
        except Exception as e:
            print(f"  Error moving {file_path.name}: {e}")
    
    print(f"✓ ルートファイル整理: {moved_count}個")

def main():
    print("=" * 60)
    print("全ファイルとフォルダーの整理")
    print("=" * 60)
    print()
    
    create_folders()
    print()
    
    organize_models()
    print()
    
    organize_logs()
    print()
    
    organize_docs()
    print()
    
    organize_scripts()
    print()
    
    remove_temp_files()
    print()
    
    organize_root_files()
    print()
    
    print("=" * 60)
    print("整理完了")
    print("=" * 60)

if __name__ == '__main__':
    main()

