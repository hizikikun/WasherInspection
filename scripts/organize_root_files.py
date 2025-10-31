#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ルートディレクトリのファイルを整理
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
        'scripts/config',
        'scripts/legacy',
        'batch',
        'config',
        'tools',
        'docs/setup',
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"✓ フォルダー作成完了: {len(folders)}個")

def organize_python_files():
    """Pythonスクリプトを整理"""
    root = Path('.')
    
    file_mapping = {
        'scripts/legacy': [
            'correct_path_deep_learning.py',
            'correct_path_defect_classifier.py',
            'create_dummy_model.py',
            'direct_sample_deep_learning.py',
            'existing_data_defect_classifier.py',
            'image_data_deep_learning.py',
            'no_visualization_deep_learning.py',
            'real_image_deep_learning.py',
            'real_path_defect_classifier.py',
            'retrain_from_feedback.py',
            'retrain_improved.py',
            'resin_washer_deep_learning.py',
            'resin_washer_dl.py',
            'resin_washer_trainer.py',
        ],
        'scripts/config': [
            'check_and_train.py',
            'organize_files.py',
            'organize_github_files.py',
            'path_diagnosis.py',
            'rename_files.py',
            'rename_remaining_files.py',
        ],
        'tools': [
            'gh_cli_token_creator.py',
            'simple_token_setup.py',
            'unified_token_creator.py',
            'fix_commit_messages.py',
        ],
        'inspectors': [
            'manual_black_spot_detection.py',
            'interactive_learning_system.py',
        ],
    }
    
    moved_count = 0
    for target_dir, files in file_mapping.items():
        target_path = Path(target_dir)
        for file_name in files:
            file_path = root / file_name
            if file_path.exists():
                try:
                    dest = target_path / file_name
                    if not dest.exists():
                        file_path.rename(dest)
                        moved_count += 1
                        print(f"  Moved: {file_name} -> {target_dir}/")
                except Exception as e:
                    print(f"  Error moving {file_name}: {e}")
    
    print(f"✓ Pythonスクリプト整理: {moved_count}個")

def organize_batch_files():
    """バッチファイルを整理"""
    root = Path('.')
    batch_dir = Path('batch')
    
    batch_files = [
        'auto_create_token.bat',
        'force_commit.bat',
        'start_auto.bat',
        'start_cursor_github.bat',
        'start_cursor_gui.bat',
        'start_desktop_server.bat',
        'start_notebook_client.bat',
        'start-auto-commit.bat',
    ]
    
    moved_count = 0
    for file_name in batch_files:
        file_path = root / file_name
        if file_path.exists():
            try:
                dest = batch_dir / file_name
                if not dest.exists():
                    file_path.rename(dest)
                    moved_count += 1
                    print(f"  Moved: {file_name} -> batch/")
            except Exception as e:
                print(f"  Error moving {file_name}: {e}")
    
    # batch_filesフォルダー内のファイルも整理
    if Path('batch_files').exists():
        for file_path in Path('batch_files').glob('*.bat'):
            try:
                dest = batch_dir / file_path.name
                if not dest.exists():
                    file_path.rename(dest)
                    moved_count += 1
                    print(f"  Moved: {file_path.name} -> batch/")
            except Exception as e:
                print(f"  Error moving {file_path.name}: {e}")
    
    print(f"✓ バッチファイル整理: {moved_count}個")

def organize_config_files():
    """設定ファイルを整理"""
    root = Path('.')
    config_dir = Path('config')
    
    config_files = [
        'github_auto_commit_config.json',
        'network_config.json',
        'setup-auto-commit.xml',
    ]
    
    moved_count = 0
    for file_name in config_files:
        file_path = root / file_name
        if file_path.exists():
            try:
                dest = config_dir / file_name
                if not dest.exists():
                    file_path.rename(dest)
                    moved_count += 1
                    print(f"  Moved: {file_name} -> config/")
            except Exception as e:
                print(f"  Error moving {file_name}: {e}")
    
    # 既存のconfigフォルダー内のファイルも確認
    if Path('config').exists():
        # 重複チェック
        for file_path in Path('config').glob('*.json'):
            if file_path.name in ['class_names.json', 'training_statistics.json']:
                continue  # 既に適切な場所にある
    
    print(f"✓ 設定ファイル整理: {moved_count}個")

def organize_docs():
    """ドキュメントを整理"""
    root = Path('.')
    docs_dir = Path('docs')
    
    doc_files = [
        'PROJECT_STRUCTURE.md',
    ]
    
    # PowerShellスクリプトはdocs/setupに
    ps_files = [
        'auto-commit.ps1',
        'fix_all_commits.ps1',
        'test-encoding.ps1',
        'test-japanese.ps1',
    ]
    
    moved_count = 0
    
    # Markdownファイル
    for file_name in doc_files:
        file_path = root / file_name
        if file_path.exists():
            try:
                dest = docs_dir / file_name
                if not dest.exists():
                    file_path.rename(dest)
                    moved_count += 1
                    print(f"  Moved: {file_name} -> docs/")
            except Exception as e:
                print(f"  Error moving {file_name}: {e}")
    
    # PowerShellスクリプト
    setup_dir = docs_dir / 'setup'
    for file_name in ps_files:
        file_path = root / file_name
        if file_path.exists():
            try:
                dest = setup_dir / file_name
                if not dest.exists():
                    file_path.rename(dest)
                    moved_count += 1
                    print(f"  Moved: {file_name} -> docs/setup/")
            except Exception as e:
                print(f"  Error moving {file_name}: {e}")
    
    print(f"✓ ドキュメントテスクリプト整理: {moved_count}個")

def organize_data_files():
    """データファイルを整理"""
    root = Path('.')
    temp_dir = Path('temp')
    
    # テストファイル
    test_files = [
        'test-file.txt',
    ]
    
    moved_count = 0
    for file_name in test_files:
        file_path = root / file_name
        if file_path.exists():
            try:
                dest = temp_dir / file_name
                if not dest.exists():
                    file_path.rename(dest)
                    moved_count += 1
                    print(f"  Moved: {file_name} -> temp/")
            except Exception as e:
                print(f"  Error moving {file_name}: {e}")
    
    # 孤立したモデルファイル（既にmodels/にある場合）
    isolated_models = [
        'best_real_defect_model.h5',
    ]
    
    for file_name in isolated_models:
        file_path = root / file_name
        if file_path.exists():
            try:
                dest = Path('models') / 'legacy' / file_name
                Path('models/legacy').mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    file_path.rename(dest)
                    moved_count += 1
                    print(f"  Moved: {file_name} -> models/legacy/")
            except Exception as e:
                print(f"  Error moving {file_name}: {e}")
    
    print(f"✓ データファイル整理: {moved_count}個")

def organize_spec_files():
    """specファイルを整理"""
    root = Path('.')
    
    spec_files = list(root.glob('*.spec'))
    
    moved_count = 0
    for file_path in spec_files:
        try:
            # buildディレクトリに移動（PyInstaller specファイル）
            dest_dir = Path('build')
            if dest_dir.exists():
                dest = dest_dir / file_path.name
                if not dest.exists():
                    file_path.rename(dest)
                    moved_count += 1
                    print(f"  Moved: {file_path.name} -> build/")
        except Exception as e:
            print(f"  Error moving {file_path.name}: {e}")
    
    print(f"✓ specファイル整理: {moved_count}個")

def main():
    print("=" * 60)
    print("ルートディレクトリのファイル整理")
    print("=" * 60)
    print()
    
    create_folders()
    print()
    
    organize_python_files()
    print()
    
    organize_batch_files()
    print()
    
    organize_config_files()
    print()
    
    organize_docs()
    print()
    
    organize_data_files()
    print()
    
    organize_spec_files()
    print()
    
    print("=" * 60)
    print("整理完了")
    print("=" * 60)
    
    # 残っているファイルを確認
    print()
    print("ルートディレクトリに残っているファイル:")
    root = Path('.')
    important_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'workspace.code-workspace',
        'camera_history.json',
        'feedback_data.json',
    ]
    
    remaining = []
    for item in root.iterdir():
        if item.is_file() and item.name not in important_files:
            if item.suffix in ['.py', '.bat', '.ps1', '.json', '.md', '.spec', '.txt', '.h5']:
                remaining.append(item.name)
    
    if remaining:
        print("  以下のファイルが残っていまム（手動で確認してください）:")
        for name in sorted(remaining):
            print(f"    - {name}")
    else:
        print("  ✓ すべて整理済み")

if __name__ == '__main__':
    main()

