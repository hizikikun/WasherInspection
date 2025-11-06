#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fドライブのcs\\washerinspectionに外観検査に必要なファイルをコピーするスクリプト
"""

import os
import shutil
from pathlib import Path

# ソースとターゲットパス
SOURCE_DIR = Path(r"C:\Users\tomoh\WasherInspection")
TARGET_DIR = Path(r"F:\cs\washerinspection")

def copy_directory(src, dst, ignore_patterns=None):
    """ディレクトリをコピー（既存ファイルは上書き）"""
    if ignore_patterns is None:
        ignore_patterns = []
    
    dst.mkdir(parents=True, exist_ok=True)
    
    for item in src.iterdir():
        # 無視パターンをチェック
        should_ignore = False
        for pattern in ignore_patterns:
            if pattern in str(item):
                should_ignore = True
                break
        
        if should_ignore:
            print(f"  [スキップ] {item.name}")
            continue
        
        dst_item = dst / item.name
        
        if item.is_dir():
            copy_directory(item, dst_item, ignore_patterns)
        else:
            try:
                shutil.copy2(item, dst_item)
                print(f"  [コピー] {item.name}")
            except Exception as e:
                print(f"  [エラー] {item.name}: {e}")

def copy_file(src, dst):
    """ファイルをコピー"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
        print(f"  [コピー] {src.name}")
        return True
    except Exception as e:
        print(f"  [エラー] {src.name}: {e}")
        return False

def main():
    print("=" * 60)
    print("Fドライブへのファイルコピーを開始します")
    print("=" * 60)
    print(f"ソース: {SOURCE_DIR}")
    print(f"ターゲット: {TARGET_DIR}")
    print()
    
    # ターゲットディレクトリを作成
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. メインアプリケーション
    print("\n[1/10] メインアプリケーションをコピー中...")
    dashboard_src = SOURCE_DIR / "dashboard"
    dashboard_dst = TARGET_DIR / "dashboard"
    if dashboard_src.exists():
        copy_directory(dashboard_src, dashboard_dst)
    else:
        print("  [警告] dashboard ディレクトリが見つかりません")
    
    # 2. 検査モジュール（必要なもののみ）
    print("\n[2/10] 検査モジュールをコピー中...")
    inspectors_src = SOURCE_DIR / "inspectors"
    inspectors_dst = TARGET_DIR / "inspectors"
    if inspectors_src.exists():
        # 主要な検査モジュールのみコピー
        important_inspectors = [
            "realtime_inspector.py",
            "resin_inspection_camera.py",
            "real_image_inspector.py",
            "high_accuracy_inspector.py",
            "camera_inspector.py",
            "basic_inspector.py"
        ]
        
        inspectors_dst.mkdir(parents=True, exist_ok=True)
        for inspector_file in important_inspectors:
            src_file = inspectors_src / inspector_file
            if src_file.exists():
                copy_file(src_file, inspectors_dst / inspector_file)
    else:
        print("  [警告] inspectors ディレクトリが見つかりません")
    
    # 3. モデルファイル（必要なもののみ）
    print("\n[3/10] モデルファイルをコピー中...")
    models_src = SOURCE_DIR / "models"
    models_dst = TARGET_DIR / "models"
    if models_src.exists():
        # sparse, ensemble, corrected ディレクトリをコピー
        for subdir in ["sparse", "ensemble", "corrected"]:
            src_subdir = models_src / subdir
            if src_subdir.exists():
                copy_directory(src_subdir, models_dst / subdir)
        
        # ルートにあるモデルファイル
        root_model_patterns = [
            "clear_sparse_best_4class_efficientnetb0_model.h5",
            "clear_sparse_ensemble_4class_efficientnetb0_model.h5"
        ]
        for pattern in root_model_patterns:
            src_file = SOURCE_DIR / pattern
            if src_file.exists():
                copy_file(src_file, TARGET_DIR / pattern)
    else:
        print("  [警告] models ディレクトリが見つかりません")
    
    # 4. 設定ファイル
    print("\n[4/10] 設定ファイルをコピー中...")
    config_src = SOURCE_DIR / "config"
    config_dst = TARGET_DIR / "config"
    if config_src.exists():
        # 重要な設定ファイルのみコピー
        important_configs = [
            "class_names.json",
            "inspection_settings.json"
        ]
        
        config_dst.mkdir(parents=True, exist_ok=True)
        for config_file in important_configs:
            src_file = config_src / config_file
            if src_file.exists():
                copy_file(src_file, config_dst / config_file)
    else:
        print("  [警告] config ディレクトリが見つかりません")
    
    # 5. アセットファイル
    print("\n[5/10] アセットファイルをコピー中...")
    assets_src = SOURCE_DIR / "assets"
    assets_dst = TARGET_DIR / "assets"
    if assets_src.exists():
        copy_directory(assets_src, assets_dst)
    else:
        print("  [警告] assets ディレクトリが見つかりません")
    
    # 6. ユーティリティスクリプト
    print("\n[6/10] ユーティリティスクリプトをコピー中...")
    utils_src = SOURCE_DIR / "scripts" / "utils"
    utils_dst = TARGET_DIR / "scripts" / "utils"
    if utils_src.exists():
        # 重要なユーティリティのみコピー
        important_utils = [
            "project_path.py",
            "system_detector.py"
        ]
        
        utils_dst.mkdir(parents=True, exist_ok=True)
        for util_file in important_utils:
            src_file = utils_src / util_file
            if src_file.exists():
                copy_file(src_file, utils_dst / util_file)
    else:
        print("  [警告] scripts/utils ディレクトリが見つかりません")
    
    # 7. requirements.txt
    print("\n[7/10] requirements.txtをコピー中...")
    req_file = SOURCE_DIR / "requirements.txt"
    if req_file.exists():
        copy_file(req_file, TARGET_DIR / "requirements.txt")
    else:
        print("  [警告] requirements.txt が見つかりません")
    
    # 8. logs ディレクトリ（空で作成）
    print("\n[8/10] logs ディレクトリを作成中...")
    logs_dst = TARGET_DIR / "logs"
    logs_dst.mkdir(parents=True, exist_ok=True)
    print("  [作成] logs ディレクトリ")
    
    # 9. inspection_results ディレクトリ（空で作成）
    print("\n[9/10] inspection_results ディレクトリを作成中...")
    results_dst = TARGET_DIR / "inspection_results"
    results_dst.mkdir(parents=True, exist_ok=True)
    print("  [作成] inspection_results ディレクトリ")
    
    # 10. feedback_data ディレクトリ（空で作成）
    print("\n[10/10] feedback_data ディレクトリを作成中...")
    feedback_dst = TARGET_DIR / "feedback_data"
    feedback_dst.mkdir(parents=True, exist_ok=True)
    print("  [作成] feedback_data ディレクトリ")
    
    print("\n" + "=" * 60)
    print("コピー完了！")
    print("=" * 60)
    print(f"\nターゲットディレクトリ: {TARGET_DIR}")
    print("\n次のコマンドでアプリケーションを起動できます:")
    print(f"  cd {TARGET_DIR / 'dashboard'}")
    print("  python integrated_washer_app.py")
    print("\nまたは、.pyw ファイルをダブルクリック:")
    print(f"  {TARGET_DIR / 'dashboard' / 'integrated_washer_app.pyw'}")

if __name__ == "__main__":
    main()

