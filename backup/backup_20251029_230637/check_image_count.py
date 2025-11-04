#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画像ファイルの正確な枚数を確認するスクリプト
"""

import os
import glob
from pathlib import Path

def check_image_count():
    """各フォルダーの画像枚数を正確に確認"""
    print("=" * 60)
    print("画像ファイル枚数確認")
    print("=" * 60)
    
    data_dir = Path("cs AI学習データ/樹脂")
    
    if not data_dir.exists():
        print(f"エラー: フォルダーが見つかりません: {data_dir}")
        return
    
    # 各カテゴリのフォルダー
    categories = {
        '良品(樹脂)': ['樹脂20251015(良品)', '良品', 'good'],
        '欠け樹脂': [],
        '黒点樹脂': [],
        '傷樹脂': []
    }
    
    total_images = 0
    
    for category_name, subfolders in categories.items():
        category_dir = data_dir / category_name
        if not category_dir.exists():
            print(f"[NG] {category_name}: フォルダーが見つかりません")
            continue
        
        category_count = 0
        found_files = set()  # 重複を避けるため
        
        # サブフォルダーがある場合
        if subfolders:
            for subfolder in subfolders:
                subfolder_path = category_dir / subfolder
                if subfolder_path.exists():
                    print(f"  サブフォルダー: {subfolder}")
                    
                    # 画像ファイルを検索
                    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
                    for ext in image_extensions:
                        files = glob.glob(str(subfolder_path / ext))
                        for file_path in files:
                            if file_path not in found_files:
                                found_files.add(file_path)
                                category_count += 1
        else:
            # 直接フォルダー内の画像を検索
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
            for ext in image_extensions:
                files = glob.glob(str(category_dir / ext))
                for file_path in files:
                    if file_path not in found_files:
                        found_files.add(file_path)
                        category_count += 1
        
        print(f"[OK] {category_name}: {category_count}枚")
        total_images += category_count
        
        # 最初の5ファイルを表示
        if found_files:
            print(f"   例: {list(found_files)[:5]}")
    
    print(f"\n合計: {total_images}枚")
    
    # ファイルの詳細情報
    print("\n" + "=" * 60)
    print("ファイル詳細情報")
    print("=" * 60)
    
    for category_name, subfolders in categories.items():
        category_dir = data_dir / category_name
        if not category_dir.exists():
            continue
            
        print(f"\nフォルダー: {category_name}")
        
        if subfolders:
            for subfolder in subfolders:
                subfolder_path = category_dir / subfolder
                if subfolder_path.exists():
                    print(f"  サブフォルダー: {subfolder}")
                    
                    # ファイル一覧を取得
                    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
                    files = []
                    for ext in image_extensions:
                        files.extend(glob.glob(str(subfolder_path / ext)))
                    
                    # 重複を除去
                    unique_files = list(set(files))
                    print(f"    総ファイル数: {len(files)}")
                    print(f"    ユニークファイル数: {len(unique_files)}")
                    
                    if len(files) != len(unique_files):
                        print(f"    [警告] 重複ファイル: {len(files) - len(unique_files)}個")
                    
                    # 最初の10ファイルを表示
                    if unique_files:
                        print(f"    ファイル例:")
                        for i, file_path in enumerate(unique_files[:10]):
                            filename = os.path.basename(file_path)
                            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                            print(f"      {i+1:2d}. {filename} ({file_size:,} bytes)")
                        
                        if len(unique_files) > 10:
                            print(f"      ... 他 {len(unique_files) - 10} ファイル")

if __name__ == "__main__":
    check_image_count()