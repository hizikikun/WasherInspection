#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画像分類ツール - 良品・不良品を手動で分類
"""

import os
import shutil
import glob
from pathlib import Path
import cv2

def classify_images_interactive(data_dir="cs AI学習データ"):
    """対話的に画像を分類"""
    print("=" * 60)
    print("樹脂製ワッシャー 画像分類ツール")
    print("=" * 60)
    
    # 画像ファイルを検索
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
    
    print(f"\n見つかった画像: {len(all_images)}枚")
    
    if len(all_images) == 0:
        print("画像が見つかりませんでした")
        return
    
    # 出力ディレクトリ作成
    good_dir = os.path.join(data_dir, "good")
    defect_dir = os.path.join(data_dir, "defect")
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(defect_dir, exist_ok=True)
    
    print("\n画像を表示します:")
    print("  g または 1: 良品")
    print("  d または 2: 不良品")  
    print("  s: スキップ")
    print("  q: 終了")
    
    good_count = 0
    defect_count = 0
    
    for i, img_path in enumerate(all_images):
        # 既に分類済みの画像はスキップ
        if 'good' in img_path.lower() or 'defect' in img_path.lower():
            continue
            
        print(f"\n[{i+1}/{len(all_images)}] {os.path.basename(img_path)}")
        
        # 画像を表示
        img = cv2.imread(img_path)
        if img is not None:
            # リサイズして表示
            h, w = img.shape[:2]
            if w > 800:
                scale = 800 / w
                img = cv2.resize(img, (int(w*scale), int(h*scale)))
            
            cv2.imshow("Image Classification", img)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            
            if key == ord('g') or key == ord('1'):
                # 良品
                good_count += 1
                new_name = f"good_{good_count:03d}{Path(img_path).suffix}"
                new_path = os.path.join(good_dir, new_name)
                shutil.copy2(img_path, new_path)
                print(f"  -> 良品: {new_name}")
                
            elif key == ord('d') or key == ord('2'):
                # 不良品
                defect_count += 1
                new_name = f"defect_{defect_count:03d}{Path(img_path).suffix}"
                new_path = os.path.join(defect_dir, new_name)
                shutil.copy2(img_path, new_path)
                print(f"  -> 不良品: {new_name}")
                
            elif key == ord('s'):
                print("  -> スキップ")
                continue
                
            elif key == ord('q'):
                print("\n分類を終了します")
                break
    
    print("\n" + "=" * 60)
    print("分類完了")
    print(f"良品: {good_count}枚")
    print(f"不良品: {defect_count}枚")
    print(f"保存先: {data_dir}")
    print("=" * 60)

if __name__ == "__main__":
    classify_images_interactive("cs AI学習データ")









