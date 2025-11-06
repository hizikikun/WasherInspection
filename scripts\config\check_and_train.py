#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
樹脂製ワッシャー データ確認と学習システム
"""

import os
import glob
from pathlib import Path

def check_dataset(data_dir="cs AI学習データ"):
    """データセットの状況を確認"""
    print("=" * 60)
    print("樹脂製ワッシャー データ確認")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        print(f"エラー: フォルダー '{data_dir}' が見つかりません")
        return False
    
    # 画像ファイルを検索
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
    
    print(f"\nフォルダー: {data_dir}")
    print(f"見つかった画像数: {len(all_images)}枚")
    
    if len(all_images) == 0:
        print("\n【画像が見つかりませんでした】")
        print("以下を確認してください:")
        print("1. 画像ファイルがフォルダー内にあるか")
        print("2. ファイル形式が .jpg, .jpeg, .png, .bmp か")
        print("3. フォルダー名が正しいか")
        return False
    
    # ファイル名からラベルを分類
    good_images = []
    defect_images = []
    unknown_images = []
    
    for img_path in all_images:
        filename = os.path.basename(img_path).lower()
        
        if 'good' in filename or '良品' in filename or 'ok' in filename:
            good_images.append(img_path)
        elif any(word in filename for word in ['defect', 'ng', '不良', 'bad', 'crack', 'black', 'spot']):
            defect_images.append(img_path)
        else:
            unknown_images.append(img_path)
    
    print("\n【データ分類】")
    print(f"良品: {len(good_images)}枚")
    print(f"不良品: {len(defect_images)}枚")
    print(f"分類不明: {len(unknown_images)}枚")
    
    # サンプル表示
    if good_images:
        print(f"\n良品サンプル: {os.path.basename(good_images[0])}")
    if defect_images:
        print(f"不良品サンプル: {os.path.basename(defect_images[0])}")
    if unknown_images:
        print(f"\n【警告】分類不明の画像があります:")
        for img in unknown_images[:5]:
            print(f"  - {os.path.basename(img)}")
    
    # データ量評価
    print("\n【データ量評価】")
    total = len(good_images) + len(defect_images)
    
    if total < 100:
        print("[NG] データ量が非常に少ないです（100枚未満）")
        print("   推奨: 最低でも良品50枚、不良品50枚以上")
    elif total < 300:
        print("[注意] データ量が少なめです（100-300枚）")
        print("   推奨: 良品200枚、不良品100枚以上")
    elif total < 1000:
        print("[OK] データ量は十分です（300-1000枚）")
    else:
        print("[OK+] データ量が豊富です（1000枚以上）")
    
    # バランス評価
    if len(good_images) > 0 and len(defect_images) > 0:
        ratio = len(good_images) / len(defect_images)
        print(f"\n良品/不良品比率: {ratio:.2f}")
        
        if ratio < 0.5 or ratio > 3:
            print("[注意] データのバランスが悪いです")
            print("   推奨: 良品と不良品の比率は 1:2 から 2:1 程度")
        else:
            print("[OK] データバランスは良好です")
    
    # 具体的な推奨事項
    print("\n【推奨事項】")
    
    if len(good_images) < 50:
        print(f"- 良品画像をあと{50-len(good_images)}枚以上追加してください")
    
    if len(defect_images) < 30:
        print(f"- 不良品画像をあと{30-len(defect_images)}枚以上追加してください")
    
    if len(unknown_images) > 0:
        print("- ファイル名に 'good' または 'defect' を含めてください")
        print("  例: IMG_7429.jpg -> good_001.jpg または defect_001.jpg")
    
    print("\n【不良部分のマーク方法】")
    print("1. 画像編集ソフトで不良箇所を赤い円で囲む")
    print("2. または、ファイル名で不良タイプを指定:")
    print("   - defect_black_spot_001.jpg (黒点)")
    print("   - defect_crack_001.jpg (ひび割れ)")
    print("   - defect_foreign_matter_001.jpg (異物)")
    
    return total >= 50  # 最低50枚あればOK

if __name__ == "__main__":
    # データ確認
    ready = check_dataset("cs AI学習データ")
    
    if ready:
        print("\n" + "=" * 60)
        print("データの準備が整っていまム！")
        print("学習を開始するには:")
        print("  python resin_washer_dl.py --train --data-dir \"cs AI学習データ\"")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("まず画像データを準備してください")
        print("=" * 60)
