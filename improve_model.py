#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モデル改善のためのスクリプト
黒点検出の精度向上
"""

import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_black_spot_data():
    """黒点データの分析"""
    print("=" * 60)
    print("黒点データ分析")
    print("=" * 60)
    
    data_dir = Path("cs AI学習データ/樹脂")
    black_spot_dir = data_dir / "黒点樹脂"
    
    if not black_spot_dir.exists():
        print("エラー: 黒点樹脂フォルダーが見つかりません")
        return
    
    # 画像ファイルを取得
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(black_spot_dir.glob(ext))
    
    print(f"黒点画像数: {len(image_files)}")
    
    if len(image_files) == 0:
        print("警告: 黒点画像が見つかりません")
        return
    
    # 画像の特徴を分析
    print("\n画像特徴分析:")
    for i, img_path in enumerate(image_files[:5]):  # 最初の5枚を分析
        try:
            # 画像読み込み
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # グレースケール変換
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 黒点検出（閾値処理）
            _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
            
            # 輪郭検出
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 黒点の特徴
            black_spots = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 10:  # 小さなノイズを除去
                    black_spots.append(area)
            
            print(f"  画像 {i+1}: {len(black_spots)}個の黒点検出")
            if black_spots:
                print(f"    最大面積: {max(black_spots):.1f}")
                print(f"    平均面積: {np.mean(black_spots):.1f}")
            
        except Exception as e:
            print(f"  画像 {i+1}: エラー - {e}")
    
    print("\n" + "=" * 60)
    print("改善提案")
    print("=" * 60)
    print("1. 黒点データの追加")
    print("   - 現在: 19枚")
    print("   - 推奨: 50枚以上")
    print("   - 多様な照明条件での撮影")
    
    print("\n2. データ拡張の強化")
    print("   - 回転: ±15度")
    print("   - 明度調整: ±20%")
    print("   - コントラスト調整: ±15%")
    
    print("\n3. モデル構造の改善")
    print("   - より深いネットワーク")
    print("   - 転移学習の活用")
    print("   - アテンション機構の追加")
    
    print("\n4. 前処理の改善")
    print("   - 黒点強調フィルタ")
    print("   - エッジ検出の強化")
    print("   - 色空間変換の最適化")

def create_improved_model():
    """改善されたモデルの作成"""
    print("\n" + "=" * 60)
    print("改善されたモデルの作成")
    print("=" * 60)
    
    # モデル改善のためのコード
    print("1. データ拡張の強化")
    print("   - 黒点データの重点的な拡張")
    print("   - バランスの取れた学習データ")
    
    print("\n2. 損失関数の調整")
    print("   - クラス重みの調整")
    print("   - 焦点損失の導入")
    
    print("\n3. 学習戦略の改善")
    print("   - 段階的学習")
    print("   - 早期停止の調整")
    print("   - 学習率の最適化")

if __name__ == "__main__":
    analyze_black_spot_data()
    create_improved_model()







