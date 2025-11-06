#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
少数派クラスのデータ拡張でデータを増やす
写真を新たに撮る必要はありません！
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import shutil
import time
from multiprocessing import Pool, cpu_count
import concurrent.futures
import random
import math

# UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

class MinorityClassAugmenter:
    def __init__(self, data_path="cs_AItraining_data", target_count=300):
        """
        Args:
            data_path: データディレクトリ
            target_count: 各少数派クラスの目標枚数（デフォルト300枚）
        """
        self.data_path = Path(data_path)
        self.target_count = target_count
        self.minority_classes = ['black_spot', 'chipping', 'scratch']  # 少数派クラス
        
        # 拡張パラメータ（OpenCVで実装）
        self.rotation_range = 15  # 回転: ±15度
        self.width_shift_range = 0.1  # 水平方向シフト: ±10%
        self.height_shift_range = 0.1  # 垂直方向シフト: ±10%
        self.zoom_range = 0.1  # ズーム: 90%-110%
        self.brightness_range = [0.9, 1.1]  # 明るさ: 90%-110%
        self.horizontal_flip_prob = 0.5  # 水平反転の確率
        
        print("=" * 60)
        print("少数派クラスのデータ拡張")
        print("=" * 60)
        print(f"目標枚数: {self.target_count}枚/クラス")
        print(f"対象クラス: {', '.join(self.minority_classes)}")
        print("=" * 60)
    
    def count_images(self, class_path):
        """クラスディレクトリ内の画像数をカウント"""
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(class_path.rglob(ext))
        return len(image_files)
    
    def load_images(self, class_path):
        """画像を読み込む"""
        images = []
        image_files = []
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(class_path.rglob(ext))
        
        for img_file in image_files:
            try:
                img = cv2.imread(str(img_file))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    images.append(img)
            except Exception as e:
                print(f"警告: {img_file} の読み込み失敗: {e}")
        
        return np.array(images)
    
    def apply_augmentation(self, img):
        """OpenCVを使ったデータ拡張を適用"""
        aug_img = img.copy()
        h, w = aug_img.shape[:2]
        
        # 1. 回転
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # 2. シフト
        if self.width_shift_range > 0 or self.height_shift_range > 0:
            tx = random.uniform(-self.width_shift_range * w, self.width_shift_range * w)
            ty = random.uniform(-self.height_shift_range * h, self.height_shift_range * h)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # 3. ズーム
        if self.zoom_range > 0:
            zoom = random.uniform(1.0 - self.zoom_range, 1.0 + self.zoom_range)
            new_w, new_h = int(w * zoom), int(h * zoom)
            aug_img = cv2.resize(aug_img, (new_w, new_h))
            # 中央を切り出し
            start_x = max(0, (new_w - w) // 2)
            start_y = max(0, (new_h - h) // 2)
            aug_img = aug_img[start_y:start_y+h, start_x:start_x+w]
            # サイズが足りない場合はリサイズ
            if aug_img.shape[0] != h or aug_img.shape[1] != w:
                aug_img = cv2.resize(aug_img, (w, h))
        
        # 4. 水平反転
        if random.random() < self.horizontal_flip_prob:
            aug_img = cv2.flip(aug_img, 1)
        
        # 5. 明るさ調整
        if self.brightness_range[0] < 1.0 or self.brightness_range[1] > 1.0:
            brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
            aug_img = cv2.convertScaleAbs(aug_img, alpha=brightness, beta=0)
        
        return aug_img
    
    def augment_class(self, class_name):
        """指定クラスの画像を拡張して保存"""
        class_path = self.data_path / 'resin' / class_name
        if not class_path.exists():
            print(f"エラー: {class_path} が見つかりません")
            return False
        
        # 現在の画像数を確認
        current_count = self.count_images(class_path)
        print(f"\n[{class_name}]")
        print(f"  現在の枚数: {current_count}枚")
        
        if current_count >= self.target_count:
            print(f"  目標枚数({self.target_count}枚)に既に達しています。スキップします。")
            return True
        
        # 必要な拡張画像数
        needed_count = self.target_count - current_count
        print(f"  必要な追加枚数: {needed_count}枚")
        
        # 画像を読み込む
        print(f"  画像を読み込み中...")
        images = self.load_images(class_path)
        
        if len(images) == 0:
            print(f"  エラー: 画像が見つかりません")
            return False
        
        print(f"  {len(images)}枚の画像を読み込みました")
        
        # 拡張画像を生成（進捗表示付き、並列処理）
        print(f"  拡張画像を生成中...")
        augmented_images = []
        
        # 各元画像から複数の拡張画像を生成
        images_per_source = max(1, needed_count // len(images) + 1)
        
        start_time = time.time()
        total_to_generate = min(needed_count, len(images) * images_per_source)
        
        for i, img in enumerate(images):
            if len(augmented_images) >= needed_count:
                break
            
            # この画像から複数の拡張画像を生成
            for j in range(min(images_per_source, needed_count - len(augmented_images))):
                aug_img = self.apply_augmentation(img)
                augmented_images.append(aug_img)
                
                # 進捗表示（10枚ごと）
                if len(augmented_images) % 10 == 0 or len(augmented_images) >= needed_count:
                    elapsed = time.time() - start_time
                    progress = (len(augmented_images) / needed_count) * 100
                    if len(augmented_images) > 0:
                        avg_time = elapsed / len(augmented_images)
                        remaining = (needed_count - len(augmented_images)) * avg_time
                        print(f"\r  進捗: {len(augmented_images)}/{needed_count}枚 ({progress:.1f}%) "
                              f"経過: {elapsed:.1f}秒 残り: {remaining:.1f}秒", end='', flush=True)
        
        print()  # 改行
        
        # 保存
        print(f"  {len(augmented_images)}枚の拡張画像を保存中...")
        
        # 拡張画像保存用ディレクトリ（元のディレクトリに保存）
        save_dir = class_path
        
        # 既存のファイル名から最大番号を取得
        existing_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        max_num = 0
        for f in existing_files:
            try:
                # ファイル名から数字を抽出（例: image_001.jpg → 1）
                num_str = ''.join(filter(str.isdigit, f.stem))
                if num_str:
                    max_num = max(max_num, int(num_str))
            except:
                pass
        
        # 拡張画像を保存
        for i, aug_img in enumerate(augmented_images):
            # BGRに変換（OpenCVの保存形式）
            aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            
            # ファイル名を生成
            new_num = max_num + i + 1
            filename = save_dir / f"aug_{class_name}_{new_num:04d}.jpg"
            
            # 保存
            cv2.imwrite(str(filename), aug_img_bgr)
        
        print(f"  ✓ {len(augmented_images)}枚の拡張画像を保存しました")
        
        # 最終的な画像数を確認
        final_count = self.count_images(class_path)
        print(f"  最終枚数: {final_count}枚（目標: {self.target_count}枚）")
        
        return True
    
    def augment_all_minority_classes(self):
        """すべての少数派クラスを拡張"""
        print("\n少数派クラスのデータ拡張を開始します...")
        print(f"写真を新たに撮る必要はありません！\n")
        
        results = {}
        for class_name in self.minority_classes:
            success = self.augment_class(class_name)
            results[class_name] = success
        
        print("\n" + "=" * 60)
        print("データ拡張完了")
        print("=" * 60)
        
        for class_name, success in results.items():
            status = "✓ 成功" if success else "✗ 失敗"
            current_count = self.count_images(self.data_path / 'resin' / class_name)
            print(f"{class_name}: {status} ({current_count}枚)")
        
        return results

def main():
    print("=" * 80)
    print("少数派クラスのデータ拡張ツール")
    print("=" * 80)
    print("\nこのツールは既存の画像から新しい画像を自動生成します。")
    print("写真を新たに撮る必要はありません！\n")
    
    # デフォルト設定
    data_path = "cs_AItraining_data"
    target_count = 300  # 各少数派クラス300枚を目標
    
    # 自動実行モード
    print(f"データパス: {data_path}")
    print(f"目標枚数: {target_count}枚/クラス")
    print("\n自動実行モード: 3秒後に開始します...")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\nキャンセルされました")
        return
    
    augmenter = MinorityClassAugmenter(data_path, target_count)
    results = augmenter.augment_all_minority_classes()
    
    print("\n" + "=" * 80)
    print("完了！")
    print("=" * 80)
    print("\n次のステップ:")
    print("1. 生成された拡張画像を確認してください")
    print("2. 問題のある画像があれば削除してください")
    print("3. 再学習を実行してください:")
    print("   python scripts/train_4class_sparse_ensemble.py")
    print("=" * 80)

if __name__ == '__main__':
    main()

