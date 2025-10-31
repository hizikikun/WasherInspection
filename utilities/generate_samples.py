#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Distortion and Dent Samples
歪みと凹みのサンプルデータを生成するスクリプト
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import json
import time

# Set UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

class DistortionDentSampleGenerator:
    def __init__(self, data_path="cs_AItraining_data"):
        self.data_path = Path(data_path)
        self.class_names = ['good', 'black_spot', 'chipping', 'scratch', 'distortion', 'dent']
        
    def create_directories(self):
        """6クラス用のディレクトリを作成"""
        print("=" * 80)
        print("6クラス用ディレクトリ構造を作成")
        print("=" * 80)
        
        resin_path = self.data_path / 'resin'
        resin_path.mkdir(parents=True, exist_ok=True)
        
        for class_name in self.class_names:
            class_path = resin_path / class_name
            class_path.mkdir(parents=True, exist_ok=True)
            print(f"作成: {class_path}")
    
    def generate_distortion_samples(self, num_samples=50):
        """歪みサンプルを生成"""
        print(f"\n歪みサンプルを生成中... ({num_samples}枚)")
        
        distortion_path = self.data_path / 'resin' / 'distortion'
        distortion_path.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_samples):
            # 基本のワッシャー形状を生成
            img = np.ones((224, 224, 3), dtype=np.uint8) * 128  # グレー背景
            
            # ワッシャーの外側円
            cv2.circle(img, (112, 112), 100, (200, 200, 200), -1)
            
            # ワッシャーの内側円（穴）
            cv2.circle(img, (112, 112), 30, (128, 128, 128), -1)
            
            # 歪み効果を追加
            # 楕円形に歪ませる
            if i % 3 == 0:
                # 水平方向の歪み
                M = cv2.getRotationMatrix2D((112, 112), 0, 1.0)
                M[0, 1] = 0.2  # せん断変換
                img = cv2.warpAffine(img, M, (224, 224))
            elif i % 3 == 1:
                # 垂直方向の歪み
                M = cv2.getRotationMatrix2D((112, 112), 0, 1.0)
                M[1, 0] = 0.2  # せん断変換
                img = cv2.warpAffine(img, M, (224, 224))
            else:
                # 複合歪み
                M = cv2.getRotationMatrix2D((112, 112), 0, 1.0)
                M[0, 1] = 0.1
                M[1, 0] = 0.1
                img = cv2.warpAffine(img, M, (224, 224))
            
            # のイズを追加
            noise = np.random.normal(0, 10, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # 明度を調整
            brightness = np.random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)
            
            # 保存
            filename = f"distortion_sample_{i+1:03d}.jpg"
            filepath = distortion_path / filename
            cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            if (i + 1) % 10 == 0:
                print(f"  歪みサンプル {i+1}/{num_samples} 生成完了")
        
        print(f"歪みサンプル生成完了: {num_samples}枚")
    
    def generate_dent_samples(self, num_samples=50):
        """凹みサンプルを生成"""
        print(f"\n凹みサンプルを生成中... ({num_samples}枚)")
        
        dent_path = self.data_path / 'resin' / 'dent'
        dent_path.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_samples):
            # 基本のワッシャー形状を生成
            img = np.ones((224, 224, 3), dtype=np.uint8) * 128  # グレー背景
            
            # ワッシャーの外側円
            cv2.circle(img, (112, 112), 100, (200, 200, 200), -1)
            
            # ワッシャーの内側円（穴）
            cv2.circle(img, (112, 112), 30, (128, 128, 128), -1)
            
            # 凹み効果を追加
            # ランダムな位置に凹みを配置
            center_x = np.random.randint(50, 174)
            center_y = np.random.randint(50, 174)
            radius = np.random.randint(10, 25)
            
            # 凹みの影効果
            cv2.circle(img, (center_x, center_y), radius, (100, 100, 100), -1)
            cv2.circle(img, (center_x, center_y), radius-2, (80, 80, 80), -1)
            
            # 凹みのハイライト効果
            highlight_x = center_x - radius // 3
            highlight_y = center_y - radius // 3
            cv2.circle(img, (highlight_x, highlight_y), radius // 3, (220, 220, 220), -1)
            
            # 複数の小さな凹みを追加（確率的に）
            if np.random.random() > 0.5:
                for _ in range(np.random.randint(1, 4)):
                    small_x = np.random.randint(50, 174)
                    small_y = np.random.randint(50, 174)
                    small_radius = np.random.randint(3, 8)
                    cv2.circle(img, (small_x, small_y), small_radius, (120, 120, 120), -1)
            
            # のイズを追加
            noise = np.random.normal(0, 8, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # 明度を調整
            brightness = np.random.uniform(0.7, 1.3)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)
            
            # 保存
            filename = f"dent_sample_{i+1:03d}.jpg"
            filepath = dent_path / filename
            cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            if (i + 1) % 10 == 0:
                print(f"  凹みサンプル {i+1}/{num_samples} 生成完了")
        
        print(f"凹みサンプル生成完了: {num_samples}枚")
    
    def generate_additional_good_samples(self, num_samples=100):
        """追加の良品サンプルを生成"""
        print(f"\n追加の良品サンプルを生成中... ({num_samples}枚)")
        
        good_path = self.data_path / 'resin' / 'good'
        good_path.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_samples):
            # 基本のワッシャー形状を生成
            img = np.ones((224, 224, 3), dtype=np.uint8) * 128  # グレー背景
            
            # ワッシャーの外側円
            cv2.circle(img, (112, 112), 100, (200, 200, 200), -1)
            
            # ワッシャーの内側円（穴）
            cv2.circle(img, (112, 112), 30, (128, 128, 128), -1)
            
            # 微細な表面テクスチャを追加
            for _ in range(np.random.randint(5, 15)):
                x = np.random.randint(30, 194)
                y = np.random.randint(30, 194)
                radius = np.random.randint(1, 3)
                brightness = np.random.randint(180, 220)
                cv2.circle(img, (x, y), radius, (brightness, brightness, brightness), -1)
            
            # 軽微なのイズを追加
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # 明度を調整
            brightness = np.random.uniform(0.9, 1.1)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)
            
            # 保存
            filename = f"good_additional_{i+1:03d}.jpg"
            filepath = good_path / filename
            cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            if (i + 1) % 20 == 0:
                print(f"  良品サンプル {i+1}/{num_samples} 生成完了")
        
        print(f"追加の良品サンプル生成完了: {num_samples}枚")
    
    def create_sample_info(self):
        """サンプル情報ファイルを作成"""
        print("\nサンプル情報ファイルを作成中...")
        
        sample_info = {
            'generated_samples': {
                'distortion': 50,
                'dent': 50,
                'good_additional': 100
            },
            'total_classes': 6,
            'class_names': self.class_names,
            'class_descriptions': {
                'good': '良品',
                'black_spot': '黒点',
                'chipping': '欠け',
                'scratch': '傷',
                'distortion': '歪み',
                'dent': '凹み'
            },
            'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'Generated samples for 6-class defect detection system'
        }
        
        info_path = self.data_path / 'sample_generation_info.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(sample_info, f, indent=2, ensure_ascii=False)
        
        print(f"サンプル情報を保存しました: {info_path}")

def main():
    """メイン実行関数"""
    print("=" * 80)
    print("Distortion and Dent Sample Generator")
    print("歪みと凹みのサンプルデータを生成します")
    print("=" * 80)
    
    # ジェネレーター初期化
    generator = DistortionDentSampleGenerator()
    
    try:
        # ディレクトリ構造作成
        print("\n[ステップ1] 6クラス用ディレクトリ構造作成")
        generator.create_directories()
        
        # 歪みサンプル生成
        print("\n[ステップ2] 歪みサンプル生成")
        generator.generate_distortion_samples(50)
        
        # 凹みサンプル生成
        print("\n[ステップ3] 凹みサンプル生成")
        generator.generate_dent_samples(50)
        
        # 追加の良品サンプル生成
        print("\n[ステップ4] 追加の良品サンプル生成")
        generator.generate_additional_good_samples(100)
        
        # サンプル情報作成
        print("\n[ステップ5] サンプル情報作成")
        generator.create_sample_info()
        
        # 結果表示
        print("\n" + "=" * 80)
        print("サンプル生成完了!")
        print("=" * 80)
        print("生成されたサンプル:")
        print("  - 歪み (distortion): 50枚")
        print("  - 凹み (dent): 50枚")
        print("  - 良品 (good): 100枚 (追加)")
        print("\n次のステップ:")
        print("  python scripts/six_class_enhanced_trainer.py")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
