#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Six-Class Inspection System
6クラス不良品検出システム（歪みテ凹み対応）
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import json
import time

# Set UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

class SixClassInspectionSystem:
    def __init__(self, model_path="six_class_enhanced_ensemble_model_1.h5"):
        self.model_path = model_path
        self.class_names = ['good', 'black_spot', 'chipping', 'scratch', 'distortion', 'dent']
        self.class_descriptions = {
            'good': '良品',
            'black_spot': '黒点',
            'chipping': '欠け',
            'scratch': '傷',
            'distortion': '歪み',
            'dent': '凹み'
        }
        self.model = None
        self.load_model()
    
    def load_model(self):
        """6クラス用モデルを読み込み"""
        try:
            if os.path.exists(self.model_path):
                print(f"モデルを読み込み中: {self.model_path}")
                self.model = tf.keras.models.load_model(self.model_path)
                print("モデル読み込み完了")
            else:
                print(f"警告: モデルファイルが見つかりません: {self.model_path}")
                print("利用可能なモデルファイルを検索中...")
                self.find_available_models()
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            self.find_available_models()
    
    def find_available_models(self):
        """利用可能なモデルファイルを検索"""
        model_files = [
            "six_class_enhanced_ensemble_model_1.h5",
            "six_class_enhanced_ensemble_model_2.h5", 
            "six_class_enhanced_ensemble_model_3.h5",
            "simple_enhanced_ensemble_model_1.h5",
            "ultra_high_performance_ensemble_model_1.h5"
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                print(f"利用可能なモデル: {model_file}")
                try:
                    self.model = tf.keras.models.load_model(model_file)
                    self.model_path = model_file
                    print(f"モデル読み込み成功: {model_file}")
                    return
                except Exception as e:
                    print(f"モデル読み込み失敗: {model_file} - {e}")
        
        print("利用可能なモデルが見つかりません")
        print("学習を実行してください: python scripts/six_class_enhanced_trainer.py")
    
    def preprocess_image(self, image):
        """画像の前処理"""
        if len(image.shape) == 3:
            # BGRからRGBに変換
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # リサイズ
        image = cv2.resize(image, (224, 224))
        
        # 正規化
        image = image.astype(np.float32) / 255.0
        
        # バッチ次元を追加
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_single_image(self, image):
        """単一画像の予測"""
        if self.model is None:
            return None, None, None
        
        try:
            # 前処理
            processed_image = self.preprocess_image(image)
            
            # 予測実行
            predictions = self.model.predict(processed_image, verbose=0)
            
            # 結果の解析
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]
            
            # 全クラスの信頼度
            class_confidences = {}
            for i, class_name in enumerate(self.class_names):
                class_confidences[class_name] = float(predictions[0][i])
            
            return predicted_class, confidence, class_confidences
            
        except Exception as e:
            print(f"予測エラー: {e}")
            return None, None, None
    
    def inspect_image_file(self, image_path):
        """画像ファイルの検査"""
        print(f"\n画像を検査中: {image_path}")
        
        # 画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"エラー: 画像を読み込めませんでした: {image_path}")
            return None
        
        # 予測実行
        predicted_class, confidence, class_confidences = self.predict_single_image(image)
        
        if predicted_class is None:
            print("予測に失敗しました")
            return None
        
        # 結果表示
        print(f"予測結果: {self.class_descriptions[predicted_class]} ({predicted_class})")
        print(f"信頼度: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # 判定
        if predicted_class == 'good':
            print("判定: OK (良品)")
        else:
            print(f"判定: NG ({self.class_descriptions[predicted_class]})")
        
        # 全クラスの信頼度
        print("\n全クラスの信頼度:")
        for class_name, conf in sorted(class_confidences.items(), key=lambda x: x[1], reverse=True):
            print(f"  {self.class_descriptions[class_name]}: {conf:.4f} ({conf*100:.2f}%)")
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_confidences': class_confidences,
            'is_ok': predicted_class == 'good'
        }
    
    def inspect_camera(self, camera_index=0):
        """カメラからのリアルタイム検査"""
        print(f"\nカメラ検査を開始します (カメラ {camera_index})")
        print("終了するには 'q' キーを押してください")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"エラー: カメラ {camera_index} を開けませんでした")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("カメラからの画像取得に失敗しました")
                    break
                
                # 予測実行
                predicted_class, confidence, class_confidences = self.predict_single_image(frame)
                
                if predicted_class is not None:
                    # 結果を画像に描画
                    result_text = f"{self.class_descriptions[predicted_class]}: {confidence*100:.1f}%"
                    color = (0, 255, 0) if predicted_class == 'good' else (0, 0, 255)
                    
                    cv2.putText(frame, result_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # 判定結果
                    status_text = "OK" if predicted_class == 'good' else "NG"
                    cv2.putText(frame, status_text, (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # 画像表示
                cv2.imshow('6-Class Inspection System', frame)
                
                # キー入力チェック
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # スクリーンショット保存
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    filename = f"inspection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"スクリーンショット保存: {filename}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def batch_inspect_directory(self, directory_path):
        """ディレクトリ内の全画像を一括検査"""
        print(f"\nディレクトリを一括検査中: {directory_path}")
        
        directory = Path(directory_path)
        if not directory.exists():
            print(f"エラー: ディレクトリが存在しません: {directory_path}")
            return
        
        # 画像ファイルを検索
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(directory.rglob(ext))
        
        if not image_files:
            print("画像ファイルが見つかりませんでした")
            return
        
        print(f"見つかった画像: {len(image_files)}枚")
        
        # 結果を保存
        results = []
        ok_count = 0
        ng_count = 0
        
        for i, image_file in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] 検査中...")
            result = self.inspect_image_file(image_file)
            
            if result:
                results.append({
                    'file_path': str(image_file),
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence'],
                    'is_ok': result['is_ok']
                })
                
                if result['is_ok']:
                    ok_count += 1
                else:
                    ng_count += 1
        
        # 結果サマリー
        print(f"\n" + "="*60)
        print("一括検査結果サマリー")
        print(f"="*60)
        print(f"総画像数: {len(image_files)}")
        print(f"OK (良品): {ok_count}枚")
        print(f"NG (不良品): {ng_count}枚")
        print(f"良品率: {ok_count/len(image_files)*100:.1f}%")
        
        # クラス別集計
        class_counts = {}
        for result in results:
            class_name = result['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\nクラス別集計:")
        for class_name, count in class_counts.items():
            print(f"  {self.class_descriptions[class_name]}: {count}枚")
        
        # 結果をJSONで保存
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_file = f"batch_inspection_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'total_images': len(image_files),
                'ok_count': ok_count,
                'ng_count': ng_count,
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n詳細結果を保存しました: {results_file}")

def main():
    """メイン実行関数"""
    print("=" * 80)
    print("Six-Class Inspection System")
    print("6クラス不良品検出システム（歪みテ凹み対応）")
    print("=" * 80)
    
    # 検査システム初期化
    inspector = SixClassInspectionSystem()
    
    if inspector.model is None:
        print("モデルが読み込めませんでした。学習を実行してください。")
        return
    
    print("\n利用可能なコマンド:")
    print("1. 単一画像検査: python scripts/six_class_inspection_system.py <画像ファイル>")
    print("2. カメラ検査: python scripts/six_class_inspection_system.py --camera")
    print("3. 一括検査: python scripts/six_class_inspection_system.py --batch <ディレクトリ>")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--camera':
            inspector.inspect_camera()
        elif sys.argv[1] == '--batch' and len(sys.argv) > 2:
            inspector.batch_inspect_directory(sys.argv[2])
        else:
            # 単一画像検査
            inspector.inspect_image_file(sys.argv[1])
    else:
        print("\n引数を指定してください。")

if __name__ == "__main__":
    main()
