#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
デバッグ用リアルタイム検査システム
文字化けを防ぐため、すべて英語で表示
"""

import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import argparse
from datetime import datetime

class RealtimeWasherInspector:
    def __init__(self, model_path="resin_washer_model/resin_washer_model.h5"):
        """リアルタイム検査システムの初期化"""
        print("=" * 60)
        print("Realtime Washer Inspection System - Debug Version")
        print("=" * 60)
        
        self.model_path = model_path
        self.model = None
        self.class_names = ['Good', 'Defect_Chip', 'Defect_Scratch', 'Defect_BlackSpot']
        
        # モデル読み込み
        self.load_model()
        
    def load_model(self):
        """モデルを読み込む"""
        print(f"Model path: {self.model_path}")
        print(f"File exists: {os.path.exists(self.model_path)}")
        
        if not os.path.exists(self.model_path):
            print(f"ERROR: Model file not found: {self.model_path}")
            return False
            
        try:
            print("Loading model...")
            self.model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            return False
    
    def preprocess_image(self, image):
        """画像の前処理"""
        if image is None:
            return None
            
        # リサイズ
        resized = cv2.resize(image, (224, 224))
        
        # 正規化
        normalized = resized.astype(np.float32) / 255.0
        
        # バッチ次元を追加
        batch = np.expand_dims(normalized, axis=0)
        
        return batch
    
    def predict_washer(self, image):
        """ワッシャーの品質を予測"""
        if self.model is None:
            return "Model not loaded", 0.0, "Unknown"
            
        # 前処理
        processed = self.preprocess_image(image)
        if processed is None:
            return "Preprocessing failed", 0.0, "Unknown"
        
        try:
            # 予測実行
            predictions = self.model.predict(processed, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            class_name = self.class_names[predicted_class]
            
            return class_name, confidence, class_name
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Prediction failed", 0.0, "Unknown"
    
    def draw_result(self, frame, result, confidence, class_name, washer_count):
        """結果を描画"""
        h, w = frame.shape[:2]
        result_image = frame.copy()
        
        # 結果表示
        if result == "Good":
            color = (0, 255, 0)  # 緑
            status = "OK"
        else:
            color = (0, 0, 255)  # 赤
            status = "NG"
        
        # メイン結果表示
        cv2.putText(result_image, f"Status: {status}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
        
        # 理由表示
        cv2.putText(result_image, f"Reason: {class_name}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 信頼度表示
        cv2.putText(result_image, f"Confidence: {confidence:.2f}", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # ワッシャー数表示
        cv2.putText(result_image, f"Washers: {washer_count}", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        return result_image

def run_camera_inspection(camera_id=0):
    """カメラ検査の実行"""
    print("=" * 60)
    print("Camera Inspection System")
    print("=" * 60)
    
    # 検査システム初期化
    inspector = RealtimeWasherInspector()
    
    if inspector.model is None:
        print("ERROR: Model could not be loaded")
        return
    
    # カメラ初期化
    print(f"Initializing camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_id}")
        return
    
    # カメラ設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 自動ピント調整（C920n Pro HD対応）
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 自動ピントON
        cap.set(cv2.CAP_PROP_FOCUS, 0)     # フォーカスをリセット
        cap.set(cv2.CAP_PROP_FOCUS, 50)    # フォーカス値を設定
        print("Auto focus enabled")
    except:
        print("Auto focus not available")
    
    # 手動ピント調整の試行
    try:
        # フォーカス範囲を設定
        cap.set(cv2.CAP_PROP_FOCUS, 30)     # 近距離フォーカス
        print("Manual focus set to 30")
    except:
        print("Manual focus not available")
    
    # その他のカメラ設定
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 自動露出
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)     # 明度
        cap.set(cv2.CAP_PROP_CONTRAST, 0.5)        # コントラスト
        print("Camera settings optimized")
    except:
        print("Camera optimization not available")
    
    print("Camera initialization complete")
    print("Starting inspection...")
    
    # 検査結果の統計
    total_inspections = 0
    good_count = 0
    defect_count = 0
    
    print("=" * 60)
    print("Controls:")
    print("  ESC: Exit")
    print("  SPACE: Inspect")
    print("  S: Save Image")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read from camera")
            break
        
        # 表示用フレーム
        display_frame = frame.copy()
        
        # 統計情報表示
        stats_text = f"Inspections: {total_inspections} | Good: {good_count} | Defect: {defect_count}"
        cv2.putText(display_frame, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 操作説明（詳細版）
        instructions = [
            "SPACE: Inspect (Check washer quality)",
            "ESC: Exit (Close application)", 
            "S: Save Image (Save current frame)",
            "F: Focus Test (Test focus quality)",
            "R: Reset Focus (Reset camera focus)"
        ]
        
        y_start = display_frame.shape[0] - 100
        for i, instruction in enumerate(instructions):
            cv2.putText(display_frame, instruction, 
                       (10, y_start + i * 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 画面表示
        cv2.imshow("Washer Inspection System", display_frame)
        
        # キー入力処理
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            # 検査実行
            print("Inspecting...")
            result, confidence, class_name = inspector.predict_washer(frame)
            
            # 結果表示
            result_frame = inspector.draw_result(frame, result, confidence, class_name, 1)
            cv2.imshow("Inspection Result", result_frame)
            
            # 統計更新
            total_inspections += 1
            if result == "Good":
                good_count += 1
            else:
                defect_count += 1
            
            print(f"Result: {result} (Confidence: {confidence:.2f})")
            
            # 結果を3秒間表示
            cv2.waitKey(3000)
            cv2.destroyWindow("Inspection Result")
            
        elif key == ord('s'):  # S
            # 画像保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inspection_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image saved: {filename}")
            
        elif key == ord('f'):  # F - フォーカステスト
            print("Testing focus quality...")
            # フォーカス品質をテスト
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            print(f"Focus quality (Laplacian variance): {laplacian_var:.2f}")
            
            if laplacian_var > 100:
                focus_status = "GOOD FOCUS"
                color = (0, 255, 0)
            elif laplacian_var > 50:
                focus_status = "MEDIUM FOCUS"
                color = (0, 255, 255)
            else:
                focus_status = "POOR FOCUS"
                color = (0, 0, 255)
            
            # フォーカス状態を表示
            cv2.putText(display_frame, f"Focus: {focus_status}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Quality: {laplacian_var:.2f}", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            
        elif key == ord('r'):  # R - フォーカスリセット
            print("Resetting camera focus...")
            try:
                cap.set(cv2.CAP_PROP_FOCUS, 0)     # フォーカスリセット
                cap.set(cv2.CAP_PROP_FOCUS, 50)    # フォーカス再設定
                print("Focus reset completed")
            except:
                print("Focus reset failed")
    
    # リソース解放
    cap.release()
    cv2.destroyAllWindows()
    
    print("=" * 60)
    print("Final Statistics")
    print("=" * 60)
    print(f"Total inspections: {total_inspections}")
    print(f"Good products: {good_count}")
    print(f"Defective products: {defect_count}")
    print("=" * 60)

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Real-time Washer Inspection System')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    
    args = parser.parse_args()
    
    print(f"Starting inspection with camera {args.camera}")
    run_camera_inspection(args.camera)

if __name__ == "__main__":
    main()