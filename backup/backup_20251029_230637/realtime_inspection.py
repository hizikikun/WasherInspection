#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
樹脂製ワッシャー リアルタイム外観検査システム
ディープラーニングモデルを使用した良品判定
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time
import argparse
from pathlib import Path

class RealtimeWasherInspector:
    """リアルタイムワッシャー検査システム"""
    
    def __init__(self, model_path="resin_washer_model/resin_washer_model.h5"):
        self.model_path = model_path
        self.model = None
        self.image_size = (224, 224)
        
        # クラス名の定義
        self.class_names = {
            0: 'good',
            1: 'chipped', 
            2: 'black_spot',
            3: 'scratched'
        }
        
        # 英語クラス名（文字化け回避）
        self.english_names = {
            0: 'Good',
            1: 'Chipped',
            2: 'Black Spot',
            3: 'Scratched'
        }
        
        # 判定閾値（より厳しく設定）
        self.confidence_threshold = 0.3
        
    def load_model(self):
        """学習済みモデルを読み込み"""
        try:
            print(f"現在のディレクトリ: {os.getcwd()}")
            print(f"モデルパス: {self.model_path}")
            print(f"ファイル存在: {os.path.exists(self.model_path)}")
            
            if not os.path.exists(self.model_path):
                print(f"エラー: モデルファイルが見つかりません: {self.model_path}")
                return False
            
            print(f"モデルを読み込み中: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            print("モデル読み込み完了")
            return True
            
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess_image(self, image):
        """画像の前処理"""
        # リサイズ
        image = cv2.resize(image, self.image_size)
        
        # 正規化
        image = image.astype('float32') / 255.0
        
        # バッチ次元を追加
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_defect(self, image):
        """欠陥予測"""
        if self.model is None:
            return None, 0.0
        
        # 前処理
        processed_image = self.preprocess_image(image)
        
        # 予測
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # 信頼度が低い場合はNG判定（安全側に倒す）
        if confidence < self.confidence_threshold:
            # 最も可能性の高い不良品クラスを選択
            defect_classes = [1, 2, 3]  # 欠け、黒点、傷
            defect_confidences = [predictions[0][i] for i in defect_classes]
            max_defect_idx = np.argmax(defect_confidences)
            predicted_class = defect_classes[max_defect_idx]
            confidence = float(defect_confidences[max_defect_idx])
        
        return predicted_class, confidence
    
    def draw_result(self, image, predicted_class, confidence):
        """結果を画像に描画"""
        result_image = image.copy()
        h, w = result_image.shape[:2]
        
        # 判定結果
        class_name = self.english_names.get(predicted_class, "Unknown")
        is_good = predicted_class == 0
        
        # 背景色
        bg_color = (0, 255, 0) if is_good else (0, 0, 255)
        text_color = (255, 255, 255)
        
        # 判定結果の背景（中央上部に配置）
        cv2.rectangle(result_image, (w//2-150, 20), (w//2+150, 130), bg_color, -1)
        
        # 判定テキスト（大きく表示）
        if is_good:
            result_text = "OK"
            reason_text = "Good Product"
        else:
            result_text = "NG"
            reason_text = f"Defect: {class_name}"
        
        # OK/NG判定（大きく表示）
        cv2.putText(result_image, result_text, (w//2-50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, text_color, 4, cv2.LINE_AA)
        
        # 理由表示
        cv2.putText(result_image, reason_text, (w//2-100, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
        
        # 信頼度
        confidence_text = f"Confidence: {confidence:.2f}"
        cv2.putText(result_image, confidence_text, (w//2-80, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
        
        # 判定基準
        if confidence < 0.4:  # より厳しい基準
            warning_text = "Warning: Very Low Confidence - Manual Check Required"
            cv2.putText(result_image, warning_text, (w//2-150, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        elif confidence < 0.6:
            warning_text = "Warning: Low Confidence - Please Verify"
            cv2.putText(result_image, warning_text, (w//2-120, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        
        return result_image
    
    def run_camera_inspection(self, camera_id=0):
        """カメラを使ったリアルタイム検査"""
        print("=" * 60)
        print("樹脂製ワッシャー リアルタイム外観検査システム")
        print("=" * 60)
        print("操作方法:")
        print("  ESC: 終了")
        print("  SPACE: 判定実行")
        print("  S: 画像保存")
        print("=" * 60)
        
        # カメラ初期化
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"エラー: カメラ {camera_id} を開けませんでした")
            return
        
        # カメラ設定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 自動ピント調整（C920n Pro HD対応）
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 自動ピントON
            cap.set(cv2.CAP_PROP_FOCUS, 0)     # フォーカスをリセット
            print("自動ピント調整を有効にしました")
        except:
            print("自動ピント調整は利用できません")
        
        # その他のカメラ設定
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 自動露出
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)     # 明度
            cap.set(cv2.CAP_PROP_CONTRAST, 0.5)        # コントラスト
            print("カメラ設定を最適化しました")
        except:
            print("カメラ設定の最適化は利用できません")
        
        print("カメラ初期化完了")
        print("検査を開始します...")
        
        # 検査結果の統計
        total_inspections = 0
        good_count = 0
        defect_count = 0
        
        # フレームレート計算用
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("エラー: フレームを読み込めませんでした")
                    break
                
                # フレームをリサイズ
                display_frame = cv2.resize(frame, (800, 600))
                
                # フレームレート計算
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps_end_time = time.time()
                    fps = 30 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                else:
                    fps = 0
                
                # フレームレート表示（右上に移動）
                if fps > 0:
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1]-100, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                
                # 統計情報表示（右上に移動）
                stats_text = f"Inspections: {total_inspections} | Good: {good_count} | Defect: {defect_count}"
                cv2.putText(display_frame, stats_text, (display_frame.shape[1]-200, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # 操作説明（複数行に分割して見やすく）
                instructions = [
                    "SPACE: Inspect",
                    "ESC: Exit", 
                    "S: Save Image"
                ]
                
                y_start = 90
                for i, instruction in enumerate(instructions):
                    cv2.putText(display_frame, instruction, 
                               (display_frame.shape[1]-200, y_start + i * 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # 画面表示（英語タイトルで文字化けを回避）
                cv2.imshow("Resin Washer Inspection System", display_frame)
                
                # キー入力処理
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # SPACE
                    # 検査実行
                    print("検査実行中...")
                    predicted_class, confidence = self.predict_defect(frame)
                    
                    if predicted_class is not None:
                        # 結果描画
                        result_frame = self.draw_result(display_frame, predicted_class, confidence)
                        cv2.imshow("Inspection Result", result_frame)
                        
                        # 統計更新
                        total_inspections += 1
                        if predicted_class == 0:  # 良品
                            good_count += 1
                            print(f"判定: 良品 (信頼度: {confidence:.2f})")
                        else:  # 不良品
                            defect_count += 1
                            defect_name = self.english_names[predicted_class]
                            print(f"判定: 不良品 - {defect_name} (信頼度: {confidence:.2f})")
                        
                        # 結果を少し表示
                        cv2.waitKey(2000)
                        cv2.destroyWindow("Inspection Result")
                    else:
                        print("エラー: 判定に失敗しました")
                
                elif key == ord('s'):  # S
                    # 画像保存
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"inspection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"画像を保存しました: {filename}")
        
        except KeyboardInterrupt:
            print("\n検査を中断しました")
        
        finally:
            # リソース解放
            cap.release()
            cv2.destroyAllWindows()
            
            # 最終統計
            print("\n" + "=" * 60)
            print("検査統計")
            print("=" * 60)
            print(f"総検査数: {total_inspections}")
            print(f"良品: {good_count}")
            print(f"不良品: {defect_count}")
            if total_inspections > 0:
                good_rate = (good_count / total_inspections) * 100
                print(f"良品率: {good_rate:.1f}%")
            print("=" * 60)
    
    def test_single_image(self, image_path):
        """単一画像のテスト"""
        if not os.path.exists(image_path):
            print(f"エラー: 画像ファイルが見つかりません: {image_path}")
            return
        
        print(f"画像を読み込み中: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print("エラー: 画像を読み込めませんでした")
            return
        
        print("判定実行中...")
        predicted_class, confidence = self.predict_defect(image)
        
        if predicted_class is not None:
            class_name = self.english_names[predicted_class]
            is_good = predicted_class == 0
            
            print("\n" + "=" * 40)
            print("判定結果")
            print("=" * 40)
            print(f"判定: {'OK' if is_good else 'NG'}")
            print(f"クラス: {class_name}")
            print(f"信頼度: {confidence:.2f}")
            print("=" * 40)
            
            # 結果を画像に描画
            result_image = self.draw_result(image, predicted_class, confidence)
            
            # 結果表示
            cv2.imshow("Inspection Result", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("エラー: 判定に失敗しました")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="樹脂製ワッシャー リアルタイム外観検査システム")
    parser.add_argument("--model", type=str, default="resin_washer_model/resin_washer_model.h5",
                       help="学習済みモデルのパス")
    parser.add_argument("--camera", type=int, default=0, help="カメラID")
    parser.add_argument("--test-image", type=str, help="テスト用画像のパス")
    
    args = parser.parse_args()
    
    # 検査システム初期化
    inspector = RealtimeWasherInspector(args.model)
    
    # モデル読み込み
    if not inspector.load_model():
        print("モデルの読み込みに失敗しました")
        return
    
    # 単一画像テスト
    if args.test_image:
        inspector.test_single_image(args.test_image)
    else:
        # リアルタイム検査
        inspector.run_camera_inspection(args.camera)

if __name__ == "__main__":
    main()