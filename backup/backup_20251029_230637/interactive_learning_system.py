#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
インタラクティブ学習システム
リアルタイムフィードバック収集と継続学習
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time
import json
from pathlib import Path
import shutil
from datetime import datetime

class InteractiveLearningSystem:
    """インタラクティブ学習システム"""
    
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
        
        # 日本語クラス名
        self.japanese_names = {
            0: '良品',
            1: '欠け',
            2: '黒点',
            3: '傷'
        }
        
        # フィードバックデータの保存先
        self.feedback_dir = Path("feedback_data")
        self.feedback_dir.mkdir(exist_ok=True)
        
        # 学習データの保存先
        self.learning_data_dir = Path("learning_data")
        self.learning_data_dir.mkdir(exist_ok=True)
        
        # フィードバック統計
        self.feedback_stats = {
            'total_inspections': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'feedback_count': 0
        }
        
        # 判定閾値
        self.confidence_threshold = 0.3
        
    def load_model(self):
        """学習済みモデルを読み込み"""
        try:
            if not os.path.exists(self.model_path):
                print(f"エラー: モデルファイルが見つかりません: {self.model_path}")
                return False
            
            print(f"モデルを読み込み中: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            print("モデル読み込み完了")
            return True
            
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
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
    
    def save_feedback_data(self, image, predicted_class, correct_class, feedback_type):
        """フィードバックデータを保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # 画像を保存
        image_filename = f"{timestamp}_{feedback_type}.jpg"
        image_path = self.feedback_dir / image_filename
        cv2.imwrite(str(image_path), image)
        
        # メタデータを保存
        metadata = {
            'timestamp': timestamp,
            'predicted_class': int(predicted_class),
            'correct_class': int(correct_class),
            'feedback_type': feedback_type,
            'image_path': str(image_path)
        }
        
        metadata_filename = f"{timestamp}_metadata.json"
        metadata_path = self.feedback_dir / metadata_filename
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"フィードバックデータを保存: {image_filename}")
        
        # 統計更新
        self.feedback_stats['feedback_count'] += 1
        if predicted_class == correct_class:
            self.feedback_stats['correct_predictions'] += 1
        else:
            self.feedback_stats['incorrect_predictions'] += 1
    
    def draw_interactive_result(self, image, predicted_class, confidence):
        """インタラクティブな結果表示"""
        result_image = image.copy()
        h, w = result_image.shape[:2]
        
        # 判定結果
        class_name = self.japanese_names.get(predicted_class, "不明")
        is_good = predicted_class == 0
        
        # 背景色
        bg_color = (0, 255, 0) if is_good else (0, 0, 255)
        text_color = (255, 255, 255)
        
        # 判定結果の背景（中央上部に配置）
        cv2.rectangle(result_image, (w//2-200, 20), (w//2+200, 150), bg_color, -1)
        
        # 判定テキスト（大きく表示）
        if is_good:
            result_text = "OK"
            reason_text = "Good Product"
        else:
            result_text = "NG"
            reason_text = f"Defect: {class_name}"
        
        # OK/NG判定（大きく表示）
        cv2.putText(result_image, result_text, (w//2-50, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, text_color, 4, cv2.LINE_AA)
        
        # 理由表示
        cv2.putText(result_image, reason_text, (w//2-120, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
        
        # 信頼度
        confidence_text = f"Confidence: {confidence:.2f}"
        cv2.putText(result_image, confidence_text, (w//2-80, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
        
        # フィードバックボタンの表示
        button_y = h - 200
        button_height = 40
        button_width = 120
        
        # 正解ボタン
        cv2.rectangle(result_image, (50, button_y), (50 + button_width, button_y + button_height), (0, 255, 0), -1)
        cv2.putText(result_image, "CORRECT (C)", (55, button_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 間違いボタン
        cv2.rectangle(result_image, (200, button_y), (200 + button_width, button_y + button_height), (0, 0, 255), -1)
        cv2.putText(result_image, "WRONG (W)", (205, button_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 正解選択ボタン
        button_y2 = button_y + 60
        buttons = [
            ("良品 (1)", 50, button_y2, (0, 255, 0)),
            ("欠け (2)", 200, button_y2, (0, 0, 255)),
            ("黒点 (3)", 350, button_y2, (0, 0, 255)),
            ("傷 (4)", 500, button_y2, (0, 0, 255))
        ]
        
        for text, x, y, color in buttons:
            cv2.rectangle(result_image, (x, y), (x + button_width, y + button_height), color, -1)
            cv2.putText(result_image, text, (x + 5, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 操作説明（複数行に分割）
        instructions = [
            "C: Correct | W: Wrong",
            "1-4: True Class | ESC: Exit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(result_image, instruction, 
                       (50, h - 40 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return result_image
    
    def run_interactive_inspection(self, camera_id=0):
        """インタラクティブ検査システム"""
        print("=" * 60)
        print("インタラクティブ学習システム")
        print("=" * 60)
        print("操作方法:")
        print("  SPACE: 検査実行")
        print("  C: 正解")
        print("  W: 間違い")
        print("  1-4: 正解クラス選択")
        print("  ESC: 終了")
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
        
        print("カメラ初期化完了")
        print("インタラクティブ検査を開始します...")
        
        # 検査結果の統計
        total_inspections = 0
        good_count = 0
        defect_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("エラー: フレームを読み込めませんでした")
                    break
                
                # フレームをリサイズ
                display_frame = cv2.resize(frame, (800, 600))
                
                # 統計情報表示
                stats_text = f"Inspections: {total_inspections} | Good: {good_count} | Defect: {defect_count}"
                cv2.putText(display_frame, stats_text, (10, display_frame.shape[0]-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # フィードバック統計表示
                feedback_text = f"Feedback: {self.feedback_stats['feedback_count']} | Correct: {self.feedback_stats['correct_predictions']}"
                cv2.putText(display_frame, feedback_text, (10, display_frame.shape[0]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # 操作説明
                cv2.putText(display_frame, "SPACE: Inspect | C: Correct | W: Wrong | 1-4: True Class | ESC: Exit", 
                           (10, display_frame.shape[0]-60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                # 画面表示
                cv2.imshow("Interactive Learning System", display_frame)
                
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
                        result_frame = self.draw_interactive_result(display_frame, predicted_class, confidence)
                        cv2.imshow("Inspection Result", result_frame)
                        
                        # 統計更新
                        total_inspections += 1
                        if predicted_class == 0:  # 良品
                            good_count += 1
                        else:  # 不良品
                            defect_count += 1
                        
                        # フィードバック待機
                        while True:
                            feedback_key = cv2.waitKey(0) & 0xFF
                            
                            if feedback_key == ord('c'):  # 正解
                                print("正解として記録")
                                self.save_feedback_data(frame, predicted_class, predicted_class, "correct")
                                break
                            elif feedback_key == ord('w'):  # 間違い
                                print("間違いとして記録 - 正解クラスを選択してください")
                                # 正解クラス選択待機
                                while True:
                                    class_key = cv2.waitKey(0) & 0xFF
                                    if class_key == ord('1'):  # 良品
                                        correct_class = 0
                                        break
                                    elif class_key == ord('2'):  # 欠け
                                        correct_class = 1
                                        break
                                    elif class_key == ord('3'):  # 黒点
                                        correct_class = 2
                                        break
                                    elif class_key == ord('4'):  # 傷
                                        correct_class = 3
                                        break
                                    elif class_key == 27:  # ESC
                                        break
                                
                                if class_key != 27:
                                    print(f"正解クラス: {self.japanese_names[correct_class]}")
                                    self.save_feedback_data(frame, predicted_class, correct_class, "incorrect")
                                break
                            elif feedback_key == 27:  # ESC
                                break
                        
                        cv2.destroyWindow("Inspection Result")
                    else:
                        print("エラー: 判定に失敗しました")
        
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
            print(f"フィードバック数: {self.feedback_stats['feedback_count']}")
            print(f"正解数: {self.feedback_stats['correct_predictions']}")
            print(f"間違い数: {self.feedback_stats['incorrect_predictions']}")
            if total_inspections > 0:
                good_rate = (good_count / total_inspections) * 100
                print(f"良品率: {good_rate:.1f}%")
            print("=" * 60)

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="インタラクティブ学習システム")
    parser.add_argument("--model", type=str, default="resin_washer_model/resin_washer_model.h5",
                       help="学習済みモデルのパス")
    parser.add_argument("--camera", type=int, default=0, help="カメラID")
    
    args = parser.parse_args()
    
    # システム初期化
    system = InteractiveLearningSystem(args.model)
    
    # モデル読み込み
    if not system.load_model():
        print("モデルの読み込みに失敗しました")
        return
    
    # インタラクティブ検査
    system.run_interactive_inspection(args.camera)

if __name__ == "__main__":
    main()

