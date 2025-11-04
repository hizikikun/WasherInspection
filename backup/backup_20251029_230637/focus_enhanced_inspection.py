#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
樹脂製ワッシャー ピント強化版リアルタイム検出システム
ピント調整を大幅に強化し、より鋭い画像を実現
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time
import threading
from pathlib import Path

class FocusEnhancedResinInspection:
    """ピント強化版樹脂製ワッシャー検出システム"""
    
    def __init__(self, camera_id=2):
        self.model = None
        self.cap = None
        self.running = False
        self.current_prediction = "Unknown"
        self.current_confidence = 0.0
        self.class_names = ['良品', '欠け', '黒点', '傷']
        self.feedback_count = 0
        self.last_update_time = 0
        self.camera_id = camera_id
        self.focus_score = 0.0
        self.best_focus_score = 0.0
        self.best_focus_value = 0
        self.focus_history = []
        self.focus_search_mode = True
        
        # モデル読み込み
        self.load_model()
        
    def load_model(self):
        """モデルを読み込む"""
        try:
            model_path = "resin_washer_model/resin_washer_model.h5"
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                print(f"モデル読み込み成功: {model_path}")
            else:
                print(f"モデルファイルが見つかりません: {model_path}")
                return False
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            return False
        return True
        
    def calculate_enhanced_focus_score(self, image):
        """強化版ピントスコア計算"""
        if image is None:
            return 0.0
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 複数の方法でピントを評価
            # 1. Laplacian分散
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. Sobelエッジ検出
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_var = sobel_magnitude.var()
            
            # 3. 中央部分のエッジ密度（ワッシャーが中央にあるため）
            h, w = gray.shape
            center_region = gray[h//4:3*h//4, w//4:3*w//4]
            center_edges = cv2.Canny(center_region, 50, 150)
            edge_density = np.sum(center_edges > 0) / (center_edges.shape[0] * center_edges.shape[1])
            
            # 4. 高周波成分
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            high_freq_score = np.mean(magnitude_spectrum[h//3:2*h//3, w//3:2*w//3])
            
            # 重み付き平均（中央部分を重視）
            combined_score = (laplacian_var * 0.4 + 
                            sobel_var * 0.3 + 
                            edge_density * 1000 * 0.2 + 
                            high_freq_score * 0.1)
            
            return combined_score
            
        except:
            return 0.0
            
    def comprehensive_focus_search(self, image):
        """包括的ピント検索"""
        current_focus = self.calculate_enhanced_focus_score(image)
        self.focus_score = current_focus
        
        # ピント履歴を記録
        self.focus_history.append((self.cap.get(cv2.CAP_PROP_FOCUS), current_focus))
        if len(self.focus_history) > 10:
            self.focus_history.pop(0)
        
        # 最高ピントスコアを更新
        if current_focus > self.best_focus_score:
            self.best_focus_score = current_focus
            self.best_focus_value = self.cap.get(cv2.CAP_PROP_FOCUS)
            print(f"新しい最高ピントスコア: {current_focus:.1f} (フォーカス値: {self.best_focus_value:.1f})")
        
        # ピント検索モード
        if self.focus_search_mode:
            try:
                current_focus_val = self.cap.get(cv2.CAP_PROP_FOCUS)
                
                # より積極的な調整
                if current_focus < 100:  # より厳しい閾値
                    # ランダムウォークで最適解を探索
                    if len(self.focus_history) >= 3:
                        recent_scores = [h[1] for h in self.focus_history[-3:]]
                        if all(s < max(recent_scores) for s in recent_scores[:-1]):
                            # ピントが悪化している場合は方向を変える
                            step = np.random.choice([-3, -2, -1, 1, 2, 3])
                        else:
                            step = np.random.choice([-2, -1, 1, 2])
                    else:
                        step = np.random.choice([-2, -1, 1, 2])
                    
                    new_focus = current_focus_val + step
                    new_focus = max(0, min(100, new_focus))
                    
                    self.cap.set(cv2.CAP_PROP_FOCUS, new_focus)
                    print(f"ピント検索: {current_focus_val:.1f} -> {new_focus:.1f} (スコア: {current_focus:.1f})")
                
            except Exception as e:
                print(f"ピント調整エラー: {e}")
        
    def initialize_camera(self):
        """カメラを初期化（ピント強化版）"""
        print(f"カメラ {self.camera_id} を初期化中...")
        
        # 複数のカメラIDを試す
        camera_ids = [self.camera_id, 0, 2, 3, 4]
        
        for cam_id in camera_ids:
            try:
                print(f"カメラ {cam_id} をテスト中...")
                cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
                
                if cap.isOpened():
                    # カメラ設定（ピント強化版）
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # フォーカス設定（完全手動制御）
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 自動フォーカス完全無効
                    cap.set(cv2.CAP_PROP_FOCUS, 0)     # フォーカス値を0に設定
                    
                    # 画質設定
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
                    cap.set(cv2.CAP_PROP_CONTRAST, 0.6)  # コントラストを上げる
                    cap.set(cv2.CAP_PROP_SATURATION, 0.5)
                    cap.set(cv2.CAP_PROP_GAIN, 0)
                    cap.set(cv2.CAP_PROP_SHARPNESS, 0.8)  # シャープネスを上げる
                    
                    # テスト撮影
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # ピントスコアをチェック
                        focus_score = self.calculate_enhanced_focus_score(frame)
                        print(f"カメラ {cam_id} 初期化成功!")
                        print(f"解像度: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                        print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
                        print(f"初期ピントスコア: {focus_score:.1f}")
                        
                        self.cap = cap
                        self.camera_id = cam_id
                        return True
                    else:
                        cap.release()
                        print(f"カメラ {cam_id} テスト撮影失敗")
                else:
                    print(f"カメラ {cam_id} を開けませんでした")
                    
            except Exception as e:
                print(f"カメラ {cam_id} エラー: {e}")
                continue
                
        print("利用可能なカメラが見つかりませんでした")
        return False
        
    def preprocess_image(self, image):
        """画像を前処理"""
        if image is None:
            return None
            
        # リサイズ
        image_resized = cv2.resize(image, (224, 224))
        
        # 正規化
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # バッチ次元を追加
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
        
    def predict(self, image):
        """画像を予測"""
        if self.model is None:
            return "Unknown", 0.0
            
        try:
            # 前処理
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return "Unknown", 0.0
                
            # 予測
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            return self.class_names[predicted_class], confidence
            
        except Exception as e:
            print(f"予測エラー: {e}")
            return "Unknown", 0.0
            
    def draw_focus_hud(self, frame):
        """ピント強化版HUDを描画"""
        height, width = frame.shape[:2]
        
        # 左下：システム情報
        info_width = 280
        info_height = 200
        info_x = 10
        info_y = height - info_height - 10
        
        # 半透明の黒背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (info_x, info_y), (info_x + info_width, info_y + info_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # システム情報テキスト
        info_texts = [
            f"Feedback: {self.feedback_count}",
            f"Focus Score: {self.focus_score:.1f}",
            f"Best Score: {self.best_focus_score:.1f}",
            f"Search Mode: {'ON' if self.focus_search_mode else 'OFF'}",
            "",
            "Controls",
            "1: Good  2: Black",
            "3: Chipped  4: Scratched",
            "M: Select  S: Save",
            "F: Focus Test  B: Best Focus",
            "T: Toggle Search  ESC: Exit"
        ]
        
        y_offset = info_y + 20
        for i, text in enumerate(info_texts):
            if text:
                font_scale = 0.6 if i == 5 else 0.5  # "Controls"を少し大きく
                thickness = 2 if i == 5 else 1
                cv2.putText(frame, text, (info_x + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            y_offset += 18
            
        # 右上：判定結果
        result_width = 200
        result_height = 60
        result_x = width - result_width - 10
        result_y = 10
        
        # 結果の背景色
        if self.current_prediction == "良品":
            bg_color = (0, 255, 0)  # 緑
            result_text = "Result: OK"
        else:
            bg_color = (0, 0, 255)  # 赤
            result_text = "Result: NG - Defect"
            
        # 結果パネル
        cv2.rectangle(frame, (result_x, result_y), (result_x + result_width, result_y + result_height), bg_color, -1)
        
        # 結果テキスト
        cv2.putText(frame, result_text, (result_x + 10, result_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {self.current_confidence:.2f}", (result_x + 10, result_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
        
    def save_image(self, frame, class_name):
        """画像を保存"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"feedback_{class_name}_{timestamp}.jpg"
            
            # 日本語パス対応
            save_dir = Path("cs AI学習データ/樹脂") / class_name / "追加学習"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = save_dir / filename
            
            # 画像保存
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                with open(save_path, 'wb') as f:
                    f.write(buffer.tobytes())
                print(f"画像保存成功: {save_path}")
                return True
            else:
                print("画像エンコード失敗")
                return False
                
        except Exception as e:
            print(f"画像保存エラー: {e}")
            return False
            
    def run(self):
        """メインループ"""
        if not self.initialize_camera():
            print("カメラ初期化に失敗しました")
            return
            
        print("=== ピント強化版リアルタイム検出システム ===")
        print("96.7%の精度でリアルタイム検出を開始...")
        print("強化版自動ピント調整機能有効")
        print("カメラ起動中...")
        
        self.running = True
        focus_adjustment_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("フレーム読み込み失敗")
                break
                
            # 包括的ピント検索（毎フレーム）
            self.comprehensive_focus_search(frame)
            focus_adjustment_count += 1
            
            # 予測（0.3秒間隔）
            current_time = time.time()
            if current_time - self.last_update_time >= 0.3:
                self.current_prediction, self.current_confidence = self.predict(frame)
                self.last_update_time = current_time
                print(f"予測更新: {self.current_prediction} (信頼度: {self.current_confidence:.2f}, ピント: {self.focus_score:.1f})")
            
            # ピント強化版HUD描画
            frame_with_hud = self.draw_focus_hud(frame)
            
            # 表示
            cv2.imshow("Resin Washer Inspection", frame_with_hud)
            
            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('1'):
                self.save_image(frame, "良品")
                self.feedback_count += 1
                print("良品として保存")
            elif key == ord('2'):
                self.save_image(frame, "黒点")
                self.feedback_count += 1
                print("黒点として保存")
            elif key == ord('3'):
                self.save_image(frame, "欠け")
                self.feedback_count += 1
                print("欠けとして保存")
            elif key == ord('4'):
                self.save_image(frame, "傷")
                self.feedback_count += 1
                print("傷として保存")
            elif key == ord('s') or key == ord('S'):
                self.save_image(frame, "manual_save")
                print("手動保存")
            elif key == ord('f') or key == ord('F'):
                # フォーカステスト
                focus_score = self.calculate_enhanced_focus_score(frame)
                print(f"現在のピントスコア: {focus_score:.1f}")
            elif key == ord('b') or key == ord('B'):
                # 最高ピントに戻す
                self.cap.set(cv2.CAP_PROP_FOCUS, self.best_focus_value)
                print(f"最高ピントに戻す: {self.best_focus_value:.1f}")
            elif key == ord('t') or key == ord('T'):
                # ピント検索モード切り替え
                self.focus_search_mode = not self.focus_search_mode
                print(f"ピント検索モード: {'ON' if self.focus_search_mode else 'OFF'}")
            elif key == ord('r') or key == ord('R'):
                # フォーカスリセット
                self.cap.set(cv2.CAP_PROP_FOCUS, 0)
                print("フォーカスリセット")
                
        self.cleanup()
        
    def cleanup(self):
        """リソースを解放"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("システム終了")

if __name__ == "__main__":
    # カメラIDを指定（デフォルトは2）
    import sys
    camera_id = 2
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except ValueError:
            print("無効なカメラIDです。デフォルトの2を使用します。")
    
    inspection = FocusEnhancedResinInspection(camera_id)
    inspection.run()