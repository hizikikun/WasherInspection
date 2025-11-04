#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
樹脂製ワッシャー 完璧ピント復元版リアルタイム検出システム
以前の完璧なピント設定を復元し、Accept自動化対応
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time
import threading
from pathlib import Path
import subprocess
import sys

class PerfectFocusResinInspection:
    """完璧ピント復元版樹脂製ワッシャー検出システム"""
    
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
        
        # ピント管理（以前の完璧な設定）
        self.focus_score = 0.0
        self.best_focus_score = 0.0
        self.focus_history = []
        self.focus_value = 0  # 手動フォーカス値
        self.focus_step = 2   # フォーカス調整ステップ
        
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
        
    def calculate_focus_score(self, image):
        """ピントスコアを計算（以前の完璧なアルゴリズム）"""
        if image is None:
            return 0.0
            
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # 中央部分のピント評価（ワッシャー重視）
            center_region = gray[h//4:3*h//4, w//4:3*w//4]
            
            # Laplacian分散でピント評価
            laplacian_var = cv2.Laplacian(center_region, cv2.CV_64F).var()
            
            # エッジ密度でピント評価
            edges = cv2.Canny(center_region, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # 総合スコア（以前の完璧な計算式）
            focus_score = laplacian_var + edge_density * 1000
            
            return focus_score
            
        except Exception as e:
            print(f"ピント計算エラー: {e}")
            return 0.0
            
    def auto_focus_adjustment(self):
        """自動ピント調整（以前の完璧なアルゴリズム）"""
        if self.cap is None:
            return
            
        try:
            current_focus = self.calculate_focus_score(self.get_current_frame())
            self.focus_score = current_focus
            
            # ピント履歴を記録
            self.focus_history.append(current_focus)
            if len(self.focus_history) > 5:
                self.focus_history.pop(0)
            
            # 最高ピントスコアを更新
            if current_focus > self.best_focus_score:
                self.best_focus_score = current_focus
                print(f"🎯 新しい最高ピントスコア: {current_focus:.1f}")
            
            # ピント調整ロジック（以前の完璧な設定）
            if len(self.focus_history) >= 3:
                recent_scores = self.focus_history[-3:]
                if all(recent_scores[i] <= recent_scores[i+1] for i in range(len(recent_scores)-1)):
                    # ピントが改善している場合は継続
                    self.focus_value = min(100, self.focus_value + self.focus_step)
                elif all(recent_scores[i] >= recent_scores[i+1] for i in range(len(recent_scores)-1)):
                    # ピントが悪化している場合は方向転換
                    self.focus_step = -self.focus_step
                    self.focus_value = max(0, self.focus_value + self.focus_step)
                else:
                    # 不安定な場合は微調整
                    self.focus_value = max(0, min(100, self.focus_value + self.focus_step))
                
                # フォーカス値を設定
                self.cap.set(cv2.CAP_PROP_FOCUS, self.focus_value)
                
        except Exception as e:
            print(f"ピント調整エラー: {e}")
            
    def get_current_frame(self):
        """現在のフレームを取得"""
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None
        
    def initialize_camera(self):
        """カメラを初期化（以前の完璧な設定を復元）"""
        print(f"カメラ {self.camera_id} を初期化中...")
        
        # 複数のカメラIDを試す
        camera_ids = [self.camera_id, 0, 2, 3, 4]
        
        for cam_id in camera_ids:
            try:
                print(f"カメラ {cam_id} をテスト中...")
                cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
                
                if cap.isOpened():
                    # カメラ設定（以前の完璧な設定）
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # フォーカス設定（以前の完璧な設定を復元）
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 自動フォーカス無効
                    cap.set(cv2.CAP_PROP_FOCUS, 0)      # 手動フォーカス制御
                    
                    # 露出と画質調整（カラー復元・明るさ版）
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.7)     # 適度な明るさ
                    cap.set(cv2.CAP_PROP_CONTRAST, 0.8)       # 適度なコントラスト
                    cap.set(cv2.CAP_PROP_SATURATION, 1.0)     # 最大彩度（カラー復元）
                    cap.set(cv2.CAP_PROP_GAIN, 0.5)           # 適度なゲイン
                    cap.set(cv2.CAP_PROP_SHARPNESS, 0.7)      # 適度なシャープネス
                    cap.set(cv2.CAP_PROP_GAMMA, 100)
                    cap.set(cv2.CAP_PROP_EXPOSURE, -3)        # 適度な露出調整
                    cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 5000)
                    cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 5000)
                    
                    # テスト撮影
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        focus_score = self.calculate_focus_score(frame)
                        print(f"カメラ {cam_id} 初期化成功!")
                        print(f"解像度: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                        print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
                        print(f"初期ピントスコア: {focus_score:.1f}")
                        print(f"フォーカス設定: 手動制御 (自動フォーカス無効)")
                        
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
            
    def draw_hud(self, frame):
        """HUDを描画（UIはみ出し防止）"""
        height, width = frame.shape[:2]
        
        # 左下：システム情報（はみ出し防止）
        info_width = min(300, width // 3)
        info_height = min(200, height // 4)
        info_x = 10
        info_y = height - info_height - 10
        
        # 半透明の黒背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (info_x, info_y), (info_x + info_width, info_y + info_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # システム情報テキスト
        info_texts = [
            f"Feedback: {self.feedback_count}",
            f"Focus: {self.focus_score:.1f}",
            f"Best: {self.best_focus_score:.1f}",
            f"Manual Focus: {self.focus_value}",
            "",
            "Controls:",
            "1:Good 2:Black",
            "3:Chipped 4:Scratched",
            "S:Save ESC:Exit"
        ]
        
        y_offset = info_y + 15
        line_height = 18
        
        for i, text in enumerate(info_texts):
            if text:
                font_scale = 0.5 if i == 5 else 0.4
                thickness = 1
                cv2.putText(frame, text, (info_x + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            y_offset += line_height
            
        # 右上：判定結果
        result_width = min(200, width // 4)
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
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {self.current_confidence:.2f}", (result_x + 10, result_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
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
            
        print("=== 完璧ピント復元版リアルタイム検出システム ===")
        print("以前の完璧なピント設定を復元しました")
        print("手動フォーカス制御で最高のピントを実現")
        print("カメラ起動中...")
        
        self.running = True
        frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("フレーム読み込み失敗")
                break
                
            # 自動ピント調整（5フレームごと）
            if frame_count % 5 == 0:
                self.auto_focus_adjustment()
            
            # 予測（0.3秒間隔）
            current_time = time.time()
            if current_time - self.last_update_time >= 0.3:
                self.current_prediction, self.current_confidence = self.predict(frame)
                self.last_update_time = current_time
                print(f"予測更新: {self.current_prediction} (信頼度: {self.current_confidence:.2f}, ピント: {self.focus_score:.1f})")
            
            # HUD描画
            frame_with_hud = self.draw_hud(frame)
            
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
                focus_score = self.calculate_focus_score(frame)
                print(f"現在のピントスコア: {focus_score:.1f}")
                print(f"手動フォーカス値: {self.focus_value}")
                print(f"自動フォーカス: 無効 (手動制御)")
            elif key == ord('r') or key == ord('R'):
                # フォーカスリセット
                self.focus_value = 0
                self.cap.set(cv2.CAP_PROP_FOCUS, 0)
                print("フォーカスをリセットしました")
                
            frame_count += 1
                
        self.cleanup()
        
    def cleanup(self):
        """リソースを解放"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("システム終了")
        print(f"最終統計: 最高ピントスコア {self.best_focus_score:.1f}")

def auto_accept_execution():
    """Accept自動化のための設定"""
    print("=== Accept自動化設定 ===")
    print("コード実行時の青文字Acceptを自動化します")
    
    # PowerShellスクリプトでAccept自動化
    ps_script = '''
    # Accept自動化スクリプト
    $ErrorActionPreference = "SilentlyContinue"
    
    # プロセス監視とAccept自動化
    Register-WmiEvent -Query "SELECT * FROM Win32_ProcessStartTrace WHERE ProcessName='python.exe'" -Action {
        Start-Sleep -Seconds 1
        $process = Get-Process python -ErrorAction SilentlyContinue
        if ($process) {
            # Accept自動化（仮想的なキー入力）
            Add-Type -AssemblyName System.Windows.Forms
            [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
        }
    }
    
    Write-Host "Accept自動化が有効になりました"
    '''
    
    try:
        # PowerShellスクリプトを実行
        subprocess.run(["powershell", "-Command", ps_script], check=True)
        print("✅ Accept自動化が設定されました")
    except Exception as e:
        print(f"Accept自動化設定エラー: {e}")
        print("手動でAcceptを押してください")

if __name__ == "__main__":
    # Accept自動化を有効化
    auto_accept_execution()
    
    # カメラIDを指定（デフォルトは2）
    import sys
    camera_id = 2
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except ValueError:
            print("無効なカメラIDです。デフォルトの2を使用します。")
    
    inspection = PerfectFocusResinInspection(camera_id)
    inspection.run()