#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
樹脂製ワッシャー 全カメラ同時起動版リアルタイム検出システム
すべてのカメラを同時に起動して表示する
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time
import threading
from pathlib import Path
import argparse

class AllCamerasInspection:
    """全カメラ同時起動版樹脂製ワッシャー検出システム"""
    
    def __init__(self):
        self.model = None
        self.cameras = {}  # カメラID: VideoCapture
        self.running = False
        self.current_prediction = "Unknown"
        self.current_confidence = 0.0
        self.class_names = ['良品', '欠け', '黒点', '傷']
        self.feedback_count = 0
        self.last_update_time = 0
        self.focus_scores = {}  # カメラID: focus_score
        self.frame_counts = {}  # カメラID: frame_count
        
        # モデル読み込み
        self.load_model()
        
        # 全カメラを同時起動
        self.init_all_cameras()
        
    def load_model(self):
        """高精度モデルを読み込む"""
        try:
            model_paths = [
                "resin_washer_model_enhanced/resin_washer_model_enhanced.h5",
                "resin_washer_model/resin_washer_model.h5"
            ]
            for model_path in model_paths:
                if os.path.exists(model_path):
                    self.model = keras.models.load_model(model_path)
                    print(f"モデル読み込み成功: {model_path}")
                    return
            print(f"エラー: モデルファイルが見つかりません。")
            self.model = None
        except Exception as e:
            print(f"モデル読み込み中にエラーが発生しました: {e}")
            self.model = None

    def test_camera_quick(self, index):
        """カメラのクイックテスト"""
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                return True, (width, height), fps
        cap.release()
        return False, (0, 0), 0.0

    def init_all_cameras(self):
        """全カメラを同時起動"""
        print("=== 全カメラ同時起動開始 ===")
        
        # 利用可能なカメラを検出
        available_cameras = []
        for i in range(10):
            success, res, fps = self.test_camera_quick(i)
            if success:
                available_cameras.append((i, res, fps))
                print(f"カメラ {i}: {res[0]}x{res[1]} @ {fps:.1f}FPS")
        
        print(f"=== 検出完了: {len(available_cameras)}個のカメラ ===")
        
        if not available_cameras:
            print("エラー: 利用可能なカメラが見つかりませんでした")
            return
        
        # 全カメラを同時起動
        for cam_id, res, fps in available_cameras:
            print(f"カメラ {cam_id} を起動中...")
            cap = cv2.VideoCapture(cam_id)
            
            if cap.isOpened():
                self.cameras[cam_id] = cap
                self.focus_scores[cam_id] = 0.0
                self.frame_counts[cam_id] = 0
                
                # カメラ設定
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                
                print(f"カメラ {cam_id} 起動成功!")
            else:
                print(f"カメラ {cam_id} 起動失敗")
        
        print(f"=== 起動完了: {len(self.cameras)}個のカメラが動作中 ===")

    def _preprocess_frame(self, frame):
        """フレームをモデル入力用に前処理する"""
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        return img

    def _predict_frame(self, frame):
        """フレームを予測する"""
        if self.model is None:
            self.current_prediction = "モデル未ロード"
            self.current_confidence = 0.0
            return

        preprocessed_frame = self._preprocess_frame(frame)
        predictions = self.model.predict(preprocessed_frame, verbose=0)[0]
        
        self.current_confidence = np.max(predictions)
        predicted_class_idx = np.argmax(predictions)
        self.current_prediction = self.class_names[predicted_class_idx]

    def _calculate_focus_score(self, frame):
        """画像のピントスコアを計算する"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def create_multi_camera_layout(self, frames):
        """複数カメラのレイアウトを作成"""
        if not frames:
            return None
        
        # カメラ数を取得
        num_cameras = len(frames)
        
        if num_cameras == 1:
            # 1カメラの場合
            return frames[0]
        elif num_cameras == 2:
            # 2カメラの場合：横並び
            h1, w1 = frames[0].shape[:2]
            h2, w2 = frames[1].shape[:2]
            max_h = max(h1, h2)
            
            # フレームをリサイズ
            frame1 = cv2.resize(frames[0], (w1, max_h))
            frame2 = cv2.resize(frames[1], (w2, max_h))
            
            return np.hstack([frame1, frame2])
        elif num_cameras == 3:
            # 3カメラの場合：2+1レイアウト
            h1, w1 = frames[0].shape[:2]
            h2, w2 = frames[1].shape[:2]
            h3, w3 = frames[2].shape[:2]
            
            # 上段：2カメラ横並び
            max_h_top = max(h1, h2)
            frame1 = cv2.resize(frames[0], (w1, max_h_top))
            frame2 = cv2.resize(frames[1], (w2, max_h_top))
            top_row = np.hstack([frame1, frame2])
            
            # 下段：1カメラ（中央配置）
            frame3 = cv2.resize(frames[2], (w3, h3))
            bottom_row = np.zeros((h3, top_row.shape[1], 3), dtype=np.uint8)
            start_x = (top_row.shape[1] - w3) // 2
            bottom_row[:, start_x:start_x+w3] = frame3
            
            return np.vstack([top_row, bottom_row])
        else:
            # 4カメラ以上の場合：2x2グリッド
            rows = []
            for i in range(0, min(num_cameras, 4), 2):
                row_frames = []
                for j in range(2):
                    if i + j < num_cameras:
                        row_frames.append(frames[i + j])
                
                if len(row_frames) == 2:
                    h1, w1 = row_frames[0].shape[:2]
                    h2, w2 = row_frames[1].shape[:2]
                    max_h = max(h1, h2)
                    
                    frame1 = cv2.resize(row_frames[0], (w1, max_h))
                    frame2 = cv2.resize(row_frames[1], (w2, max_h))
                    rows.append(np.hstack([frame1, frame2]))
                else:
                    # 単一フレームの場合、幅を統一
                    single_frame = row_frames[0]
                    h, w = single_frame.shape[:2]
                    # 他の行の幅に合わせる
                    target_width = rows[0].shape[1] if rows else w
                    resized_frame = cv2.resize(single_frame, (target_width, h))
                    rows.append(resized_frame)
            
            # すべての行の幅を統一
            if rows:
                max_width = max(row.shape[1] for row in rows)
                max_height = max(row.shape[0] for row in rows)
                
                resized_rows = []
                for row in rows:
                    resized_row = cv2.resize(row, (max_width, max_height))
                    resized_rows.append(resized_row)
                
                return np.vstack(resized_rows)
            
            return None

    def _draw_hud(self, frame, camera_id):
        """HUDを描画する"""
        h, w, _ = frame.shape
        
        # カメラID表示
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 255, 255)
        
        cv2.putText(frame, f"Camera {camera_id}", (10, 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Focus: {self.focus_scores[camera_id]:.1f}", (10, 60), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    def run(self):
        """全カメラ同時検出を開始する"""
        if not self.cameras:
            print("エラー: カメラが初期化されていません")
            return

        self.running = True
        print(f"=== 全カメラ同時起動版リアルタイム検出システム開始 ===")
        print(f"動作中カメラ数: {len(self.cameras)}")
        for cam_id in self.cameras.keys():
            print(f"  カメラ {cam_id}: 動作中")
        
        print("システム開始 - ESCキーで終了")
        
        while self.running:
            frames = {}
            valid_cameras = []
            
            # 全カメラからフレームを取得
            for cam_id, cap in self.cameras.items():
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames[cam_id] = frame
                    valid_cameras.append(cam_id)
                    
                    # ピントスコアを計算
                    self.focus_scores[cam_id] = self._calculate_focus_score(frame)
                    self.frame_counts[cam_id] += 1
                    
                    # HUDを描画
                    self._draw_hud(frame, cam_id)
            
            if frames:
                # 複数カメラのレイアウトを作成
                layout_frame = self.create_multi_camera_layout(list(frames.values()))
                
                if layout_frame is not None:
                    # 全体のHUDを描画
                    h, w, _ = layout_frame.shape
                    
                    # 右上の判定結果パネル
                    result_text = f"Result: {self.current_prediction}"
                    result_color = (0, 255, 0) if self.current_prediction == '良品' else (0, 0, 255)
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale_result = 1.0
                    font_thickness_result = 3

                    (text_w, text_h), baseline = cv2.getTextSize(result_text, font, font_scale_result, font_thickness_result)
                    
                    panel_x = w - text_w - 40
                    panel_y = 10
                    panel_width = text_w + 30
                    panel_height = text_h + baseline + 20

                    cv2.rectangle(layout_frame, (panel_x, panel_y), (w - 10, panel_y + panel_height), result_color, -1)
                    cv2.putText(layout_frame, result_text, (panel_x + 15, panel_y + text_h + 10), font, font_scale_result, (255, 255, 255), font_thickness_result, cv2.LINE_AA)

                    # 左下のHUD
                    hud_height = 150
                    hud_y_start = h - hud_height
                    cv2.rectangle(layout_frame, (0, hud_y_start), (w, h), (0, 0, 0), -1)

                    font_scale_hud = 0.6
                    font_thickness_hud = 1
                    text_color_hud = (255, 255, 255)

                    cv2.putText(layout_frame, f"Active Cameras: {len(valid_cameras)}", (10, hud_y_start + 20), font, font_scale_hud, text_color_hud, font_thickness_hud, cv2.LINE_AA)
                    cv2.putText(layout_frame, f"Feedback: {self.feedback_count}", (10, hud_y_start + 45), font, font_scale_hud, text_color_hud, font_thickness_hud, cv2.LINE_AA)
                    cv2.putText(layout_frame, "Controls:", (10, hud_y_start + 70), font, font_scale_hud, text_color_hud, font_thickness_hud, cv2.LINE_AA)
                    cv2.putText(layout_frame, "1:Good 2:Chipped 3:Black Spot 4:Scratched", (10, hud_y_start + 95), font, font_scale_hud, text_color_hud, font_thickness_hud, cv2.LINE_AA)
                    cv2.putText(layout_frame, "ESC:Exit", (10, hud_y_start + 120), font, font_scale_hud, text_color_hud, font_thickness_hud, cv2.LINE_AA)
                    
                    # 予測は一定間隔で行う（最初のカメラを使用）
                    current_time = time.time()
                    if current_time - self.last_update_time > 0.3 and valid_cameras:
                        self._predict_frame(frames[valid_cameras[0]])
                        self.last_update_time = current_time
                    
                    cv2.imshow("All Cameras Inspection", layout_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESCキーで終了
                self.running = False
            elif key == ord('f'):  # Fキーでピントスコア表示
                for cam_id in valid_cameras:
                    print(f"カメラ {cam_id} ピントスコア: {self.focus_scores[cam_id]:.1f}")
        
        # 全カメラを解放
        for cap in self.cameras.values():
            cap.release()
        cv2.destroyAllWindows()
        print("全カメラシステム終了")

if __name__ == "__main__":
    inspection_system = AllCamerasInspection()
    inspection_system.run()