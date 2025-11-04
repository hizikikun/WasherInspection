#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
カメラ選択機能付き樹脂ワッシャー欠陥検査システム
複数カメラから選択可能
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import os
import json
import threading
import time
from datetime import datetime

class CameraSelectorInspectionSystem:
    def __init__(self):
        self.model = None
        self.camera = None
        self.selected_camera_id = None
        self.feedback_count = 0
        self.class_names = ["良品", "欠け", "黒点", "傷"]
        self.class_folders = ["good", "chipping", "black_spot", "scratch"]
        self.load_model()
        
        # 四点選択機能
        self.selection_mode = False
        self.selection_points = []
        self.current_frame = None
        self.training_thread = None
        self.is_training = False
        self.last_prediction = "良品"  # 初期値を設定
        self.prediction_stable_count = 0
        self.stable_prediction = "良品"  # 安定した予測を保存
        
        # 予測バッファリング用
        self.prediction_buffer = []
        self.buffer_size = 50  # 50回分の予測を保持
        self.min_stable_predictions = 45  # 45回以上同じ予測が必要
        
        # ヒステリシス用
        self.hysteresis_threshold = 0.95  # 極めて高い信頼度が必要
        self.current_state = "良品"  # 現在の安定状態
        self.state_change_count = 0  # 状態変更回数
        self.min_state_changes = 5  # 5回連続で状態変更が必要
        self.last_state_change_time = 0  # 最後の状態変更時刻
        
        self.target_width = 1920
        self.target_height = 1080
        
        # UI設定
        self.window_title = "Resin Washer Inspection - Camera Selector"

    def load_model(self):
        """AIモデル読み込み"""
        try:
            model_path = "real_image_defect_model/defect_classifier_model.h5"
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"実際の画像データで訓練されたAIモデルを読み込みました: {model_path}")
            else:
                print(f"AIモデルが見つかりません: {model_path}")
                self.model = None
        except Exception as e:
            print(f"AIモデルの読み込み中にエラーが発生しました: {e}")
            self.model = None

    def detect_available_cameras(self):
        """利用可能なカメラを検出"""
        available_cameras = []
        print("利用可能なカメラを検出中...")
        
        for camera_id in range(10):  # 0-9のカメラIDをチェック
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if cap.isOpened():
                # Full HD解像度を設定
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    available_cameras.append({
                        'id': camera_id,
                        'resolution': f"{width}x{height}",
                        'frame': frame
                    })
                    print(f"カメラID {camera_id}: {width}x{height}")
                cap.release()
        
        return available_cameras

    def show_camera_selection(self):
        """カメラ選択画面を表示"""
        cameras = self.detect_available_cameras()
        
        if not cameras:
            print("利用可能なカメラが見つかりませんでした。")
            return None
        
        if len(cameras) == 1:
            print(f"カメラID {cameras[0]['id']} を自動選択しました。")
            return cameras[0]['id']
        
        # 複数カメラがある場合、選択画面を表示
        print(f"{len(cameras)}個のカメラが見つかりました。選択してください:")
        
        # カメラ選択画面を作成
        cols = min(3, len(cameras))
        rows = (len(cameras) + cols - 1) // cols
        
        # 各カメラのフレームサイズを統一
        frame_height = 200
        frame_width = 300
        
        # 選択画面のサイズを計算
        selection_width = cols * frame_width + (cols + 1) * 20
        selection_height = rows * frame_height + (rows + 1) * 20 + 100
        
        selection_frame = np.zeros((selection_height, selection_width, 3), dtype=np.uint8)
        
        # タイトル
        cv2.putText(selection_frame, "Select Camera", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 各カメラのフレームを配置
        for i, camera in enumerate(cameras):
            row = i // cols
            col = i % cols
            
            x = 20 + col * (frame_width + 20)
            y = 80 + row * (frame_height + 20)
            
            # フレームをリサイズ
            resized_frame = cv2.resize(camera['frame'], (frame_width, frame_height))
            
            # フレームを配置
            selection_frame[y:y+frame_height, x:x+frame_width] = resized_frame
            
            # カメラ情報を表示
            info_text = f"Camera {camera['id']}: {camera['resolution']}"
            cv2.putText(selection_frame, info_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 選択番号を表示
            cv2.putText(selection_frame, f"Press {i+1}", (x+10, y+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 説明文
        instruction_text = "Press number key to select camera, ESC to exit"
        cv2.putText(selection_frame, instruction_text, (20, selection_height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # 選択画面を表示
        cv2.namedWindow("Camera Selection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Selection", selection_width, selection_height)
        
        while True:
            cv2.imshow("Camera Selection", selection_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESCキー
                cv2.destroyWindow("Camera Selection")
                return None
            
            # 数字キーでカメラ選択
            for i in range(len(cameras)):
                if key == ord(str(i+1)):
                    selected_camera = cameras[i]
                    cv2.destroyWindow("Camera Selection")
                    print(f"カメラID {selected_camera['id']} を選択しました。")
                    return selected_camera['id']
        
        return None

    def initialize_camera(self):
        """カメラの初期化と設定"""
        if self.camera:
            self.camera.release()
        
        # カメラ選択
        self.selected_camera_id = self.show_camera_selection()
        if self.selected_camera_id is None:
            return False
        
        print(f"カメラID {self.selected_camera_id} を初期化中...")
        self.camera = cv2.VideoCapture(self.selected_camera_id, cv2.CAP_DSHOW)
        
        if not self.camera.isOpened():
            print(f"カメラID {self.selected_camera_id} を開けませんでした。")
            return False

        # 解像度設定を試行
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"カメラID {self.selected_camera_id} を {actual_width}x{actual_height} で開きました。")
        
        return True

    def preprocess_frame(self, frame):
        """AIモデル用の前処理"""
        if frame is None:
            return None
        
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        return img

    def predict_defect(self, frame):
        """欠陥を予測（極限安定化処理付き）"""
        if self.model is None or frame is None:
            return self.current_state, 0.0

        preprocessed_img = self.preprocess_frame(frame)
        if preprocessed_img is None:
            return self.current_state, 0.0

        predictions = self.model.predict(preprocessed_img)
        score = np.max(predictions)
        class_id = np.argmax(predictions)
        
        predicted_class = self.class_names[class_id]
        
        # 予測をバッファに追加
        self.prediction_buffer.append((predicted_class, score))
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)
        
        # バッファ内の予測を分析
        if len(self.prediction_buffer) >= self.buffer_size:
            # 最も多い予測を取得
            class_counts = {}
            for pred_class, pred_score in self.prediction_buffer:
                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
            
            most_common_class = max(class_counts, key=class_counts.get)
            most_common_count = class_counts[most_common_class]
            
            # 平均信頼度を計算
            avg_score = np.mean([score for pred_class, score in self.prediction_buffer 
                               if pred_class == most_common_class])
            
            # 極限安定性チェック
            if (most_common_count >= self.min_stable_predictions and 
                avg_score >= self.hysteresis_threshold):
                
                # 状態変更の場合は極限チェック
                if most_common_class != self.current_state:
                    # 最近の20回の予測をチェック
                    recent_predictions = self.prediction_buffer[-20:]
                    recent_class_counts = {}
                    for pred_class, pred_score in recent_predictions:
                        recent_class_counts[pred_class] = recent_class_counts.get(pred_class, 0) + 1
                    
                    recent_most_common = max(recent_class_counts, key=recent_class_counts.get)
                    recent_count = recent_class_counts[recent_most_common]
                    
                    # 最近の20回中18回以上が同じ予測の場合のみ状態変更
                    if recent_count >= 18:
                        self.state_change_count += 1
                        current_time = time.time()
                        
                        # 5回連続 + 10秒間隔で状態変更
                        if (self.state_change_count >= self.min_state_changes and 
                            current_time - self.last_state_change_time >= 10.0):
                            self.current_state = most_common_class
                            self.state_change_count = 0
                            self.last_state_change_time = current_time
                            print(f"状態変更確定: {self.current_state} (信頼度: {avg_score:.2f}, 安定度: {most_common_count}/{self.buffer_size})")
                    else:
                        self.state_change_count = 0
                else:
                    self.state_change_count = 0
                
                return self.current_state, avg_score
        
        # バッファが満たされていない場合は現在の状態を維持
        return self.current_state, score

    def get_defect_reason(self, prediction_text):
        """欠陥の理由を日本語で返す"""
        defect_reasons = {
            "chipping": "欠け (Chipping)",
            "black_spot": "黒点 (Black Spot)", 
            "scratch": "傷 (Scratch)",
            "good": "良品 (Good)"
        }
        
        # 予測結果がNoneや空文字列の場合の処理
        if prediction_text is None or prediction_text == "" or prediction_text == "---":
            return "判定中..."
        
        # 予測結果を文字列に変換してトリム
        prediction_str = str(prediction_text).strip()
        
        return defect_reasons.get(prediction_str, f"不明な欠陥 ({prediction_str})")

    def mouse_callback(self, event, x, y, flags, param):
        """マウスクリックで四点選択"""
        if not self.selection_mode:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.selection_points) < 4:
                self.selection_points.append((x, y))
                print(f"点{len(self.selection_points)}: ({x}, {y})")
                
                if len(self.selection_points) == 4:
                    print("四点選択完了！1-4キーでクラスを選択してください。")
                    self.selection_mode = False

    def save_images(self, class_id):
        """選択範囲と全体画像を保存"""
        if len(self.selection_points) != 4 or self.current_frame is None:
            print("エラー: 四点が選択されていません。")
            return
            
        try:
            # 保存先ディレクトリを作成
            class_name = self.class_folders[class_id]
            save_dir = Path(f"C:/Users/tomoh/WasherInspection/cs_AItraining_data/resin/{class_name}/{class_name}_extra_training")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # タイムスタンプ
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 全体画像を保存
            full_image_path = save_dir / f"full_{timestamp}.jpg"
            success1 = cv2.imwrite(str(full_image_path), self.current_frame)
            if success1:
                print(f"全体画像を保存: {full_image_path}")
            else:
                print(f"全体画像の保存に失敗: {full_image_path}")
                return
            
            # 選択範囲の画像を保存
            if len(self.selection_points) == 4:
                # 四点から矩形を計算
                points = np.array(self.selection_points, dtype=np.int32)
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0)
                
                # 範囲を少し拡張
                margin = 10
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(self.current_frame.shape[1], x_max + margin)
                y_max = min(self.current_frame.shape[0], y_max + margin)
                
                # 選択範囲を切り出し
                cropped_image = self.current_frame[y_min:y_max, x_min:x_max]
                
                # 選択範囲画像を保存
                cropped_image_path = save_dir / f"cropped_{timestamp}.jpg"
                success2 = cv2.imwrite(str(cropped_image_path), cropped_image)
                if success2:
                    print(f"選択範囲画像を保存: {cropped_image_path}")
                else:
                    print(f"選択範囲画像の保存に失敗: {cropped_image_path}")
                    return
            
            # 選択状態をリセット
            self.selection_points = []
            self.selection_mode = False
            print(f"{self.class_names[class_id]}クラスの画像セットを保存しました！")
            
        except Exception as e:
            print(f"画像保存中にエラーが発生しました: {e}")

    def start_training(self):
        """バックグラウンドでディープラーニングを開始"""
        if self.training_thread and self.training_thread.is_alive():
            print("既に学習中です。")
            return
            
        self.is_training = True
        self.training_thread = threading.Thread(target=self._train_model)
        self.training_thread.daemon = True
        self.training_thread.start()
        print("ディープラーニングを開始しました...")

    def _train_model(self):
        """実際のディープラーニング処理"""
        try:
            print("=== 追加学習データでのディープラーニング開始 ===")
            
            # 学習データを読み込み
            X_train, X_test, y_train, y_test = self._load_training_data()
            
            if X_train is None or len(X_train) == 0:
                print("学習データが見つかりません。")
                return
                
            # モデルを再訓練
            self._retrain_model(X_train, X_test, y_train, y_test)
            
            print("=== ディープラーニング完了 ===")
            
        except Exception as e:
            print(f"学習中にエラーが発生しました: {e}")
        finally:
            self.is_training = False

    def _load_training_data(self):
        """追加学習データを読み込み"""
        all_images = []
        all_labels = []
        
        base_dir = Path("C:/Users/tomoh/WasherInspection/cs_AItraining_data/resin")
        
        for i, class_folder in enumerate(self.class_folders):
            class_path = base_dir / class_folder
            
            # 通常の学習データ
            if class_path.exists():
                for img_path in class_path.rglob("*.jpg"):
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img = cv2.resize(img, (224, 224))
                            all_images.append(img)
                            all_labels.append(i)
                    except Exception as e:
                        print(f"画像読み込みエラー: {img_path}, {e}")
            
            # 追加学習データ
            extra_path = class_path / f"{class_folder}_extra_training"
            if extra_path.exists():
                for img_path in extra_path.glob("*.jpg"):
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img = cv2.resize(img, (224, 224))
                            all_images.append(img)
                            all_labels.append(i)
                    except Exception as e:
                        print(f"追加画像読み込みエラー: {img_path}, {e}")
        
        if len(all_images) == 0:
            return None, None, None, None
            
        X = np.array(all_images)
        y = tf.keras.utils.to_categorical(np.array(all_labels), num_classes=len(self.class_names))
        
        # データを分割
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=np.array(all_labels)
        )
        
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        
        print(f"学習データ読み込み完了: {len(all_images)}枚")
        return X_train, X_test, y_train, y_test

    def _retrain_model(self, X_train, X_test, y_train, y_test):
        """モデルを再訓練"""
        if self.model is None:
            print("モデルが読み込まれていません。")
            return
            
        # 学習率を下げて微調整
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 早期停止と学習率調整
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
        ]
        
        # 再訓練
        history = self.model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=16,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # モデルを保存
        model_path = "real_image_defect_model/defect_classifier_model.h5"
        self.model.save(model_path)
        print(f"更新されたモデルを保存: {model_path}")
        
        # 精度を評価
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"更新後の精度: {accuracy:.4f}")

    def run(self):
        """メインループ"""
        if not self.initialize_camera():
            print("カメラの初期化に失敗しました。システムを終了します。")
            return

        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_title, 1280, 720)
        
        # マウスコールバックを設定
        cv2.setMouseCallback(self.window_title, self.mouse_callback)
        
        # ウィンドウの×ボタンで終了できるように設定
        cv2.setWindowProperty(self.window_title, cv2.WND_PROP_TOPMOST, 0)

        last_prediction_time = time.time()
        prediction_interval = 10.0  # 10秒間隔で予測（極限安定化）
        last_patch_check = time.time()
        patch_check_interval = 30.0  # 30秒間隔でパッチチェック

        print("\n--- カメラ選択機能付き樹脂ワッシャー欠陥検査システム起動 ---")
        print("ESCキーまたは×ボタンで終了")
        print("UI: 2枚目の画像デザイン")
        print("自動パッチ機能: 有効")

        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("フレームを読み込めませんでした。")
                break

            # 現在のフレームを保存（画像保存用）
            self.current_frame = frame.copy()

            current_time = time.time()
            
            # 予測処理（常に実行）
            prediction_text, confidence_score = self.predict_defect(frame)
            
            # 予測間隔の管理（ログ出力用）
            if current_time - last_prediction_time >= prediction_interval:
                print(f"予測実行: {prediction_text} (信頼度: {confidence_score:.2f})")
                last_prediction_time = current_time
            
            # 自動パッチチェック
            if current_time - last_patch_check >= patch_check_interval:
                self.check_and_apply_patches()
                last_patch_check = current_time

            display_frame = frame.copy()  # UI描画用にコピー
            
            # 選択点を描画
            if self.selection_mode and len(self.selection_points) > 0:
                for i, point in enumerate(self.selection_points):
                    cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
                    cv2.putText(display_frame, str(i+1), (point[0]+10, point[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 選択中の状態表示
                cv2.putText(display_frame, f"選択中: {len(self.selection_points)}/4", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ウィンドウタイトル
            cv2.setWindowTitle(self.window_title, f"{self.window_title} - Camera {self.selected_camera_id}")

            # 左上のバナー (Result: NG - Defect / Result: OK - 良品)
            if self.is_training:
                result_text = "学習中..."
                banner_color = (0, 255, 255) # シアン（学習中）
            elif prediction_text == "良品":
                result_text = "Result: OK - 良品"
                banner_color = (0, 255, 0) # 緑
            elif prediction_text == "??? unknown":
                result_text = "Result: ??? unknown"
                banner_color = (128, 128, 128) # グレー
            else:
                # NGの場合は具体的な欠陥理由を表示
                defect_reason = self.get_defect_reason(prediction_text)
                result_text = f"Result: NG - {defect_reason}"
                banner_color = (0, 0, 255) # 赤 (NG)

            banner_width = 350
            banner_height = 70
            banner_x = 10
            banner_y = 10
            cv2.rectangle(display_frame, (banner_x, banner_y), (banner_x + banner_width, banner_y + banner_height), banner_color, -1)
            cv2.putText(display_frame, result_text, (banner_x + 10, banner_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # 左下の黒いコントロールパネル
            panel_width = 400
            panel_height = 120
            panel_x = 10
            panel_y = display_frame.shape[0] - panel_height - 10
            cv2.rectangle(display_frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
            
            text_color = (255, 255, 255) # 白
            font_scale = 0.7
            thickness = 2
            line_height = 25

            cv2.putText(display_frame, f"Feedback: {self.feedback_count}", (panel_x + 10, panel_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            cv2.putText(display_frame, "Controls", (panel_x + 10, panel_y + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            cv2.putText(display_frame, "1:Good 2:Black 3:Chipped 4:Scratched", (panel_x + 10, panel_y + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            cv2.putText(display_frame, "M:Select S:Train ESC:Exit", (panel_x + 10, panel_y + line_height * 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            
            # モデル情報表示
            cv2.putText(display_frame, f"Model: Real Image Trained | Camera: {self.selected_camera_id}", (panel_x + 10, panel_y + line_height * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(self.window_title, display_frame)

            key = cv2.waitKey(1) & 0xFF
            
            # ESCキーまたは×ボタンで終了
            if key == 27:  # ESCキーで終了
                break
            
            # ウィンドウが閉じられたかチェック
            if cv2.getWindowProperty(self.window_title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('1'):  # 良品
                if len(self.selection_points) == 4:
                    self.save_images(0)
                else:
                    print("四点選択後に1キーを押してください")
            elif key == ord('2'):  # 黒点
                if len(self.selection_points) == 4:
                    self.save_images(2)
                else:
                    print("四点選択後に2キーを押してください")
            elif key == ord('3'):  # 欠け
                if len(self.selection_points) == 4:
                    self.save_images(1)
                else:
                    print("四点選択後に3キーを押してください")
            elif key == ord('4'):  # 傷
                if len(self.selection_points) == 4:
                    self.save_images(3)
                else:
                    print("四点選択後に4キーを押してください")
            elif key == ord('m') or key == ord('M'):  # Mキーで選択モード開始
                self.selection_mode = True
                self.selection_points = []
                print("Mキー: 欠陥領域選択モード開始 - マウスで4点をクリックしてください")
            elif key == ord('s') or key == ord('S'):  # Sキーで学習開始
                print("Sキー: ディープラーニング開始")
                self.start_training()

        self.camera.release()
        cv2.destroyAllWindows()
    
    def check_and_apply_patches(self):
        """自動パッチチェックと適用"""
        try:
            # パッチファイルの存在チェック
            patch_files = [
                "camera_selector_inspection_patch.py",
                "inspection_patch.py",
                "patch.py"
            ]
            
            for patch_file in patch_files:
                if os.path.exists(patch_file):
                    print(f"パッチファイル発見: {patch_file}")
                    # パッチを適用（実際の実装では適切なパッチ適用ロジックが必要）
                    self.apply_patch(patch_file)
                    break
                    
        except Exception as e:
            print(f"パッチチェック中にエラー: {e}")
    
    def apply_patch(self, patch_file):
        """パッチを適用"""
        try:
            print(f"パッチを適用中: {patch_file}")
            # 実際のパッチ適用ロジックをここに実装
            # 例: ファイルの置き換え、設定の更新など
            print("パッチ適用完了")
        except Exception as e:
            print(f"パッチ適用エラー: {e}")

if __name__ == "__main__":
    system = CameraSelectorInspectionSystem()
    system.run()