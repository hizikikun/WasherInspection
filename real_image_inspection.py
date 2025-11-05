#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
実際の画像データで訓練されたモデルを使用したリアルタイム樹脂ワッシャー欠陥検査システム
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

class RealImageInspectionSystem:
    def __init__(self):
        self.model = None
        self.camera = None
        self.camera_id = 0  # カメラ0固定
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
        self.last_prediction = "---"
        self.prediction_stable_count = 0
        
        self.target_width = 1920
        self.target_height = 1080
        
        # UI設定
        self.window_title = "Resin Washer Inspection - Real Image Model"
        
    def load_model(self):
        """実際の画像データで訓練されたAIモデル読み込み"""
        try:
            # 実際の画像データで訓練されたモデルを読み込み
            model_path = "real_image_defect_model/defect_classifier_model.h5"
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"実際の画像データで訓練されたAIモデルを読み込みました: {model_path}")
            else:
                # フォールバック: 最良モデル
                fallback_path = "best_real_defect_model.h5"
                if os.path.exists(fallback_path):
                    self.model = tf.keras.models.load_model(fallback_path)
                    print(f"最良モデルを読み込みました: {fallback_path}")
                else:
                    print(f"警告: AIモデルが見つかりません")
                    self.model = None
        except Exception as e:
            print(f"AIモデルの読み込み中にエラーが発生しました: {e}")
            self.model = None

    def initialize_camera(self):
        """カメラの初期化と設定"""
        if self.camera:
            self.camera.release()
        
        print(f"カメラID {self.camera_id} を初期化中...")
        self.camera = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        
        if not self.camera.isOpened():
            print(f"エラー: カメラID {self.camera_id} を開けませんでした。")
            return False

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"カメラID {self.camera_id} を {actual_width}x{actual_height} で開きました。")
        
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
        """欠陥を予測（安定化処理付き）"""
        if self.model is None or frame is None:
            return "??? unknown", 0.0

        preprocessed_img = self.preprocess_frame(frame)
        if preprocessed_img is None:
            return "??? unknown", 0.0

        predictions = self.model.predict(preprocessed_img)
        score = np.max(predictions)
        class_id = np.argmax(predictions)
        
        if score < 0.5:  # 信頼度が低い場合は unknown
            return "??? unknown", score
        
        predicted_class = self.class_names[class_id]
        
        # 予測の安定化処理
        if predicted_class == self.last_prediction:
            self.prediction_stable_count += 1
        else:
            self.prediction_stable_count = 0
            self.last_prediction = predicted_class
        
        # 3回連続で同じ予測が出るまで待つ
        if self.prediction_stable_count < 3:
            return self.last_prediction, score
        
        return predicted_class, score

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
                print(f"✅ 全体画像を保存: {full_image_path}")
            else:
                print(f"❌ 全体画像の保存に失敗: {full_image_path}")
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
                    print(f"✅ 選択範囲画像を保存: {cropped_image_path}")
                else:
                    print(f"❌ 選択範囲画像の保存に失敗: {cropped_image_path}")
                    return
            
            # 選択状態をリセット
            self.selection_points = []
            self.selection_mode = False
            print(f"🎉 {self.class_names[class_id]}クラスの画像セットを保存しました！")
            
        except Exception as e:
            print(f"❌ 画像保存中にエラーが発生しました: {e}")

    def start_training(self):
        """バックグラウンドでディープラーニングを開始"""
        if self.training_thread and self.training_thread.is_alive():
            print("⚠️ 既に学習中です。")
            return
            
        self.is_training = True
        self.training_thread = threading.Thread(target=self._train_model)
        self.training_thread.daemon = True
        self.training_thread.start()
        print("🚀 ディープラーニングを開始しました...")

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
            print(f"❌ 学習中にエラーが発生しました: {e}")
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

        last_prediction_time = time.time()
        prediction_interval = 0.5  # 0.5秒間隔で予測

        print("\n--- 実際の画像データで訓練されたモデルによる検査システム起動 ---")
        print("ESCキーで終了")
        print("UI: 2枚目の画像デザイン")

        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("フレームを読み込めませんでした。")
                break

            # 現在のフレームを保存（画像保存用）
            self.current_frame = frame.copy()

            current_time = time.time()
            prediction_text = "---"
            confidence_score = 0.0
            
            if current_time - last_prediction_time >= prediction_interval:
                prediction_text, confidence_score = self.predict_defect(frame)
                last_prediction_time = current_time

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
            cv2.setWindowTitle(self.window_title, self.window_title)

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
                result_text = f"Result: NG - {prediction_text}" # NGの場合は詳細表示
                banner_color = (0, 0, 255) # 赤 (NG)

            banner_width = 350
            banner_height = 70
            banner_x = 10
            banner_y = 10
            cv2.rectangle(display_frame, (banner_x, banner_y), (banner_x + banner_width, banner_y + banner_height), banner_color, -1)
            cv2.putText(display_frame, result_text, (banner_x + 10, banner_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # 信頼度表示
            if confidence_score > 0:
                confidence_text = f"Confidence: {confidence_score:.2f}"
                cv2.putText(display_frame, confidence_text, (banner_x + 10, banner_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # 左下の黒いコントロールパネル
            panel_width = 450
            panel_height = 140
            panel_x = 10
            panel_y = display_frame.shape[0] - panel_height - 10
            cv2.rectangle(display_frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
            
            text_color = (255, 255, 255)  # 白
            font_scale = 0.7
            thickness = 2
            line_height = 25

            cv2.putText(display_frame, f"Feedback: {self.feedback_count}", (panel_x + 10, panel_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            cv2.putText(display_frame, "Controls", (panel_x + 10, panel_y + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            cv2.putText(display_frame, "1:Good 2:Black 3:Chipped 4:Scratched", (panel_x + 10, panel_y + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            cv2.putText(display_frame, "M:Select S:Train ESC:Exit", (panel_x + 10, panel_y + line_height * 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            
            # モデル情報表示
            cv2.putText(display_frame, "Model: Real Image Trained", (panel_x + 10, panel_y + line_height * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(self.window_title, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESCキーで終了
                break
            elif key == ord('1'):  # 良品フィードバック
                self.feedback_count += 1
                print("良品フィードバック記録")
            elif key == ord('2'):  # 黒点フィードバック
                self.feedback_count += 1
                print("黒点フィードバック記録")
            elif key == ord('3'):  # 欠けフィードバック
                self.feedback_count += 1
                print("欠けフィードバック記録")
            elif key == ord('4'):  # 傷フィードバック
                self.feedback_count += 1
                print("傷フィードバック記録")
            elif key == ord('m') or key == ord('M'):  # Mキーで選択モード開始
                self.selection_mode = True
                self.selection_points = []
                print("Mキー: 欠陥領域選択モード開始 - マウスで4点をクリックしてください")
            elif key == ord('s') or key == ord('S'):  # Sキーで学習開始
                print("Sキー: ディープラーニング開始")
                self.start_training()
            elif key == ord('1'):  # 良品
                if len(self.selection_points) == 4:
                    self.save_images(0)
                else:
                    print("⚠️ 四点選択後に1キーを押してください")
            elif key == ord('2'):  # 黒点
                if len(self.selection_points) == 4:
                    self.save_images(2)
                else:
                    print("⚠️ 四点選択後に2キーを押してください")
            elif key == ord('3'):  # 欠け
                if len(self.selection_points) == 4:
                    self.save_images(1)
                else:
                    print("⚠️ 四点選択後に3キーを押してください")
            elif key == ord('4'):  # 傷
                if len(self.selection_points) == 4:
                    self.save_images(3)
                else:
                    print("⚠️ 四点選択後に4キーを押してください")

        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = RealImageInspectionSystem()
    system.run()
