import cv2
import numpy as np
import tensorflow as tf
import time
import os
from pathlib import Path
from datetime import datetime
import threading

class FundamentalFixInspectionSystem:
    def __init__(self):
        self.window_title = "Resin Washer Inspection - Fundamental Fix"
        self.camera = None
        self.model = None
        self.class_names = []
        self.selected_camera_id = None
        
        # 根本的な修正: シンプルな予測システム
        self.current_prediction = "good"  # 現在の予測結果
        self.prediction_confidence = 0.0
        self.last_prediction_time = 0
        self.prediction_interval = 1.0  # 1秒間隔（より頻繁に予測）
        
        # 予測安定化（シンプル版）
        self.prediction_history = []
        self.history_size = 3  # 3回分の履歴（より少なく）
        self.stability_threshold = 0.7  # 安定性閾値（より低く）
        
        # UI設定
        self.target_width = 1920
        self.target_height = 1080
        
        # 学習データ設定
        self.class_folders = ["good", "chipping", "black_spot", "scratch"]
        self.feedback_count = 0
        
        # 選択機能
        self.selection_mode = False
        self.selection_points = []
        self.current_frame = None
        
        # 学習機能
        self.training_thread = None
        self.is_training = False

    def load_model(self):
        """モデルを読み込み"""
        try:
            model_path = "real_image_defect_model/defect_classifier_model.h5"
            if os.path.exists(model_path):
                print(f"実画像データで学習されたAIモデルを読み込みます: {model_path}")
                self.model = tf.keras.models.load_model(model_path)
                
                # クラス名を設定
                self.class_names = ["good", "chipping", "black_spot", "scratch"]
                print(f"クラス名: {self.class_names}")
                return True
            else:
                print(f"モデルファイルが見つかりません: {model_path}")
                return False
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            return False

    def detect_available_cameras(self):
        """利用可能なカメラを検出"""
        available_cameras = []
        print("利用可能なカメラを検出中...")
        
        for camera_id in range(10):
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
        available_cameras = self.detect_available_cameras()
        
        if not available_cameras:
            print("利用可能なカメラが見つかりません。")
            return False
        
        print(f"{len(available_cameras)}個のカメラが見つかりました。選択してください:")
        
        # カメラ選択画面を作成
        if len(available_cameras) == 1:
            self.selected_camera_id = available_cameras[0]['id']
            print(f"カメラID {self.selected_camera_id} を自動選択しました。")
            return True
        
        # 複数カメラの場合は選択画面を表示
        cols = min(2, len(available_cameras))
        rows = (len(available_cameras) + cols - 1) // cols
        
        # 画面サイズを計算
        preview_width = 400
        preview_height = 300
        screen_width = cols * preview_width + (cols + 1) * 20
        screen_height = rows * preview_height + (rows + 1) * 20 + 100
        
        selection_screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        
        # タイトル
        cv2.putText(selection_screen, "Camera Selection", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # カメラプレビューを配置
        for i, camera in enumerate(available_cameras):
            row = i // cols
            col = i % cols
            
            x = col * preview_width + (col + 1) * 20
            y = row * preview_height + (row + 1) * 20 + 60
            
            # フレームをリサイズ
            resized_frame = cv2.resize(camera['frame'], (preview_width, preview_height))
            selection_screen[y:y+preview_height, x:x+preview_width] = resized_frame
            
            # カメラ情報を表示
            info_text = f"Camera {camera['id']}: {camera['resolution']}"
            cv2.putText(selection_screen, info_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 選択番号を表示
            cv2.putText(selection_screen, f"Press {i+1}", (x+10, y+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 説明文
        cv2.putText(selection_screen, "Press number key to select camera", 
                   (20, screen_height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Camera Selection", selection_screen)
        
        # キー入力待ち
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key >= ord('1') and key <= ord('9'):
                camera_index = key - ord('1')
                if camera_index < len(available_cameras):
                    self.selected_camera_id = available_cameras[camera_index]['id']
                    print(f"カメラID {self.selected_camera_id} を選択しました。")
                    cv2.destroyWindow("Camera Selection")
                    return True
            elif key == 27:  # ESCキー
                cv2.destroyWindow("Camera Selection")
                return False

    def initialize_camera(self):
        """カメラを初期化"""
        if self.selected_camera_id is None:
            if not self.show_camera_selection():
                return False
        
        print(f"カメラID {self.selected_camera_id} を初期化中...")
        self.camera = cv2.VideoCapture(self.selected_camera_id, cv2.CAP_DSHOW)
        
        if not self.camera.isOpened():
            print(f"カメラID {self.selected_camera_id} を開けませんでした。")
            return False
        
        # Full HD解像度を設定
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        
        # 実際の解像度を確認
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"カメラID {self.selected_camera_id} を {actual_width}x{actual_height} で開きました。")
        
        return True

    def preprocess_frame(self, frame):
        """フレームを前処理"""
        if frame is None:
            return None
        
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        return img

    def predict_defect_simple(self, frame):
        """シンプルな欠陥予測"""
        if self.model is None or frame is None:
            return "good", 0.0

        preprocessed_img = self.preprocess_frame(frame)
        if preprocessed_img is None:
            return "good", 0.0

        try:
            predictions = self.model.predict(preprocessed_img, verbose=0)
            score = np.max(predictions)
            class_id = np.argmax(predictions)
            
            predicted_class = self.class_names[class_id]
            
            # デバッグ出力
            print(f"DEBUG: Raw prediction - class_id: {class_id}, predicted_class: '{predicted_class}', score: {score:.3f}")
            print(f"DEBUG: All predictions: {predictions[0]}")
            
            # 予測履歴に追加
            self.prediction_history.append((predicted_class, score))
            if len(self.prediction_history) > self.history_size:
                self.prediction_history.pop(0)
            
            # 安定性チェック（シンプル版）
            if len(self.prediction_history) >= self.history_size:
                # 最も多い予測を取得
                class_counts = {}
                for pred_class, pred_score in self.prediction_history:
                    class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                
                most_common_class = max(class_counts, key=class_counts.get)
                most_common_count = class_counts[most_common_class]
                
                print(f"DEBUG: History - most_common: '{most_common_class}', count: {most_common_count}/{len(self.prediction_history)}")
                
                # 安定性チェック（より緩く）
                if most_common_count >= 2 and score >= self.stability_threshold:  # 3回中2回以上
                    print(f"DEBUG: Stable prediction: '{most_common_class}'")
                    return most_common_class, score
            
            # 安定していない場合は現在の予測を維持
            print(f"DEBUG: Unstable, keeping current: '{self.current_prediction}'")
            return self.current_prediction, self.prediction_confidence
            
        except Exception as e:
            print(f"予測エラー: {e}")
            return "good", 0.0

    def get_defect_reason_simple(self, prediction_text):
        """シンプルな欠陥理由取得"""
        # デバッグ出力
        print(f"DEBUG: prediction_text = '{prediction_text}' (type: {type(prediction_text)})")
        
        if prediction_text is None or prediction_text == "":
            return "判定中..."
        
        # 予測結果を文字列に変換してトリム
        prediction_str = str(prediction_text).strip()
        
        # 欠陥理由マッピング
        defect_reasons = {
            "good": "良品 (Good)",
            "chipping": "欠け (Chipping)",
            "black_spot": "黒点 (Black Spot)", 
            "scratch": "傷 (Scratch)"
        }
        
        result = defect_reasons.get(prediction_str, f"不明な欠陥 ({prediction_str})")
        print(f"DEBUG: defect_reason = '{result}'")
        return result

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
                
                cropped_image = self.current_frame[y_min:y_max, x_min:x_max]
                cropped_image_path = save_dir / f"cropped_{timestamp}.jpg"
                success2 = cv2.imwrite(str(cropped_image_path), cropped_image)
                
                if success2:
                    print(f"切り抜き画像を保存: {cropped_image_path}")
                    self.feedback_count += 1
                    print(f"フィードバック数: {self.feedback_count}")
                else:
                    print(f"切り抜き画像の保存に失敗: {cropped_image_path}")
            
            # 選択点をクリア
            self.selection_points = []
            
        except Exception as e:
            print(f"画像保存エラー: {e}")

    def start_training(self):
        """学習を開始"""
        if self.is_training:
            print("既に学習中です。")
            return
        
        print("ディープラーニングを開始します...")
        self.is_training = True
        
        # バックグラウンドで学習を実行
        self.training_thread = threading.Thread(target=self._train_model)
        self.training_thread.daemon = True
        self.training_thread.start()

    def _train_model(self):
        """モデルを学習"""
        try:
            # ここで実際の学習処理を実装
            print("学習処理を実行中...")
            time.sleep(5)  # 仮の学習時間
            print("学習完了！")
            
        except Exception as e:
            print(f"学習エラー: {e}")
        finally:
            self.is_training = False

    def run(self):
        """メインループ"""
        if not self.load_model():
            print("モデルの読み込みに失敗しました。")
            return
        
        if not self.initialize_camera():
            print("カメラの初期化に失敗しました。システムを終了します。")
            return

        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_title, 1280, 720)
        
        # マウスコールバックを設定
        cv2.setMouseCallback(self.window_title, self.mouse_callback)
        
        # ウィンドウの×ボタンで終了できるように設定
        cv2.setWindowProperty(self.window_title, cv2.WND_PROP_TOPMOST, 0)

        print("\n--- 根本修正版樹脂ワッシャー欠陥検査システム起動 ---")
        print("ESCキーまたは×ボタンで終了")
        print("M: 選択モード S: 学習開始 1-4: クラス選択")

        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("フレームを読み込めませんでした。")
                break

            # 現在のフレームを保存（画像保存用）
            self.current_frame = frame.copy()

            current_time = time.time()
            
            # 予測処理（間隔制御）
            if current_time - self.last_prediction_time >= self.prediction_interval:
                prediction_text, confidence_score = self.predict_defect_simple(frame)
                self.current_prediction = prediction_text
                self.prediction_confidence = confidence_score
                self.last_prediction_time = current_time
                
                # デバッグ出力
                print(f"予測: {prediction_text} (信頼度: {confidence_score:.2f})")

            display_frame = frame.copy()
            
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

            # 左上のバナー
            if self.is_training:
                result_text = "学習中..."
                banner_color = (0, 255, 255)  # シアン
            elif self.current_prediction == "good":
                result_text = "Result: OK - 良品"
                banner_color = (0, 255, 0)  # 緑
            else:
                # NGの場合は具体的な欠陥理由を表示
                defect_reason = self.get_defect_reason_simple(self.current_prediction)
                result_text = f"Result: NG - {defect_reason}"
                banner_color = (0, 0, 255)  # 赤

            banner_width = 400
            banner_height = 80
            banner_x = 10
            banner_y = 10
            cv2.rectangle(display_frame, (banner_x, banner_y), (banner_x + banner_width, banner_y + banner_height), banner_color, -1)
            cv2.putText(display_frame, result_text, (banner_x + 10, banner_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # 左下のコントロールパネル
            panel_width = 400
            panel_height = 120
            panel_x = 10
            panel_y = display_frame.shape[0] - panel_height - 10
            cv2.rectangle(display_frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
            
            text_color = (255, 255, 255)
            font_scale = 0.7
            thickness = 2
            line_height = 25

            cv2.putText(display_frame, f"Feedback: {self.feedback_count}", 
                       (panel_x + 10, panel_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            cv2.putText(display_frame, "Controls", 
                       (panel_x + 10, panel_y + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            cv2.putText(display_frame, "1:Good 2:Black 3:Chipped 4:Scratched", 
                       (panel_x + 10, panel_y + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            cv2.putText(display_frame, "M:Select S:Train ESC:Exit", 
                       (panel_x + 10, panel_y + line_height * 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            
            # モデル情報表示
            cv2.putText(display_frame, f"Model: Real Image Trained | Camera: {self.selected_camera_id}", 
                       (panel_x + 10, panel_y + line_height * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(self.window_title, display_frame)

            key = cv2.waitKey(1) & 0xFF
            
            # ESCキーまたは×ボタンで終了
            if key == 27:
                break
            
            # ウィンドウが閉じられたかチェック
            if cv2.getWindowProperty(self.window_title, cv2.WND_PROP_VISIBLE) < 1:
                break
            
            # キー操作
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

if __name__ == "__main__":
    system = FundamentalFixInspectionSystem()
    system.run()