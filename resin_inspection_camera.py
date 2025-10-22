#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resin Washer Inspection Camera - 樹脂ワッシャー外観検査システム
"""
import cv2
import numpy as np
import tensorflow as tf
import os
import time
import json
from datetime import datetime
import threading
import subprocess
import math

class ResinWasherInspector:
    def __init__(self):
        self.model = None
        self.class_names = ['Good', 'Chipped', 'Scratched', 'Black Spot']
        self._loading = True
        self.model_path = None
        self.model_mtime = 0.0
        # マウス選択用の状態
        self.selection_mode = False
        self.selection_points = []
        self.current_frame = None
        # モデルは別スレッドで読み込み（UIフリーズ防止）
        t = threading.Thread(target=self._load_model_async, daemon=True)
        t.start()
        # フィードバック保存先
        self.feedback_root = os.path.join(os.getcwd(), 'feedback_data')
        self._ensure_feedback_dirs()

    def _load_model_async(self):
        self.load_model()
        self._loading = False

    def _ensure_feedback_dirs(self):
        classes = ['Good', 'Black Spot', 'Chipped', 'Scratched']
        for cls in classes:
            os.makedirs(os.path.join(self.feedback_root, cls.replace(' ', '_')), exist_ok=True)
        # 学習データ側の追加学習フォルダも用意
        self.cs_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cs AI学習データ', '樹脂')
        self.class_to_cs = {
            'Good': os.path.join(self.cs_root, '良品(樹脂)', '追加学習'),
            'Black Spot': os.path.join(self.cs_root, '黒点樹脂', '追加学習'),
            'Chipped': os.path.join(self.cs_root, '欠け樹脂', '追加学習'),
            'Scratched': os.path.join(self.cs_root, '傷樹脂', '追加学習'),
        }
        # 可能な限り日本語パスに保存するが、非対応環境ではASCIIミラーに退避
        try:
            for p in self.class_to_cs.values():
                os.makedirs(p, exist_ok=True)
            self.cs_fallback_root = None
        except Exception:
            self.cs_fallback_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cs_ai_dataset_resin')
            for name in ['Good','Black_Spot','Chipped','Scratched']:
                os.makedirs(os.path.join(self.cs_fallback_root, name, '追加学習'), exist_ok=True)
        
    def load_model(self):
        """モデルを読み込む（スクリプトのディレクトリ基準の絶対パス）"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "resin_washer_model", "resin_washer_model.h5")
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print("Model loaded successfully")
                self.model_path = model_path
                try:
                    self.model_mtime = os.path.getmtime(model_path)
                except Exception:
                    self.model_mtime = time.time()
            else:
                print(f"Model file not found: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def preprocess_image(self, image):
        """画像を前処理"""
        # リサイズ
        image = cv2.resize(image, (224, 224))
        # 正規化
        image = image.astype(np.float32) / 255.0
        # バッチ次元を追加
        image = np.expand_dims(image, axis=0)
        return image
    
    def predict_defect(self, image):
        """欠陥を予測"""
        if self.model is None:
            return "Model not loaded", 0.0
        
        try:
            processed_image = self.preprocess_image(image)
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            class_name = self.class_names[predicted_class]
            return class_name, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error", 0.0

    def mouse_callback(self, event, x, y, flags, param):
        """マウスクリックで欠陥箇所を選択（4点選択）"""
        if event == cv2.EVENT_LBUTTONDOWN and self.selection_mode:
            self.selection_points.append((x, y))
            print(f"Selected point {len(self.selection_points)}: ({x}, {y})")
            if len(self.selection_points) >= 4:
                self.selection_mode = False
                print("Selection complete (4 points). Press 'S' to save cropped defect area.")

    def crop_defect_area(self, frame, points):
        """選択された欠陥箇所をクロップ（4点から矩形領域を計算）"""
        if len(points) < 4:
            return None
        
        # 4点から矩形の境界を計算
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)
        
        # 境界を少し拡張
        h, w = frame.shape[:2]
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        # クロップ
        cropped = frame[y1:y2, x1:x2]
        return cropped, (x1, y1, x2, y2)
    
    def draw_result(self, frame, result, confidence):
        """結果を描画（信頼度や時刻は表示しない）"""
        # 背景色を設定
        if result.startswith("OK"):
            color = (0, 255, 0)  # 緑
            bg_color = (0, 200, 0)
        else:
            color = (0, 0, 255)  # 赤
            bg_color = (0, 0, 200)
        
        # 結果テキスト
        result_text = f"Result: {result}"
        
        # 背景矩形（不透明）
        (tw, th), _ = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        pad = 10
        # 背景は残しつつもコンパクトに
        cv2.rectangle(frame, (10, 10), (10 + tw + pad * 2, 10 + th + pad * 2), bg_color, -1)
        
        # テキスト描画（不透明背景上に）
        cv2.putText(frame, result_text, (10 + pad, 10 + pad + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame

def draw_hud(frame, status_line: str = "", selection_mode: bool = False):
    """操作ガイドをフレーム内左下に小さめに常時表示する。status_lineがあれば先頭に追加表示"""
    lines = []
    if status_line:
        lines.append(status_line)
    if selection_mode:
        lines.append("SELECTION MODE: Click 4 points to select defect area")
    lines += [
        "Controls",
        "1:Good  2:Black  3:Chipped  4:Scratched",
        "M:Select  S:Save   ESC:Exit",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1

    # テキストサイズ計測と背景ボックス計算
    margin = 6
    line_height = int(18 * scale) + 6

    h, w = frame.shape[:2]
    x0 = 10
    y0 = h - (line_height * len(lines) + margin) - 10

    # テキスト描画（左下から上方向に）
    y = y0 + margin + int(14 * scale)
    for text in lines:
        # 白文字＋黒影で視認性を確保（背景に依存しにくい）
        cv2.putText(frame, text, (x0 + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (x0, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height
    return frame

def draw_toast(frame, message, color=(255,255,255)):
    """画面左下に短時間表示する小型トースト"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(message, font, scale, thickness)
    pad = 8
    x0, y0 = 10, 10 + th + pad
    # 右下に寄せる
    h, w = frame.shape[:2]
    x = 10
    y = h - 10
    # 影
    cv2.putText(frame, message, (x+1, y-1), font, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(frame, message, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return frame

def main():
    print("=== RESIN WASHER INSPECTION SYSTEM ===")
    print("Tree resin washer defect detection using deep learning")
    
    # インスペクター初期化
    inspector = ResinWasherInspector()
    
    # カメラ初期化（DirectShow優先、失敗時に再試行）
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("WARN: Camera 0 failed with CAP_DSHOW, retrying default backend...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            return
    
    # カメラ設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # スクリプトの場所へ移動（相対パス問題を防止）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        os.chdir(script_dir)
    except Exception:
        pass

    # ウィンドウ作成（タイトルは簡潔に）
    window_name = "Resin Washer Inspection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    cv2.moveWindow(window_name, 100, 100)
    
    # マウスコールバックを設定
    cv2.setMouseCallback(window_name, inspector.mouse_callback)
    
    print("Camera ready for resin washer inspection!")
    print("Controls: 1:Good  2:BlackSpot  3:Chipped  4:Scratched  M:Select  S:Save  ESC:Exit")
    
    frame_count = 0
    last_infer_time = 0.0
    last_result = ("Model not loaded", 0.0)
    confidence_threshold = 0.55

    # フィードバック/学習可視化用の状態
    toast_message = ""
    toast_until = 0.0
    training_until = 0.0  # この時刻まで"Training..."を表示
    feedback_count = 0
    log_path = os.path.join(os.getcwd(), 'feedback_data', 'feedback_log.jsonl')
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                feedback_count = sum(1 for _ in f)
        except Exception:
            feedback_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # 読み取り失敗時もイベント処理を継続
                frame = np.full((480, 640, 3), 128, dtype=np.uint8)
                cv2.putText(frame, 'Camera not ready...', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            # 一定間隔で自動判定（0.5秒間隔）
            now = time.time()
            if now - last_infer_time >= 0.5:
                if inspector._loading:
                    result, confidence = ('Loading model...', 0.0)
                else:
                    result, confidence = inspector.predict_defect(frame)
                last_result = (result, confidence)
                last_infer_time = now

                # 学習によりモデルが更新されていたら自動リロード
                try:
                    if inspector.model_path and os.path.exists(inspector.model_path):
                        mtime = os.path.getmtime(inspector.model_path)
                        if mtime > inspector.model_mtime + 1:  # 1秒以上更新
                            print("Detected new model file. Reloading...")
                            inspector.load_model()
                            # リロード後は即座に再推論
                            result, confidence = inspector.predict_defect(frame)
                            last_result = (result, confidence)
                            print(f"Model reloaded. New prediction: {result} ({confidence:.2f})")
                except Exception as e:
                    print(f"Model reload check failed: {e}")

            # OK/NG 判定とNG理由の表示
            result, confidence = last_result
            if inspector._loading:
                result_text = "Loading model..."
            elif inspector.model is None:
                result_text = "NG — Model not loaded"
            elif result == 'Good' and confidence >= confidence_threshold:
                result_text = "OK"
            else:
                # NG 理由（最も近い欠陥名を表示）
                reason = result if result in ['Black Spot', 'Chipped', 'Scratched'] else 'Defect'
                # OpenCVのフォントで文字化けを避けるためASCIIハイフンを使用
                result_text = f"NG - {reason}"

            # 結果表示
            frame = inspector.draw_result(frame, result_text, confidence)

            # 学習ステータス行の組み立て
            status = f"Feedback: {feedback_count}"
            now_time = time.time()
            if now_time < training_until:
                dots = "." * int((now_time * 2) % 4)
                status += f"   Training{dots}"

            # 選択モードの視覚化
            if inspector.selection_mode and len(inspector.selection_points) > 0:
                for i, (x, y) in enumerate(inspector.selection_points):
                    cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
                    cv2.putText(frame, f"{i+1}", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if len(inspector.selection_points) < 4:
                    cv2.putText(frame, f"Click point {len(inspector.selection_points)+1}/4", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 操作HUD
            frame = draw_hud(frame, status, inspector.selection_mode)

            # トースト表示（有効期間中）
            if now_time < toast_until and toast_message:
                frame = draw_toast(frame, toast_message, (0,255,0))
            
            # フレーム表示
            cv2.imshow(window_name, frame)
            
            frame_count += 1
            
            # キー入力チェック
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27:  # ESC
                print("Exiting...")
                break
            # 選択モード切り替え
            elif key == ord('m'):
                inspector.selection_mode = not inspector.selection_mode
                inspector.selection_points = []
                if inspector.selection_mode:
                    print("Selection mode ON: Click 4 points to select defect area")
                else:
                    print("Selection mode OFF")
            
            # 手動分類と保存
            elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('s')):
                cls = None
                if key == ord('1'):
                    cls = 'Good'
                elif key == ord('2'):
                    cls = 'Black Spot'
                elif key == ord('3'):
                    cls = 'Chipped'
                elif key == ord('4'):
                    cls = 'Scratched'

                # 保存のみ(S)
                if key == ord('s'):
                    cls = result if result in ['Good', 'Black Spot', 'Chipped', 'Scratched'] else 'Good'

                # 選択された欠陥箇所がある場合はクロップして保存
                save_frame = frame
                if len(inspector.selection_points) >= 4:
                    cropped_result = inspector.crop_defect_area(frame, inspector.selection_points)
                    if cropped_result:
                        cropped_frame, (x1, y1, x2, y2) = cropped_result
                        save_frame = cropped_frame
                        print(f"Cropped defect area: {x1},{y1} to {x2},{y2}")
                        # 選択点をリセット
                        inspector.selection_points = []
                        inspector.selection_mode = False

                # 画像保存
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                subdir = cls.replace(' ', '_')
                save_dir = os.path.join(inspector.feedback_root, subdir)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{cls.replace(' ', '')}_{ts}.jpg")
                cv2.imwrite(save_path, save_frame)

                # CS学習データ側にも保存
                try:
                    cs_dir = inspector.class_to_cs.get(cls)
                    target_dir = cs_dir
                    if inspector.cs_fallback_root:
                        # ASCIIミラーパスを使用
                        mapname = {'Good':'Good','Black Spot':'Black_Spot','Chipped':'Chipped','Scratched':'Scratched'}
                        target_dir = os.path.join(inspector.cs_fallback_root, mapname.get(cls, 'Good'), '追加学習')
                    if target_dir:
                        os.makedirs(target_dir, exist_ok=True)
                        cs_path = os.path.join(target_dir, f"{cls.replace(' ', '')}_{ts}.jpg")
                        # Unicodeパスでの失敗に備えimencode+tofileを使用
                        ok, buf = cv2.imencode('.jpg', save_frame)
                        if ok:
                            try:
                                buf.tofile(cs_path)
                                print(f"Saved also to CS dataset: {cs_path}")
                            except Exception as e:
                                print(f"WARN: Save to CS dataset failed: {cs_path} ({e})")
                        else:
                            print("WARN: imencode failed for CS dataset save")
                except Exception as e:
                    print(f"Save to CS dataset skipped: {e}")

                # ログ保存
                log = {
                    'timestamp': ts,
                    'label': cls,
                    'predicted': result,
                    'confidence': float(confidence)
                }
                with open(os.path.join(inspector.feedback_root, 'feedback_log.jsonl'), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log, ensure_ascii=False) + "\n")
                print(f"Saved feedback: {cls} -> {save_path}")
                # 可視化: カウンタ更新とトースト
                feedback_count += 1
                toast_message = f"Saved: {cls}"
                toast_until = time.time() + 1.5

                # バックグラウンド再学習を起動（存在する方を使用）
                try:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    retrain_script = os.path.join(script_dir, 'retrain_from_feedback.py')
                    if os.path.exists(retrain_script):
                        proc = subprocess.Popen([os.path.join(script_dir, '.venv', 'Scripts', 'python.exe'), retrain_script],
                                                cwd=script_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        if proc and proc.pid:
                            print(f"Retrain started (PID {proc.pid})")
                        training_until = time.time() + 120
                    else:
                        trainer_script = os.path.join(script_dir, 'resin_washer_trainer.py')
                        if os.path.exists(trainer_script):
                            proc = subprocess.Popen([os.path.join(script_dir, '.venv', 'Scripts', 'python.exe'), trainer_script, '--quick'],
                                                    cwd=script_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            if proc and proc.pid:
                                print(f"Quick training started (PID {proc.pid})")
                            training_until = time.time() + 120
                except Exception as e:
                    print(f"Background retrain not started: {e}")
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")

if __name__ == "__main__":
    main()
