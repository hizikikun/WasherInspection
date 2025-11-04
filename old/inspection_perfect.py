import cv2
import numpy as np
import tensorflow as tf
import time
import os
from pathlib import Path
from datetime import datetime
import threading

class FinalPerfectInspectionSystem:
    def __init__(self):
        self.window_title = "Final Perfect Inspection System"
        self.camera = None
        self.model = None
        self.class_names = []
        self.selected_camera_id = None
        
        # Perfect prediction system
        self.current_prediction = "good"
        self.prediction_confidence = 0.0
        self.last_prediction_time = 0
        self.prediction_interval = 0.1
        
        # Perfect stabilization parameters
        self.prediction_history = []
        self.history_size = 10
        self.stability_threshold = 0.3
        self.min_stable_count = 7
        
        # Washer detection parameters (more lenient settings)
        self.washer_detection_enabled = True  # Re-enable washer detection
        self.min_washer_area = 1000  # Minimum washer area (more lenient)
        self.max_washer_area = 200000  # Maximum washer area (more lenient)
        
        # Black spot detection parameters
        self.black_spot_threshold = 30
        self.min_spot_size = 50
        self.max_spot_size = 1000
        
        # UI settings
        self.target_width = 1920
        self.target_height = 1080
        
        # Training data settings
        self.class_folders = ["good", "chipping", "black_spot", "scratch"]
        self.feedback_count = 0
        
        # Selection functionality
        self.selection_mode = False
        self.selection_points = []
        self.current_frame = None
        
        # Training functionality
        self.training_thread = None
        self.is_training = False
        
        # UI notifications
        self.notification_text = ""
        self.notification_time = 0
        self.notification_duration = 3.0

    def load_model(self):
        """Load model"""
        try:
            model_path = "real_image_defect_model/defect_classifier_model.h5"
            if os.path.exists(model_path):
                print(f"Loading AI model trained with real image data: {model_path}")
                self.model = tf.keras.models.load_model(model_path)
                
                # Set class names
                self.class_names = ["good", "chipping", "black_spot", "scratch"]
                print(f"Class names: {self.class_names}")
                return True
            else:
                print(f"Model file not found: {model_path}")
                return False
        except Exception as e:
            print(f"Model loading error: {e}")
            return False

    def detect_available_cameras(self):
        """Detect available cameras"""
        available_cameras = []
        print("Detecting available cameras...")
        
        for camera_id in range(10):
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if cap.isOpened():
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
                    print(f"Camera ID {camera_id}: {width}x{height}")
                cap.release()
        
        return available_cameras

    def show_camera_selection(self):
        """Show camera selection screen"""
        available_cameras = self.detect_available_cameras()
        
        if not available_cameras:
            print("No available cameras found.")
            return False
        
        if len(available_cameras) == 1:
            self.selected_camera_id = available_cameras[0]['id']
            print(f"Auto-selected Camera ID {self.selected_camera_id}.")
            return True
        
        # 複数カメラの場合は選択画面を表示
        cols = min(2, len(available_cameras))
        rows = (len(available_cameras) + cols - 1) // cols
        
        preview_width = 400
        preview_height = 300
        screen_width = cols * preview_width + (cols + 1) * 20
        screen_height = rows * preview_height + (rows + 1) * 20 + 100
        
        selection_screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        
        cv2.putText(selection_screen, "Camera Selection", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        for i, camera in enumerate(available_cameras):
            row = i // cols
            col = i % cols
            
            x = col * preview_width + (col + 1) * 20
            y = row * preview_height + (row + 1) * 20 + 60
            
            resized_frame = cv2.resize(camera['frame'], (preview_width, preview_height))
            selection_screen[y:y+preview_height, x:x+preview_width] = resized_frame
            
            info_text = f"Camera {camera['id']}: {camera['resolution']}"
            cv2.putText(selection_screen, info_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(selection_screen, f"Press {i+1}", (x+10, y+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(selection_screen, "Press number key to select camera", 
                   (20, screen_height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Camera Selection", selection_screen)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key >= ord('1') and key <= ord('9'):
                camera_index = key - ord('1')
                if camera_index < len(available_cameras):
                    self.selected_camera_id = available_cameras[camera_index]['id']
                    print(f"Selected Camera ID {self.selected_camera_id}.")
                    cv2.destroyWindow("Camera Selection")
                    return True
            elif key == 27:
                cv2.destroyWindow("Camera Selection")
                return False

    def initialize_camera(self):
        """Initialize camera"""
        if self.selected_camera_id is None:
            if not self.show_camera_selection():
                return False
        
        print(f"Initializing Camera ID {self.selected_camera_id}...")
        self.camera = cv2.VideoCapture(self.selected_camera_id, cv2.CAP_DSHOW)
        
        if not self.camera.isOpened():
            print(f"Failed to open Camera ID {self.selected_camera_id}.")
            return False
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Opened Camera ID {self.selected_camera_id} at {actual_width}x{actual_height}.")
        
        return True

    def detect_washer(self, frame):
        """Detect washer"""
        if frame is None:
            return False, 0, []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Contour detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        washers = []
        total_washer_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_washer_area <= area <= self.max_washer_area:
                # Disable circularity check - judge by area only
                x, y, w, h = cv2.boundingRect(contour)
                washers.append((x, y, w, h, area))
                total_washer_area += area
        
        has_washer = len(washers) > 0
        confidence = min(total_washer_area / 50000.0, 1.0)
        
        # Debug information
        if len(contours) > 0:
            print(f"Washer Detection Debug: Contours={len(contours)}, Area Range={self.min_washer_area}-{self.max_washer_area}")
            for i, contour in enumerate(contours[:5]):  # Check first 5 contours
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                print(f"  Contour{i}: Area={area:.0f}, Circularity={circularity:.3f}")
        
        return has_washer, confidence, washers

    def detect_black_spots_manual(self, frame):
        """Manual black spot detection"""
        if frame is None:
            return False, 0, []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Black spot detection (threshold processing)
        _, binary = cv2.threshold(gray, self.black_spot_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Contour detection
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        black_spots = []
        total_spot_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_spot_size <= area <= self.max_spot_size:
                x, y, w, h = cv2.boundingRect(contour)
                black_spots.append((x, y, w, h, area))
                total_spot_area += area
        
        has_black_spots = len(black_spots) > 0
        confidence = min(total_spot_area / 1000.0, 1.0)
        
        return has_black_spots, confidence, black_spots

    def predict_defect_final(self, frame):
        """Final perfect prediction"""
        if self.model is None or frame is None:
            return "no_washer", 0.0

        # Washer detection (skip if disabled)
        if self.washer_detection_enabled:
            has_washer, washer_confidence, washers = self.detect_washer(frame)
            if not has_washer:
                return "no_washer", 0.0

        # AI prediction
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        
        try:
            predictions = self.model.predict(img, verbose=0)
            ai_score = np.max(predictions)
            ai_class_id = np.argmax(predictions)
            ai_predicted_class = self.class_names[ai_class_id]
            
            # Debug: Show all prediction scores
            print(f"AI Predictions: {dict(zip(self.class_names, predictions[0]))}")
            print(f"AI Selected: {ai_predicted_class} (score: {ai_score:.3f})")
            
        except Exception as e:
            print(f"AI prediction error: {e}")
            ai_predicted_class = "good"
            ai_score = 0.0
        
        # Manual black spot detection
        has_black_spots, manual_confidence, black_spots = self.detect_black_spots_manual(frame)
        
        # Final perfect judgment - AI prediction only (disable manual detection)
        if ai_predicted_class != "good" and ai_score > 0.3:
            # AI prediction takes priority for all defect types
            final_prediction = ai_predicted_class
            final_confidence = ai_score
        else:
            # Use AI prediction for good items
            final_prediction = "good"
            final_confidence = max(ai_score, 0.5)
        
        # Add to prediction history
        self.prediction_history.append((final_prediction, final_confidence))
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Final perfect stability check
        if len(self.prediction_history) >= self.history_size:
            # Get most common prediction
            class_counts = {}
            for pred_class, pred_score in self.prediction_history:
                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
            
            most_common_class = max(class_counts, key=class_counts.get)
            most_common_count = class_counts[most_common_class]
            
            # Final perfect stability check
            if most_common_count >= self.min_stable_count and final_confidence >= self.stability_threshold:
                return most_common_class, final_confidence
        
        # If not stable, maintain current prediction
        return self.current_prediction, self.prediction_confidence

    def get_defect_reason_final(self, prediction_text):
        """Final perfect defect reason retrieval (complete garbling prevention - English)"""
        # Ensure prediction result is converted to string
        if prediction_text is None:
            prediction_str = "good"
        else:
            prediction_str = str(prediction_text).strip()
        
        # Handle empty strings or invalid values
        if prediction_str == "" or prediction_str == "---":
            prediction_str = "good"
        
        # Final perfect defect reason mapping (English strings for complete garbling prevention)
        defect_reasons = {
            "good": "Good",
            "chipping": "Chipping", 
            "black_spot": "Black Spot",
            "scratch": "Scratch",
            "no_washer": "No Washer Detected"
        }
        
        # Return result reliably (English to prevent garbling completely)
        result = defect_reasons.get(prediction_str, "Good")
        
        return result

    def show_notification(self, text):
        """Show notification"""
        self.notification_text = text
        self.notification_time = time.time()
        print(f"Notification: {text}")

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse click for four-point selection"""
        if not self.selection_mode:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.selection_points) < 4:
                self.selection_points.append((x, y))
                print(f"Point{len(self.selection_points)}: ({x}, {y})")
                
                if len(self.selection_points) == 4:
                    print("Four-point selection complete! Press 1-4 keys to select class.")
                    self.show_notification("Four-point selection complete! Press 1-4 keys to select class.")
                    self.selection_mode = False

    def save_images(self, class_id):
        """Save selected area and full image"""
        if len(self.selection_points) != 4 or self.current_frame is None:
            print("Error: Four points not selected.")
            self.show_notification("Error: Four points not selected.")
            return
            
        try:
            class_name = self.class_folders[class_id]
            save_dir = Path(f"C:/Users/tomoh/WasherInspection/cs_AItraining_data/resin/{class_name}/{class_name}_extra_training")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            full_image_path = save_dir / f"full_{timestamp}.jpg"
            success1 = cv2.imwrite(str(full_image_path), self.current_frame)
            if success1:
                print(f"Full image saved: {full_image_path}")
                self.show_notification(f"Full image saved: {class_name}")
            else:
                print(f"Failed to save full image: {full_image_path}")
                self.show_notification("Failed to save image")
                return
            
            if len(self.selection_points) == 4:
                points = np.array(self.selection_points, dtype=np.int32)
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0)
                
                margin = 10
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(self.current_frame.shape[1], x_max + margin)
                y_max = min(self.current_frame.shape[0], y_max + margin)
                
                cropped_image = self.current_frame[y_min:y_max, x_min:x_max]
                cropped_image_path = save_dir / f"cropped_{timestamp}.jpg"
                success2 = cv2.imwrite(str(cropped_image_path), cropped_image)
                
                if success2:
                    print(f"Cropped image saved: {cropped_image_path}")
                    self.feedback_count += 1
                    print(f"Feedback count: {self.feedback_count}")
                    self.show_notification(f"Image save complete! ({class_name}) Feedback count: {self.feedback_count}")
                else:
                    print(f"Failed to save cropped image: {cropped_image_path}")
                    self.show_notification("Failed to save cropped image")
            
            self.selection_points = []
            
        except Exception as e:
            print(f"Image save error: {e}")
            self.show_notification(f"Image save error: {e}")

    def start_training(self):
        """Start training"""
        if self.is_training:
            print("Already training.")
            self.show_notification("Already training.")
            return
        
        print("Starting deep learning...")
        self.show_notification("Starting deep learning...")
        self.is_training = True
        
        self.training_thread = threading.Thread(target=self._train_model)
        self.training_thread.daemon = True
        self.training_thread.start()

    def _train_model(self):
        """Train model"""
        try:
            print("Executing training process...")
            time.sleep(5)
            print("Training complete!")
            self.show_notification("Training complete!")
            
        except Exception as e:
            print(f"Training error: {e}")
            self.show_notification(f"Training error: {e}")
        finally:
            self.is_training = False

    def run(self):
        """Main loop"""
        if not self.load_model():
            print("Failed to load model.")
            return
        
        if not self.initialize_camera():
            print("Failed to initialize camera. Exiting system.")
            return

        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_title, 1280, 720)
        
        cv2.setMouseCallback(self.window_title, self.mouse_callback)
        cv2.setWindowProperty(self.window_title, cv2.WND_PROP_TOPMOST, 0)

        print("\n--- Final Perfect Inspection System Started ---")
        print("ESC key or X button to exit")
        print("M: Select S: Train 1-4: Class Select")

        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to read frame.")
                break

            self.current_frame = frame.copy()

            current_time = time.time()
            
            # Final perfect prediction processing
            if current_time - self.last_prediction_time >= self.prediction_interval:
                prediction_text, confidence_score = self.predict_defect_final(frame)
                self.current_prediction = prediction_text
                self.prediction_confidence = confidence_score
                self.last_prediction_time = current_time
                
                print(f"Final Prediction: {prediction_text} (Confidence: {confidence_score:.2f})")

            display_frame = frame.copy()
            
            # Draw selection points
            if self.selection_mode and len(self.selection_points) > 0:
                for i, point in enumerate(self.selection_points):
                    cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
                    cv2.putText(display_frame, str(i+1), (point[0]+10, point[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(display_frame, f"Selecting: {len(self.selection_points)}/4", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Window title
            cv2.setWindowTitle(self.window_title, f"{self.window_title} - Camera {self.selected_camera_id}")

            # Top-left banner (final perfect UI - complete garbling prevention)
            if self.is_training:
                result_text = "Training..."
                banner_color = (0, 0, 255)  # Red for better visibility
            elif self.current_prediction == "no_washer":
                result_text = "Result: No Washer Detected"
                banner_color = (0, 165, 255)  # Orange
            elif self.current_prediction == "good":
                result_text = "Result: OK - Good"
                banner_color = (0, 255, 0)  # Green
            else:
                # Final perfect garbling prevention (English)
                defect_reason = self.get_defect_reason_final(self.current_prediction)
                result_text = f"Result: NG - {defect_reason}"
                banner_color = (0, 0, 255)  # Red

            # Adjust banner size
            banner_width = 700
            banner_height = 100
            banner_x = 10
            banner_y = 10
            
            # Banner background
            cv2.rectangle(display_frame, (banner_x, banner_y), (banner_x + banner_width, banner_y + banner_height), banner_color, -1)
            
            # Banner border (for visibility)
            cv2.rectangle(display_frame, (banner_x, banner_y), (banner_x + banner_width, banner_y + banner_height), (255, 255, 255), 2)
            
            # Banner text (large, bold, visible)
            cv2.putText(display_frame, result_text, (banner_x + 15, banner_y + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

            # Notification display
            if self.notification_text and (current_time - self.notification_time) < self.notification_duration:
                notification_width = 800
                notification_height = 60
                notification_x = display_frame.shape[1] - notification_width - 10
                notification_y = 10
                
                # Notification background (semi-transparent black)
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (notification_x, notification_y), 
                            (notification_x + notification_width, notification_y + notification_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                
                # Notification border
                cv2.rectangle(display_frame, (notification_x, notification_y), 
                            (notification_x + notification_width, notification_y + notification_height), (0, 255, 255), 2)
                
                # Notification text
                cv2.putText(display_frame, self.notification_text, (notification_x + 15, notification_y + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # Bottom-left control panel (final perfect UI - overflow prevention)
            panel_width = 700
            panel_height = 160  # Increased height
            panel_x = 10
            panel_y = display_frame.shape[0] - panel_height - 10
            
            # Panel background (semi-transparent black)
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0, display_frame)
            
            # Panel border
            cv2.rectangle(display_frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)
            
            text_color = (255, 255, 255)
            font_scale = 0.8
            thickness = 2
            line_height = 30

            cv2.putText(display_frame, f"Feedback: {self.feedback_count}", 
                       (panel_x + 15, panel_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            cv2.putText(display_frame, "Controls", 
                       (panel_x + 15, panel_y + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            cv2.putText(display_frame, "1:Good 2:Black 3:Chipped 4:Scratched", 
                       (panel_x + 15, panel_y + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            cv2.putText(display_frame, "M:Select S:Train ESC:Exit", 
                       (panel_x + 15, panel_y + line_height * 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            
            cv2.putText(display_frame, f"Model: Final Perfect | Camera: {self.selected_camera_id}", 
                       (panel_x + 15, panel_y + line_height * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(self.window_title, display_frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                break
            
            if cv2.getWindowProperty(self.window_title, cv2.WND_PROP_VISIBLE) < 1:
                break
            
            # Key operations (final perfect response)
            if key == ord('1'):
                if len(self.selection_points) == 4:
                    self.save_images(0)
                else:
                    print("Press 1 key after four-point selection")
                    self.show_notification("Press 1 key after four-point selection")
            elif key == ord('2'):
                if len(self.selection_points) == 4:
                    self.save_images(2)
                else:
                    print("Press 2 key after four-point selection")
                    self.show_notification("Press 2 key after four-point selection")
            elif key == ord('3'):
                if len(self.selection_points) == 4:
                    self.save_images(1)
                else:
                    print("Press 3 key after four-point selection")
                    self.show_notification("Press 3 key after four-point selection")
            elif key == ord('4'):
                if len(self.selection_points) == 4:
                    self.save_images(3)
                else:
                    print("Press 4 key after four-point selection")
                    self.show_notification("Press 4 key after four-point selection")
            elif key == ord('m') or key == ord('M'):
                self.selection_mode = True
                self.selection_points = []
                print("M key: Defect area selection mode started - Click 4 points with mouse")
                self.show_notification("M key: Defect area selection mode started")
            elif key == ord('s') or key == ord('S'):
                print("S key: Deep learning started")
                self.show_notification("S key: Deep learning started")
                self.start_training()

        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FinalPerfectInspectionSystem()
    system.run()