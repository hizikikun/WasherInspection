#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High Accuracy Real-time Inspection System
- Uses the new high accuracy model
- Improved prediction logic
- Better defect detection
"""

import cv2
import numpy as np
import tensorflow as tf
import os
import json
import time
from collections import deque

class HighAccuracyInspectionSystem:
    def __init__(self, camera_id=2):
        self.camera_id = camera_id
        self.cap = None
        self.model = None
        self.class_names = ['good', 'chipping', 'black_spot', 'scratch']
        self.window_title = "High Accuracy Inspection System - Camera 2"
        
        # Prediction stabilization
        self.prediction_history = deque(maxlen=15)
        self.stability_threshold = 0.6
        self.min_stable_count = 10
        self.current_prediction = "good"
        self.prediction_confidence = 0.0
        self.last_prediction_time = 0
        self.prediction_interval = 0.5
        
        # UI settings
        self.hud_height = 180
        self.hud_width = 700
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load the high accuracy model"""
        model_path = "high_accuracy_resin_washer_model.h5"
        class_names_path = "class_names.json"
        
        if os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                print(f"High accuracy model loaded successfully!")
                print(f"Model input shape: {self.model.input_shape}")
                print(f"Model output shape: {self.model.output_shape}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        else:
            print(f"Model file not found: {model_path}")
            self.model = None
        
        if os.path.exists(class_names_path):
            try:
                with open(class_names_path, 'r') as f:
                    self.class_names = json.load(f)
                print(f"Class names loaded: {self.class_names}")
            except Exception as e:
                print(f"Error loading class names: {e}")
    
    def initialize_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
        
        # Set camera properties for high quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        
        print(f"Camera {self.camera_id} initialized successfully")
        return True
    
    def detect_washer(self, frame):
        """Detect if washer is present in frame"""
        if frame is None:
            return False, 0.0, []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 3000
        max_area = 150000
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # More lenient circularity check
                        valid_contours.append(contour)
        
        if valid_contours:
            # Get the largest valid contour
            largest_contour = max(valid_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            confidence = min(area / 50000, 1.0)  # Normalize confidence
            return True, confidence, [largest_contour]
        
        return False, 0.0, []
    
    def predict_defect(self, frame):
        """Predict defect using high accuracy model"""
        if self.model is None or frame is None:
            return "good", 0.0
        
        # Check if washer is present
        has_washer, washer_confidence, washers = self.detect_washer(frame)
        if not has_washer:
            return "no_washer", 0.0
        
        # Preprocess image for model
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32) / 255.0
        
        try:
            # Get prediction
            predictions = self.model.predict(img, verbose=0)
            max_score = np.max(predictions)
            predicted_class_id = np.argmax(predictions)
            predicted_class = self.class_names[predicted_class_id]
            
            # Debug output
            print(f"AI Predictions: {dict(zip(self.class_names, predictions[0]))}")
            print(f"AI Selected: {predicted_class} (score: {max_score:.3f})")
            
            return predicted_class, max_score
            
        except Exception as e:
            print(f"AI prediction error: {e}")
            return "good", 0.0
    
    def get_defect_reason(self, prediction):
        """Get defect reason in English"""
        reason_map = {
            "good": "Good",
            "chipping": "Chipping",
            "black_spot": "Black Spot",
            "scratch": "Scratch",
            "no_washer": "No Washer"
        }
        return reason_map.get(prediction, "Unknown")
    
    def show_notification(self, frame, message, duration=2.0):
        """Show notification message"""
        current_time = time.time()
        if hasattr(self, 'notification_start_time') and current_time - self.notification_start_time < duration:
            # Draw notification
            cv2.putText(frame, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            self.notification_start_time = current_time
    
    def run(self):
        """Run the inspection system"""
        if not self.initialize_camera():
            return
        
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_title, 1280, 720)
        
        print("High Accuracy Inspection System Started!")
        print("Press ESC to exit, M to select area, S to train, 1-4 for feedback")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Predict defect
            current_time = time.time()
            if current_time - self.last_prediction_time >= self.prediction_interval:
                prediction, confidence = self.predict_defect(frame)
                
                # Add to prediction history
                self.prediction_history.append((prediction, confidence))
                
                # Check for stability
                if len(self.prediction_history) >= self.min_stable_count:
                    # Get most common prediction
                    class_counts = {}
                    for pred_class, pred_score in self.prediction_history:
                        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                    
                    most_common_class = max(class_counts, key=class_counts.get)
                    most_common_count = class_counts[most_common_class]
                    
                    # Check if stable
                    if most_common_count >= self.min_stable_count and confidence >= self.stability_threshold:
                        self.current_prediction = most_common_class
                        self.prediction_confidence = confidence
                
                self.last_prediction_time = current_time
            
            # Draw UI
            h, w = frame.shape[:2]
            
            # Top-left banner
            if self.current_prediction == "no_washer":
                result_text = "Result: No Washer Detected"
                banner_color = (0, 165, 255)  # Orange
            elif self.current_prediction == "good":
                result_text = "Result: OK - Good"
                banner_color = (0, 255, 0)  # Green
            else:
                reason = self.get_defect_reason(self.current_prediction)
                result_text = f"Result: NG - {reason}"
                banner_color = (0, 0, 255)  # Red
            
            # Draw banner background
            cv2.rectangle(frame, (10, 10), (w-10, 80), banner_color, -1)
            cv2.putText(frame, result_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Bottom-left control panel
            hud_x = 10
            hud_y = h - self.hud_height - 10
            
            # Draw HUD background
            cv2.rectangle(frame, (hud_x, hud_y), (hud_x + self.hud_width, hud_y + self.hud_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (hud_x, hud_y), (hud_x + self.hud_width, hud_y + self.hud_height), (255, 255, 255), 2)
            
            # HUD content
            y_offset = hud_y + 30
            cv2.putText(frame, "Feedback: 0", (hud_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
            cv2.putText(frame, "Controls", (hud_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 25
            cv2.putText(frame, "1:Good 2:Black 3:Chipped 4:Scratched", (hud_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            cv2.putText(frame, "M:Select S:Train ESC:Exit", (hud_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            cv2.putText(frame, "Model: High Accuracy | Camera: 2", (hud_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show notification
            if hasattr(self, 'notification_message'):
                self.show_notification(frame, self.notification_message)
            
            # Display frame
            cv2.imshow(self.window_title, frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('m'):
                self.notification_message = "Selection mode activated"
                print("Selection mode activated")
            elif key == ord('s'):
                self.notification_message = "Training started"
                print("Training started")
            elif key == ord('1'):
                self.notification_message = "Feedback: Good"
                print("Feedback: Good")
            elif key == ord('2'):
                self.notification_message = "Feedback: Black Spot"
                print("Feedback: Black Spot")
            elif key == ord('3'):
                self.notification_message = "Feedback: Chipped"
                print("Feedback: Chipped")
            elif key == ord('4'):
                self.notification_message = "Feedback: Scratched"
                print("Feedback: Scratched")
        
        # Cleanup
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("High Accuracy Inspection System stopped")

def main():
    """Main function"""
    system = HighAccuracyInspectionSystem()
    system.run()

if __name__ == "__main__":
    main()