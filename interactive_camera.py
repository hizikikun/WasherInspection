#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Camera Inspection System with Manual Learning
"""

import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import argparse
import json
from datetime import datetime

class InteractiveInspector:
    def __init__(self, model_path="resin_washer_model/resin_washer_model.h5"):
        """Initialize interactive inspector"""
        print("=" * 60)
        print("Interactive Camera Inspection System")
        print("=" * 60)
        
        self.model_path = model_path
        self.model = None
        self.class_names = ['Good', 'Defect_Chip', 'Defect_Scratch', 'Defect_BlackSpot']
        self.feedback_data = []
        self.feedback_file = "feedback_data.json"
        
        # Load model
        self.load_model()
        
        # Load existing feedback data
        self.load_feedback_data()
    
    def load_model(self):
        """Load the trained model"""
        print("Loading Model...")
        
        if not os.path.exists(self.model_path):
            print(f"ERROR: Model file not found: {self.model_path}")
            return False
            
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            return False
    
    def load_feedback_data(self):
        """Load existing feedback data"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    self.feedback_data = json.load(f)
                print(f"Loaded {len(self.feedback_data)} feedback entries")
            except:
                self.feedback_data = []
        else:
            self.feedback_data = []
    
    def save_feedback_data(self):
        """Save feedback data to file"""
        try:
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
            print(f"Feedback data saved: {len(self.feedback_data)} entries")
        except Exception as e:
            print(f"Error saving feedback data: {e}")
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        if image is None:
            return None
            
        # Resize to model input size
        resized = cv2.resize(image, (224, 224))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        batch = np.expand_dims(normalized, axis=0)
        
        return batch
    
    def predict_washer(self, image):
        """Predict washer quality"""
        if self.model is None:
            return "Model not loaded", 0.0, "Unknown"
            
        # Preprocess
        processed = self.preprocess_image(image)
        if processed is None:
            return "Preprocessing failed", 0.0, "Unknown"
        
        try:
            # Predict
            predictions = self.model.predict(processed, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            class_name = self.class_names[predicted_class]
            
            return class_name, confidence, class_name
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Prediction failed", 0.0, "Unknown"
    
    def test_focus(self, frame):
        """Test focus quality"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var > 100:
            return "GOOD FOCUS", (0, 255, 0), laplacian_var
        elif laplacian_var > 50:
            return "MEDIUM FOCUS", (0, 255, 255), laplacian_var
        else:
            return "POOR FOCUS", (0, 0, 255), laplacian_var
    
    def draw_result(self, frame, result, confidence, class_name, show_buttons=True):
        """Draw inspection result with interactive buttons"""
        h, w = frame.shape[:2]
        result_image = frame.copy()
        
        # Result display
        if result == "Good":
            color = (0, 255, 0)  # Green
            status = "OK"
        else:
            color = (0, 0, 255)  # Red
            status = "NG"
        
        # Main result
        cv2.putText(result_image, f"Status: {status}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
        
        # Reason
        cv2.putText(result_image, f"Reason: {class_name}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Confidence
        cv2.putText(result_image, f"Confidence: {confidence:.2f}", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        if show_buttons:
            # Interactive buttons
            button_y = h - 200
            button_width = 120
            button_height = 40
            
            # Correct/Incorrect buttons
            cv2.rectangle(result_image, (50, button_y), (50 + button_width, button_y + button_height), (0, 255, 0), -1)
            cv2.putText(result_image, "CORRECT", (60, button_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
            cv2.rectangle(result_image, (200, button_y), (200 + button_width, button_y + button_height), (0, 0, 255), -1)
            cv2.putText(result_image, "WRONG", (220, button_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Class selection buttons
            class_buttons = [
                ("GOOD", (50, button_y + 60), (0, 255, 0)),
                ("CHIP", (200, button_y + 60), (0, 0, 255)),
                ("SCRATCH", (350, button_y + 60), (0, 0, 255)),
                ("BLACK SPOT", (500, button_y + 60), (0, 0, 255))
            ]
            
            for text, (x, y), color in class_buttons:
                cv2.rectangle(result_image, (x, y), (x + button_width, y + button_height), color, -1)
                cv2.putText(result_image, text, (x + 5, y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Instructions
            cv2.putText(result_image, "Press C for CORRECT, W for WRONG", (50, button_y + 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_image, "Press 1-4 for class selection", (50, button_y + 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        return result_image
    
    def add_feedback(self, image, predicted_class, correct_class, is_correct):
        """Add feedback data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save image
        image_filename = f"feedback_{timestamp}.jpg"
        cv2.imwrite(image_filename, image)
        
        # Add to feedback data
        feedback_entry = {
            "timestamp": timestamp,
            "image_file": image_filename,
            "predicted_class": predicted_class,
            "correct_class": correct_class,
            "is_correct": is_correct
        }
        
        self.feedback_data.append(feedback_entry)
        print(f"Feedback added: {correct_class} (was {predicted_class})")
    
    def run_inspection(self, camera_id=0):
        """Run interactive inspection"""
        # Initialize camera
        print(f"Initializing camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera {camera_id}")
            return
        
        # Camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Auto focus settings
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            cap.set(cv2.CAP_PROP_FOCUS, 50)
            print("Auto focus enabled")
        except:
            print("Auto focus not available")
        
        print("Camera ready!")
        print("=" * 60)
        print("Controls:")
        print("  SPACE: Inspect")
        print("  F: Test Focus")
        print("  R: Reset Focus")
        print("  S: Save Image")
        print("  C: Correct (after inspection)")
        print("  W: Wrong (after inspection)")
        print("  1-4: Select correct class")
        print("  ESC: Exit")
        print("=" * 60)
        
        # Statistics
        total_inspections = 0
        good_count = 0
        defect_count = 0
        feedback_count = 0
        
        # State variables
        last_inspection_result = None
        last_inspection_image = None
        waiting_for_feedback = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Cannot read from camera")
                break
            
            # Display frame
            display_frame = frame.copy()
            
            # Statistics
            stats_text = f"Inspections: {total_inspections} | Good: {good_count} | Defect: {defect_count} | Feedback: {feedback_count}"
            cv2.putText(display_frame, stats_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Instructions
            instructions = [
                "SPACE: Inspect",
                "F: Test Focus", 
                "R: Reset Focus",
                "S: Save Image",
                "C: Correct | W: Wrong",
                "1-4: Select Class",
                "ESC: Exit"
            ]
            
            y_start = display_frame.shape[0] - 140
            for i, instruction in enumerate(instructions):
                cv2.putText(display_frame, instruction, 
                           (10, y_start + i * 18), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Show frame
            cv2.imshow("Interactive Washer Inspection", display_frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE
                print("Inspecting...")
                result, confidence, class_name = self.predict_washer(frame)
                
                # Show result with buttons
                result_frame = self.draw_result(frame, result, confidence, class_name, show_buttons=True)
                cv2.imshow("Inspection Result", result_frame)
                
                # Update statistics
                total_inspections += 1
                if result == "Good":
                    good_count += 1
                else:
                    defect_count += 1
                
                print(f"Result: {result} (Confidence: {confidence:.2f})")
                print("Press C for CORRECT, W for WRONG, or 1-4 for class selection")
                
                # Store for feedback
                last_inspection_result = (result, confidence, class_name)
                last_inspection_image = frame.copy()
                waiting_for_feedback = True
                
            elif key == ord('f'):  # F
                print("Testing focus...")
                focus_status, color, quality = self.test_focus(frame)
                print(f"Focus: {focus_status} (Quality: {quality:.2f})")
                
                # Show focus status
                cv2.putText(display_frame, f"Focus: {focus_status}", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                
            elif key == ord('r'):  # R
                print("Resetting focus...")
                try:
                    cap.set(cv2.CAP_PROP_FOCUS, 0)
                    cap.set(cv2.CAP_PROP_FOCUS, 50)
                    print("Focus reset completed")
                except:
                    print("Focus reset failed")
                    
            elif key == ord('s'):  # S
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"inspection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Image saved: {filename}")
            
            # Feedback handling
            elif waiting_for_feedback:
                if key == ord('c'):  # C - Correct
                    print("Marked as CORRECT")
                    feedback_count += 1
                    waiting_for_feedback = False
                    cv2.destroyWindow("Inspection Result")
                    
                elif key == ord('w'):  # W - Wrong
                    print("Marked as WRONG - waiting for correct class selection")
                    print("Press 1: Good, 2: Chip, 3: Scratch, 4: Black Spot")
                    
                elif key == ord('1'):  # 1 - Good
                    if waiting_for_feedback:
                        self.add_feedback(last_inspection_image, last_inspection_result[2], "Good", False)
                        feedback_count += 1
                        waiting_for_feedback = False
                        cv2.destroyWindow("Inspection Result")
                        
                elif key == ord('2'):  # 2 - Chip
                    if waiting_for_feedback:
                        self.add_feedback(last_inspection_image, last_inspection_result[2], "Defect_Chip", False)
                        feedback_count += 1
                        waiting_for_feedback = False
                        cv2.destroyWindow("Inspection Result")
                        
                elif key == ord('3'):  # 3 - Scratch
                    if waiting_for_feedback:
                        self.add_feedback(last_inspection_image, last_inspection_result[2], "Defect_Scratch", False)
                        feedback_count += 1
                        waiting_for_feedback = False
                        cv2.destroyWindow("Inspection Result")
                        
                elif key == ord('4'):  # 4 - Black Spot
                    if waiting_for_feedback:
                        self.add_feedback(last_inspection_image, last_inspection_result[2], "Defect_BlackSpot", False)
                        feedback_count += 1
                        waiting_for_feedback = False
                        cv2.destroyWindow("Inspection Result")
        
        # Save feedback data
        self.save_feedback_data()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("=" * 60)
        print("Final Statistics")
        print("=" * 60)
        print(f"Total inspections: {total_inspections}")
        print(f"Good products: {good_count}")
        print(f"Defective products: {defect_count}")
        print(f"Feedback entries: {feedback_count}")
        print("=" * 60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Interactive Camera Inspection')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    
    args = parser.parse_args()
    
    inspector = InteractiveInspector()
    if inspector.model is None:
        print("ERROR: Cannot load model")
        return
    
    print(f"Starting interactive inspection with camera {args.camera}")
    inspector.run_inspection(args.camera)

if __name__ == "__main__":
    main()




