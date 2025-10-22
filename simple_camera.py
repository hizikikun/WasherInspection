#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Camera Inspection System
No Japanese characters to avoid encoding issues
"""

import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import argparse
from datetime import datetime

def load_model(model_path="resin_washer_model/resin_washer_model.h5"):
    """Load the trained model"""
    print("=" * 60)
    print("Loading Model...")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return None
        
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return None

def preprocess_image(image):
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

def predict_washer(model, image):
    """Predict washer quality"""
    if model is None:
        return "Model not loaded", 0.0, "Unknown"
        
    # Preprocess
    processed = preprocess_image(image)
    if processed is None:
        return "Preprocessing failed", 0.0, "Unknown"
    
    try:
        # Predict
        predictions = model.predict(processed, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        class_names = ['Good', 'Defect_Chip', 'Defect_Scratch', 'Defect_BlackSpot']
        class_name = class_names[predicted_class]
        
        return class_name, confidence, class_name
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Prediction failed", 0.0, "Unknown"

def draw_result(frame, result, confidence, class_name):
    """Draw inspection result on frame"""
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
    
    return result_image

def test_focus(frame):
    """Test focus quality"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var > 100:
        return "GOOD FOCUS", (0, 255, 0)
    elif laplacian_var > 50:
        return "MEDIUM FOCUS", (0, 255, 255)
    else:
        return "POOR FOCUS", (0, 0, 255)

def run_camera_inspection(camera_id=0):
    """Run camera inspection"""
    print("=" * 60)
    print("Simple Camera Inspection System")
    print("=" * 60)
    
    # Load model
    model = load_model()
    if model is None:
        print("ERROR: Cannot load model")
        return
    
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
    print("  ESC: Exit")
    print("=" * 60)
    
    # Statistics
    total_inspections = 0
    good_count = 0
    defect_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read from camera")
            break
        
        # Display frame
        display_frame = frame.copy()
        
        # Statistics
        stats_text = f"Inspections: {total_inspections} | Good: {good_count} | Defect: {defect_count}"
        cv2.putText(display_frame, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Instructions
        instructions = [
            "SPACE: Inspect",
            "F: Test Focus", 
            "R: Reset Focus",
            "S: Save Image",
            "ESC: Exit"
        ]
        
        y_start = display_frame.shape[0] - 90
        for i, instruction in enumerate(instructions):
            cv2.putText(display_frame, instruction, 
                       (10, y_start + i * 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Show frame
        cv2.imshow("Washer Inspection", display_frame)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            print("Inspecting...")
            result, confidence, class_name = predict_washer(model, frame)
            
            # Show result
            result_frame = draw_result(frame, result, confidence, class_name)
            cv2.imshow("Inspection Result", result_frame)
            
            # Update statistics
            total_inspections += 1
            if result == "Good":
                good_count += 1
            else:
                defect_count += 1
            
            print(f"Result: {result} (Confidence: {confidence:.2f})")
            
            # Show result for 3 seconds
            cv2.waitKey(3000)
            cv2.destroyWindow("Inspection Result")
            
        elif key == ord('f'):  # F
            print("Testing focus...")
            focus_status, color = test_focus(frame)
            print(f"Focus: {focus_status}")
            
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
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("=" * 60)
    print("Final Statistics")
    print("=" * 60)
    print(f"Total inspections: {total_inspections}")
    print(f"Good products: {good_count}")
    print(f"Defective products: {defect_count}")
    print("=" * 60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simple Camera Inspection')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    
    args = parser.parse_args()
    
    print(f"Starting inspection with camera {args.camera}")
    run_camera_inspection(args.camera)

if __name__ == "__main__":
    main()




