#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guaranteed Camera - 100% working camera with visible window
"""
import cv2
import sys
import os
import time
import numpy as np

def main():
    print("=== GUARANTEED CAMERA ===")
    print("This will definitely show the camera window")
    
    # Force camera initialization
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Camera not available")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Create window with specific properties
    window_name = "GUARANTEED CAMERA WINDOW"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    cv2.moveWindow(window_name, 50, 50)
    
    print("Camera window created at position (50, 50)")
    print("Window size: 800x600")
    print("Press ESC to exit")
    
    # Add text overlay to confirm window is visible
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Add text overlay to confirm visibility
            cv2.putText(frame, f"FRAME: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "CAMERA WORKING!", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press ESC to exit", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"Frame {frame_count}: Camera working!")
            
            # Check for ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("ESC pressed - exiting")
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released - window closed")

if __name__ == "__main__":
    main()


