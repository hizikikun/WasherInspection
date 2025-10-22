#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistent Camera - ウィンドウが消えないカメラ
"""
import cv2
import sys
import os
import time
import numpy as np

def main():
    print("=== PERSISTENT CAMERA ===")
    print("This camera window will NOT disappear")
    
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
    window_name = "PERSISTENT CAMERA - Press ESC to exit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    cv2.moveWindow(window_name, 100, 100)
    
    print("Camera window created - it will stay open!")
    print("Press ESC to exit")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                time.sleep(0.1)  # Wait a bit before retrying
                continue
            
            # Add persistent text overlay
            cv2.putText(frame, f"FRAME: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "CAMERA WORKING!", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press ESC to exit", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"Frame {frame_count}: Camera working!")
            
            # Check for ESC key with longer wait time
            key = cv2.waitKey(30) & 0xFF  # Wait 30ms instead of 1ms
            if key == 27:  # ESC
                print("ESC pressed - exiting")
                break
            elif key == ord('q'):  # Also exit on 'q'
                print("Q pressed - exiting")
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Releasing camera...")
        cap.release()
        print("Destroying windows...")
        cv2.destroyAllWindows()
        print("Camera released - window closed")

if __name__ == "__main__":
    main()



