#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Simple Camera Test - Minimal dependencies, maximum compatibility
"""
import cv2
import sys
import os

def main():
    print("=== Ultra Simple Camera Test ===")
    print("Current directory:", os.getcwd())
    
    # Test camera access
    print("Testing camera 0...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera 0")
        print("Trying camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("ERROR: Cannot open camera 1")
            return
    
    print("Camera opened successfully!")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press ESC to exit")
    print("Camera window should appear now...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Display frame
            cv2.imshow('Ultra Simple Camera Test', frame)
            
            # Check for ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
                
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



