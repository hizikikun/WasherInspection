#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Force Camera Window - Guaranteed to show camera window
"""
import cv2
import sys
import os
import time

def main():
    print("=== Force Camera Window ===")
    print("This will force the camera window to appear")
    
    # Test multiple cameras
    for camera_id in [0, 1, 2]:
        print(f"Trying camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            print(f"Camera {camera_id} is available!")
            
            # Set window properties to force visibility
            cv2.namedWindow('FORCE CAMERA WINDOW', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('FORCE CAMERA WINDOW', 800, 600)
            cv2.moveWindow('FORCE CAMERA WINDOW', 100, 100)
            
            print("Camera window should be visible now!")
            print("Press ESC to exit")
            
            try:
                for i in range(100):  # Try for 10 seconds
                    ret, frame = cap.read()
                    if ret:
                        cv2.imshow('FORCE CAMERA WINDOW', frame)
                        print(f"Frame {i}: Camera working!")
                    
                    key = cv2.waitKey(100) & 0xFF
                    if key == 27:  # ESC
                        break
                        
            except Exception as e:
                print(f"Error: {e}")
            finally:
                cap.release()
                cv2.destroyAllWindows()
                break
        else:
            print(f"Camera {camera_id} not available")
            cap.release()
    
    print("Test completed")

if __name__ == "__main__":
    main()



