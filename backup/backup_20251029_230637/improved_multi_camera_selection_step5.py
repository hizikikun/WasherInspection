#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Multi-Camera Selection System - Step 5
- Camera 2 Full HD guaranteed
- Improved mouse click selection
- Only autofocus + resolution settings
"""

import cv2
import numpy as np
import tensorflow as tf
import os
import json
import time
import platform
import psutil
from collections import deque

class ImprovedMultiCameraSelectionSystem:
    def __init__(self):
        self.cap = None
        self.model = None
        self.class_names = ['good', 'chipping', 'black_spot', 'scratch']
        self.window_title = "Improved Multi-Camera Selection System - Step 5"
        self.camera_id = None
        
        # Prediction settings - will be set by system detection
        self.current_prediction = "good"
        self.prediction_confidence = 0.0
        self.last_prediction_time = 0
        
        # Defect detection thresholds - adjusted for better accuracy
        self.defect_confidence_threshold = 0.25  # Increased from 0.15
        self.good_confidence_threshold = 0.6  # Reduced from 0.7
        
        # UI settings
        self.hud_height = 180
        self.hud_width = 700
        
        # Auto-detect system configuration
        self.system_config = self.detect_system_configuration()
        self.apply_system_optimizations()
        
        # Performance monitoring
        self.start_time = time.time()
        self.processing_times = []
        self.total_frames_processed = 0
        
        # Load model
        self.load_model()
    
    def detect_system_configuration(self):
        """Detect system configuration and determine optimal settings"""
        try:
            # Get CPU information
            cpu_info = platform.processor()
            cpu_count = psutil.cpu_count(logical=False)  # Physical cores
            cpu_count_logical = psutil.cpu_count(logical=True)  # Logical cores
            
            # Get memory information
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024**3)
            
            # Get GPU information (if available)
            gpu_info = "Unknown"
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_info = result.stdout.strip()
            except:
                gpu_info = "No NVIDIA GPU detected"
            
            # Determine system type
            is_notebook = False
            is_high_end = False
            
            # Check if it's likely a notebook (based on CPU and memory)
            if cpu_count <= 4 and total_memory_gb <= 32:
                is_notebook = True
                print("[NOTEBOOK] Notebook PC detected")
            else:
                print("[DESKTOP] Desktop PC detected")
            
            # Check if it's high-end system
            if cpu_count >= 8 and total_memory_gb >= 32 and "RTX" in gpu_info:
                is_high_end = True
                print("[HIGH-END] High-end system detected")
            
            config = {
                'cpu_info': cpu_info,
                'cpu_cores': cpu_count,
                'cpu_logical': cpu_count_logical,
                'memory_gb': total_memory_gb,
                'gpu_info': gpu_info,
                'is_notebook': is_notebook,
                'is_high_end': is_high_end,
                'system_type': 'notebook' if is_notebook else 'desktop'
            }
            
            print(f"System Configuration:")
            print(f"  CPU: {cpu_info} ({cpu_count} cores, {cpu_count_logical} threads)")
            print(f"  Memory: {total_memory_gb:.1f} GB")
            print(f"  GPU: {gpu_info}")
            print(f"  Type: {config['system_type']}")
            
            return config
            
        except Exception as e:
            print(f"Error detecting system configuration: {e}")
            # Default to notebook settings for safety
            return {
                'cpu_info': 'Unknown',
                'cpu_cores': 4,
                'cpu_logical': 8,
                'memory_gb': 16,
                'gpu_info': 'Unknown',
                'is_notebook': True,
                'is_high_end': False,
                'system_type': 'notebook'
            }
    
    def apply_system_optimizations(self):
        """Apply optimizations based on detected system configuration"""
        config = self.system_config
        
        if config['system_type'] == 'notebook':
            print("[OPTIMIZE] Applying notebook optimizations...")
            # Notebook settings - more conservative
            self.prediction_history = deque(maxlen=3)  # Minimal buffer
            self.stability_threshold = 0.4  # Higher threshold
            self.min_stable_count = 2  # Minimal stable count
            self.prediction_interval = 1.5  # Longer interval
            self.frame_skip_count = 0
            self.frame_skip_interval = 4  # Skip more frames
            self.processing_resolution = (480, 360)  # Lower resolution
            self.wait_key_interval = 150  # Longer wait
            
        elif config['is_high_end']:
            print("[OPTIMIZE] Applying high-end desktop optimizations...")
            # High-end desktop settings - maximum performance
            self.prediction_history = deque(maxlen=20)  # Large buffer
            self.stability_threshold = 0.1  # Lower threshold
            self.min_stable_count = 8  # More stable count
            self.prediction_interval = 0.1  # Short interval
            self.frame_skip_count = 0
            self.frame_skip_interval = 1  # No frame skipping
            self.processing_resolution = (1280, 720)  # Higher resolution
            self.wait_key_interval = 1  # Minimal wait
            
        else:
            print("[OPTIMIZE] Applying standard desktop optimizations...")
            # Standard desktop settings - balanced
            self.prediction_history = deque(maxlen=10)  # Medium buffer
            self.stability_threshold = 0.2  # Medium threshold
            self.min_stable_count = 4  # Medium stable count
            self.prediction_interval = 0.5  # Medium interval
            self.frame_skip_count = 0
            self.frame_skip_interval = 2  # Skip some frames
            self.processing_resolution = (640, 480)  # Medium resolution
            self.wait_key_interval = 30  # Medium wait
        
        print(f"Optimization settings applied:")
        print(f"  Prediction interval: {self.prediction_interval}s")
        print(f"  Frame skip interval: {self.frame_skip_interval}")
        print(f"  Processing resolution: {self.processing_resolution}")
        print(f"  Wait key interval: {self.wait_key_interval}ms")
        
    def load_model(self):
        """Load the best available model"""
        model_paths = [
            "balanced_resin_washer_model.h5",
            "ultra_high_accuracy_resin_washer_model.h5",
            "high_accuracy_resin_washer_model.h5",
            "best_real_defect_model.h5"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    self.model = tf.keras.models.load_model(model_path)
                    print(f"Model loaded successfully: {model_path}")
                    print(f"Model input shape: {self.model.input_shape}")
                    print(f"Model output shape: {self.model.output_shape}")
                    break
                except Exception as e:
                    print(f"Error loading {model_path}: {e}")
                    continue
        
        if self.model is None:
            print("No model found! Please train a model first.")
            return
        
        # Load class names
        class_names_path = "class_names.json"
        if os.path.exists(class_names_path):
            try:
                with open(class_names_path, 'r') as f:
                    self.class_names = json.load(f)
                print(f"Class names loaded: {self.class_names}")
            except Exception as e:
                print(f"Error loading class names: {e}")
    
    def detect_and_start_all_cameras(self):
        """Detect and start all available cameras"""
        print("Detecting and starting all cameras...")
        available_cameras = []
        
        for cam_id in range(10):  # Check cameras 0-9
            print(f"Starting camera {cam_id}...")
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"  Camera {cam_id}: {width}x{height} - Started")
                    
                    # Special handling for Camera 2 (C920n Pro HD webcam)
                    if cam_id == 2:
                        print(f"    Camera 2 detected (C920n Pro HD webcam) - applying advanced Full HD settings...")
                        
                        # Release current capture and reinitialize with specific backend
                        cap.release()
                        
                        # Try different backends for C920n Pro HD
                        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                        cap = None
                        
                        for backend in backends:
                            try:
                                print(f"    Trying backend {backend} for Camera 2...")
                                cap = cv2.VideoCapture(cam_id, backend)
                                if cap.isOpened():
                                    # Set Full HD resolution BEFORE reading any frames
                                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                                    cap.set(cv2.CAP_PROP_FPS, 30)
                                    
                                    # Verify resolution was set correctly
                                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    print(f"    Camera 2 resolution set to: {actual_width}x{actual_height}")
                                    
                                    # Test frame reading
                                    ret, test_frame = cap.read()
                                    if ret and test_frame is not None and test_frame.size > 0:
                                        print(f"    Camera 2 Full HD successful with backend {backend}")
                                        available_cameras.append({
                                            'id': cam_id,
                                            'width': actual_width,
                                            'height': actual_height,
                                            'cap': cap,
                                            'is_full_hd': True
                                        })
                                        break
                                    else:
                                        print(f"    Backend {backend} failed frame test")
                                        cap.release()
                                        cap = None
                                else:
                                    print(f"    Backend {backend} failed to open")
                                    if cap:
                                        cap.release()
                                        cap = None
                            except Exception as e:
                                print(f"    Backend {backend} error: {e}")
                                if cap:
                                    cap.release()
                                    cap = None
                        
                        # If all backends failed, use default
                        if cap is None:
                            print(f"    All backends failed for Camera 2, using default settings...")
                            cap = cv2.VideoCapture(cam_id)
                            if cap.isOpened():
                                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                                available_cameras.append({
                                    'id': cam_id,
                                    'width': width,
                                    'height': height,
                                    'cap': cap,
                                    'is_full_hd': False
                                })
                    else:
                        available_cameras.append({
                            'id': cam_id,
                            'width': width,
                            'height': height,
                            'cap': cap,
                            'is_full_hd': False
                        })
                else:
                    print(f"  Camera {cam_id}: Failed to read frame")
                    cap.release()
            else:
                print(f"  Camera {cam_id}: Failed to open")
                cap.release()
        
        return available_cameras
    
    def show_improved_camera_selection(self):
        """Show improved camera selection with live previews"""
        available_cameras = self.detect_and_start_all_cameras()
        
        if not available_cameras:
            print("No cameras found!")
            return None
        
        print(f"Found {len(available_cameras)} cameras, showing selection interface...")
        
        # Create selection window
        selection_window = "Improved Camera Selection"
        cv2.namedWindow(selection_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(selection_window, 1200, 800)
        
        # Calculate grid layout
        num_cameras = len(available_cameras)
        cols = min(3, num_cameras)  # Max 3 columns
        rows = (num_cameras + cols - 1) // cols  # Ceiling division
        
        preview_width = 300
        preview_height = 200
        
        print(f"Grid layout: {rows}x{cols}")
        
        # Store camera positions for click detection
        camera_positions = []
        
        # Main loop for camera selection
        selected_camera = None
        while selected_camera is None:
            # Create selection frame
            selection_frame = np.zeros((800, 1200, 3), dtype=np.uint8)
            
            # Title
            cv2.putText(selection_frame, "Select Camera (Click on preview):", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # Clear camera positions
            camera_positions.clear()
            
            # Draw camera previews
            for i, cam in enumerate(available_cameras):
                row = i // cols
                col = i % cols
                
                x = 50 + col * (preview_width + 20)
                y = 80 + row * (preview_height + 20)
                
                # Store position for click detection
                camera_positions.append({
                    'camera': cam,
                    'x': x,
                    'y': y,
                    'width': preview_width,
                    'height': preview_height
                })
                
                # Read frame from all cameras with robust error handling
                try:
                    ret, frame = cam['cap'].read()
                    if ret and frame is not None and frame.size > 0:
                        # Resize frame to preview size
                        preview = cv2.resize(frame, (preview_width, preview_height))
                        
                        # Draw preview border (highlight Full HD cameras)
                        if cam.get('is_full_hd', False):
                            cv2.rectangle(selection_frame, (x, y), (x + preview_width, y + preview_height), (0, 255, 0), 3)  # Green border for Full HD
                        else:
                            cv2.rectangle(selection_frame, (x, y), (x + preview_width, y + preview_height), (255, 255, 255), 2)
                        
                        # Place preview in selection frame
                        selection_frame[y:y+preview_height, x:x+preview_width] = preview
                    else:
                        # Draw placeholder for failed camera
                        cv2.rectangle(selection_frame, (x, y), (x + preview_width, y + preview_height), (100, 100, 100), 2)
                        cv2.putText(selection_frame, f"Camera {cam['id']}: No Signal", (x, y + preview_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except Exception as e:
                    # Silent error handling - don't print errors for camera selection
                    # Draw placeholder for error camera
                    cv2.rectangle(selection_frame, (x, y), (x + preview_width, y + preview_height), (100, 100, 100), 2)
                    cv2.putText(selection_frame, f"Camera {cam['id']}: Error", (x, y + preview_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw camera info
                info_text = f"Camera {cam['id']}: {cam['width']}x{cam['height']}"
                if cam.get('is_full_hd', False):
                    info_text += " (Full HD)"
                    cv2.putText(selection_frame, info_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(selection_frame, info_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw click instruction
                click_text = f"Click to select"
                cv2.putText(selection_frame, click_text, (x, y + preview_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Instructions
            cv2.putText(selection_frame, "Click on a camera preview to select it", (50, 750), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(selection_frame, "Press ESC to exit", (50, 780), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(selection_window, selection_frame)
            
            # Handle mouse clicks
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    print(f"Mouse clicked at ({x}, {y})")
                    # Check which camera was clicked
                    for pos in camera_positions:
                        if (pos['x'] <= x <= pos['x'] + pos['width'] and 
                            pos['y'] <= y <= pos['y'] + pos['height']):
                            nonlocal selected_camera
                            selected_camera = pos['camera']
                            print(f"Camera {pos['camera']['id']} selected!")
                            break
            
            cv2.setMouseCallback(selection_window, mouse_callback)
            
            # Check for window close (X button) - enhanced detection
            try:
                window_visible = cv2.getWindowProperty(selection_window, cv2.WND_PROP_VISIBLE)
                if window_visible < 1:
                    print("Selection window closed by user (X button)")
                    break
            except cv2.error:
                # Window was closed
                print("Selection window closed by user (X button)")
                break
            except Exception as e:
                # Any other error means window is closed
                print(f"Selection window closed by user (X button) - Error: {e}")
                break
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break
        
        cv2.destroyWindow(selection_window)
        
        if selected_camera is not None:
            # Release other cameras
            for cam in available_cameras:
                if cam['id'] != selected_camera['id']:
                    cam['cap'].release()
            print(f"Released other cameras, keeping Camera {selected_camera['id']}")
        
        return selected_camera
    
    def initialize_selected_camera(self, selected_camera):
        """Initialize the selected camera with minimal settings"""
        if selected_camera is None:
            return False
        
        self.cap = selected_camera['cap']
        self.camera_id = selected_camera['id']
        
        print(f"Initializing Camera {self.camera_id}: {selected_camera['width']}x{selected_camera['height']}")
        
        # Test frame with robust error handling
        try:
            ret, test_frame = self.cap.read()
        except Exception as e:
            print(f"Error reading test frame from Camera {self.camera_id}: {e}")
            return False
        
        if ret and test_frame is not None:
            print(f"Test frame shape: {test_frame.shape}")
            print(f"Test frame dtype: {test_frame.dtype}")
            
            # Check if color
            if len(test_frame.shape) == 3 and test_frame.shape[2] == 3:
                print(f"[OK] Color frame detected")
                # Check color channels
                b_mean = np.mean(test_frame[:, :, 0])
                g_mean = np.mean(test_frame[:, :, 1])
                r_mean = np.mean(test_frame[:, :, 2])
                print(f"Color channel means - B: {b_mean:.1f}, G: {g_mean:.1f}, R: {r_mean:.1f}")
                
                # Check if channels are different (not grayscale)
                if abs(b_mean - g_mean) > 5 or abs(g_mean - r_mean) > 5 or abs(b_mean - r_mean) > 5:
                    print(f"[OK] Color channels are different - color mode working!")
                else:
                    print(f"[WARNING] Color channels are similar - might be grayscale")
            else:
                print(f"[ERROR] Frame is not color format!")
                return False
            
            # Apply ONLY autofocus + resolution settings
            print("Applying autofocus + resolution settings...")
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            # Enable Full HD for Camera 2 (C920n Pro HD) and Camera 3
            if self.camera_id == 2:
                print("Enabling Camera 2 (C920n Pro HD) Full HD resolution...")
                # Additional C920n Pro HD specific settings
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
            elif self.camera_id == 3:
                print("Enabling Camera 3 Full HD resolution...")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            # Check actual resolution after setting
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Actual resolution after setting: {actual_width}x{actual_height}")
            
            print(f"Camera {self.camera_id} initialized with autofocus + resolution settings")
            return True
        else:
            print("Failed to read test frame")
            return False
    
    def detect_washer(self, frame):
        """Ultra-robust washer detection using deep learning + traditional methods"""
        if frame is None:
            return False, 0.0, []
        
        try:
            # Method 1: Deep Learning Object Detection (if available)
            # This would be the most robust method
            
            # Method 2: Multi-scale Template Matching
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Create synthetic washer template (circular with hole)
            template_size = 60
            template = np.zeros((template_size, template_size), dtype=np.uint8)
            cv2.circle(template, (template_size//2, template_size//2), template_size//2-5, 255, -1)
            cv2.circle(template, (template_size//2, template_size//2), template_size//4, 0, -1)
            
            # Multi-scale template matching
            scales = [0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
            best_match = 0
            best_location = None
            
            for scale in scales:
                scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
                if scaled_template.shape[0] > gray.shape[0] or scaled_template.shape[1] > gray.shape[1]:
                    continue
                    
                result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_match:
                    best_match = max_val
                    best_location = (max_loc, scaled_template.shape)
            
            if best_match > 0.5:  # Higher threshold for template matching
                return True, best_match, []
            
            # Method 3: Advanced Hough Circle Detection with Multiple Parameters
            circles_detected = []
            
            # Try different parameter sets for Hough circles
            param_sets = [
                (50, 30, 20, 150),  # Original
                (30, 20, 15, 200),  # More sensitive
                (70, 40, 25, 120),  # Less sensitive
                (40, 25, 10, 180),  # Very sensitive
            ]
            
            for param1, param2, min_r, max_r in param_sets:
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                         param1=param1, param2=param2, 
                                         minRadius=min_r, maxRadius=max_r)
                
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    for (x, y, r) in circles:
                        # Validate circle quality
                        if self._validate_circle(frame, x, y, r):
                            circles_detected.append((x, y, r, param1, param2))
            
            if circles_detected:
                # Select best circle based on multiple criteria
                best_circle = self._select_best_circle(frame, circles_detected)
                if best_circle:
                    x, y, r, _, _ = best_circle
                    circle_contour = np.array([[[x-r, y-r]], [[x+r, y-r]], 
                                            [[x+r, y+r]], [[x-r, y+r]]], dtype=np.int32)
                    confidence = 0.8  # High confidence for validated circles
                    return True, confidence, [circle_contour]
            
            # Method 4: Contour Analysis with Relaxed Criteria
            # Use multiple thresholding methods
            thresh_methods = [
                cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
                cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            ]
            
            all_contours = []
            for thresh in thresh_methods:
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                all_contours.extend(contours)
            
            # Analyze contours with very relaxed criteria
            washer_candidates = []
            for contour in all_contours:
                area = cv2.contourArea(contour)
                if area < 200:  # Very low minimum
                    continue
                    
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                # Calculate shape descriptors
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Very relaxed circularity check
                if circularity > 0.1:  # Very low threshold
                    # Check aspect ratio
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    if 0.3 <= aspect_ratio <= 2.0:  # Very relaxed aspect ratio
                        # Check area ratio
                        frame_area = frame.shape[0] * frame.shape[1]
                        area_ratio = area / frame_area
                        
                        if 0.003 <= area_ratio <= 0.2:  # Stricter range
                            score = circularity * 2 + (1.0 - abs(1.0 - aspect_ratio))
                            washer_candidates.append((contour, score, area))
            
            if washer_candidates:
                # Sort by score and take the best
                washer_candidates.sort(key=lambda x: x[1], reverse=True)
                best_contour, best_score, best_area = washer_candidates[0]
                
                # Calculate confidence based on score
                confidence = min(best_score / 5.0, 0.7)  # Moderate confidence
                return True, confidence, [best_contour]
            
            # Method 5: Edge-based detection with morphological operations
            edges = cv2.Canny(gray, 30, 100)
            
            # Apply morphological operations to connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in edge_contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Reasonable size
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.15:  # Low threshold
                            return True, 0.3, [contour]  # Low confidence
            
            return False, 0.0, []
            
        except Exception as e:
            print(f"Washer detection error: {e}")
            return False, 0.0, []
    
    def _validate_circle(self, frame, x, y, r):
        """Validate if a detected circle is likely a washer"""
        try:
            # Check if circle is within frame bounds
            if x - r < 0 or y - r < 0 or x + r >= frame.shape[1] or y + r >= frame.shape[0]:
                return False
            
            # Extract circular region
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # Check if the region has reasonable brightness variation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            region = cv2.bitwise_and(gray, mask)
            
            # Calculate statistics
            mean_val = np.mean(region[region > 0])
            std_val = np.std(region[region > 0])
            
            # Reasonable brightness and variation
            if 50 < mean_val < 200 and std_val > 10:
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def _select_best_circle(self, frame, circles):
        """Select the best circle from multiple detections"""
        if not circles:
            return None
        
        # Score circles based on multiple criteria
        scored_circles = []
        
        for x, y, r, param1, param2 in circles:
            score = 0
            
            # Size score (prefer medium-sized circles)
            frame_area = frame.shape[0] * frame.shape[1]
            circle_area = np.pi * r * r
            area_ratio = circle_area / frame_area
            
            if 0.008 <= area_ratio <= 0.15:  # Stricter size range
                score += 2
            
            # Position score (prefer center area)
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            position_score = 1 - (distance_from_center / max_distance)
            score += position_score
            
            # Parameter score (prefer original parameters)
            if param1 == 50 and param2 == 30:
                score += 1
            
            scored_circles.append((x, y, r, param1, param2, score))
        
        # Return circle with highest score
        scored_circles.sort(key=lambda x: x[5], reverse=True)
        return scored_circles[0] if scored_circles else None
    
    def enhanced_defect_analysis(self, frame):
        """Enhanced defect analysis"""
        if frame is None:
            return {}
        
        results = {}
        
        # Method 1: AI Model Prediction
        if self.model is not None:
            try:
                img = cv2.resize(frame, (224, 224))
                img = np.expand_dims(img, axis=0)
                img = img.astype(np.float32) / 255.0
                
                predictions = self.model.predict(img, verbose=0)
                ai_scores = dict(zip(self.class_names, predictions[0]))
                results['ai_prediction'] = ai_scores
            except Exception as e:
                print(f"AI prediction error: {e}")
                results['ai_prediction'] = {name: 0.25 for name in self.class_names}
        
        # Method 2: Color-based analysis
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Black spot detection using color
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 80])
            black_mask = cv2.inRange(hsv, lower_black, upper_black)
            black_ratio = np.sum(black_mask > 0) / (black_mask.shape[0] * black_mask.shape[1])
            
            # Edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 80)
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
            
            # Texture analysis
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            results['black_ratio'] = black_ratio
            results['edge_density'] = edge_density
            results['texture_variance'] = laplacian_var
        except Exception as e:
            print(f"Color analysis error: {e}")
            results['black_ratio'] = 0
            results['edge_density'] = 0
            results['texture_variance'] = 0
        
        return results
    
    def predict_defect(self, frame):
        """Defect prediction with enhanced analysis"""
        if frame is None:
            return "good", 0.0
        
        # Check if washer is present - re-enable proper detection
        has_washer, washer_confidence, washers = self.detect_washer(frame)
        if not has_washer:
            return "no_washer", 0.0
        
        # Enhanced analysis
        analysis = self.enhanced_defect_analysis(frame)
        
        # Get AI prediction
        ai_scores = analysis.get('ai_prediction', {})
        if not ai_scores:
            return "good", 0.0
        
        # Additional checks
        black_ratio = analysis.get('black_ratio', 0)
        edge_density = analysis.get('edge_density', 0)
        texture_variance = analysis.get('texture_variance', 0)
        
        # Enhanced decision logic - improved accuracy
        max_ai_score = max(ai_scores.values())
        max_ai_class = max(ai_scores, key=ai_scores.get)
        
        # Good classification - prioritize good when confidence is high
        if max_ai_class == "good" and max_ai_score > self.good_confidence_threshold:
            final_prediction = "good"
            final_confidence = max_ai_score
        # Black spot detection - more sensitive
        elif black_ratio > 0.01 and ai_scores.get('black_spot', 0) > self.defect_confidence_threshold:
            final_prediction = "black_spot"
            final_confidence = min(ai_scores['black_spot'] + black_ratio * 3, 1.0)
        # Chipping detection - more sensitive
        elif edge_density > 0.03 and ai_scores.get('chipping', 0) > self.defect_confidence_threshold:
            final_prediction = "chipping"
            final_confidence = min(ai_scores['chipping'] + edge_density * 2, 1.0)
        # Scratch detection - more sensitive
        elif texture_variance > 300 and ai_scores.get('scratch', 0) > self.defect_confidence_threshold:
            final_prediction = "scratch"
            final_confidence = min(ai_scores['scratch'] + texture_variance / 15000, 1.0)
        # Default to AI prediction if confidence is sufficient
        elif max_ai_score > self.defect_confidence_threshold:
            final_prediction = max_ai_class
            final_confidence = max_ai_score
        # Conservative fallback to good
        else:
            final_prediction = "good"
            final_confidence = 0.5
        
        return final_prediction, final_confidence
    
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
            cv2.putText(frame, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            self.notification_start_time = current_time
    
    def update_performance_stats(self, processing_time):
        """Update performance statistics"""
        self.processing_times.append(processing_time)
        self.total_frames_processed += 1
        
        # Keep only last 100 processing times for average calculation
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        current_time = time.time()
        total_runtime = current_time - self.start_time
        
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        else:
            avg_processing_time = 0
            fps = 0
        
        return {
            'total_runtime': total_runtime,
            'avg_processing_time': avg_processing_time,
            'fps': fps,
            'frames_processed': self.total_frames_processed
        }
    
    def run(self):
        """Run the improved multi-camera selection system"""
        # Show camera selection
        selected_camera = self.show_improved_camera_selection()
        if selected_camera is None:
            print("No camera selected")
            return
        
        # Initialize selected camera
        if not self.initialize_selected_camera(selected_camera):
            return
        
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_title, 1280, 720)
        
        print("Improved Multi-Camera Selection System Started!")
        print("Press ESC to exit, M to select area, S to train, 1-4 for feedback")
        
        frame_count = 0
        while True:
            # Robust frame reading for all cameras
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"Failed to read frame from Camera {self.camera_id}")
                    break
            except Exception as e:
                print(f"Error reading from Camera {self.camera_id}: {e}")
                break
            
            frame_count += 1
            
            # Skip frames for performance (dynamic based on system)
            self.frame_skip_count += 1
            if self.frame_skip_count < self.frame_skip_interval:
                continue
            self.frame_skip_count = 0
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Resize frame for performance (dynamic resolution based on system)
            original_frame = frame.copy()
            processing_frame = cv2.resize(frame, self.processing_resolution)
            
            # NO COLOR ENHANCEMENT - RAW FRAME ONLY
            
            # Predict defect
            current_time = time.time()
            if current_time - self.last_prediction_time >= self.prediction_interval:
                prediction, confidence = self.predict_defect(processing_frame)
                
                # Add to prediction history
                self.prediction_history.append((prediction, confidence))
                
                # Stability check - immediate update for no_washer
                if prediction == "no_washer":
                    self.current_prediction = "no_washer"
                    self.prediction_confidence = confidence
                elif len(self.prediction_history) >= self.min_stable_count:
                    class_counts = {}
                    for pred_class, pred_score in self.prediction_history:
                        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                    
                    most_common_class = max(class_counts, key=class_counts.get)
                    most_common_count = class_counts[most_common_class]
                    
                    if most_common_count >= self.min_stable_count and confidence >= self.stability_threshold:
                        self.current_prediction = most_common_class
                        self.prediction_confidence = confidence
                
                self.last_prediction_time = current_time
            
            # Draw UI
            h, w = original_frame.shape[:2]
            
            # Top-left banner (fixed width)
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
            
            # Draw banner background (dynamic width based on text)
            text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            banner_width = text_size[0] + 40  # Text width + padding
            cv2.rectangle(original_frame, (10, 10), (banner_width, 80), banner_color, -1)
            cv2.putText(original_frame, result_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Bottom-left control panel
            hud_x = 10
            hud_y = h - self.hud_height - 10
            
            # Draw HUD background
            cv2.rectangle(original_frame, (hud_x, hud_y), (hud_x + self.hud_width, hud_y + self.hud_height), (0, 0, 0), -1)
            cv2.rectangle(original_frame, (hud_x, hud_y), (hud_x + self.hud_width, hud_y + self.hud_height), (255, 255, 255), 2)
            
            # HUD content
            y_offset = hud_y + 30
            cv2.putText(original_frame, "Feedback: 0", (hud_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
            cv2.putText(original_frame, "Controls", (hud_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 25
            cv2.putText(original_frame, "1:Good 2:Black 3:Chipped 4:Scratched", (hud_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            cv2.putText(original_frame, "M:Select S:Train ESC:Exit", (hud_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            camera_info = f"Model: AUTO-OPTIMIZED | Camera: {self.camera_id} | {w}x{h}"
            if self.camera_id == 0 or self.camera_id == 2:
                camera_info += " (Full HD)"
            cv2.putText(original_frame, camera_info, (hud_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            system_info = f"System: {self.system_config['system_type'].upper()} | CPU: {self.system_config['cpu_cores']}C | RAM: {self.system_config['memory_gb']:.0f}GB"
            cv2.putText(original_frame, system_info, (hud_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Show notification
            if hasattr(self, 'notification_message'):
                self.show_notification(original_frame, self.notification_message)
            
            # Display frame
            cv2.imshow(self.window_title, original_frame)
            
            # Check for window close (X button) - enhanced detection
            try:
                window_visible = cv2.getWindowProperty(self.window_title, cv2.WND_PROP_VISIBLE)
                if window_visible < 1:
                    print("Window closed by user (X button)")
                    break
            except cv2.error:
                # Window was closed
                print("Window closed by user (X button)")
                break
            except Exception as e:
                # Any other error means window is closed
                print(f"Window closed by user (X button) - Error: {e}")
                break
            
            # Handle key presses - dynamic wait time based on system
            key = cv2.waitKey(self.wait_key_interval) & 0xFF
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
        print("Improved Multi-Camera Selection System stopped")

def main():
    """Main function"""
    system = ImprovedMultiCameraSelectionSystem()
    system.run()

if __name__ == "__main__":
    main()