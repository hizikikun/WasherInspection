#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network-Enabled Washer Inspection System
- Desktop PC: WebSocket Server mode
- Notebook PC: WebSocket Client mode
- Real-time data synchronization between devices
"""

import cv2
import numpy as np
import tensorflow as tf
import os
import json
import time
import platform
import psutil
import asyncio
import websockets
import threading
from collections import deque
from datetime import datetime

class NetworkWasherInspectionSystem:
    def __init__(self, mode='auto'):
        """
        Initialize the network-enabled washer inspection system
        mode: 'server', 'client', or 'auto'
        """
        self.mode = mode
        self.cap = None
        self.model = None
        self.class_names = ['good', 'chipping', 'black_spot', 'scratch']
        self.window_title = "Network Washer Inspection System"
        self.camera_id = None
        
        # Network settings
        self.server_host = "192.168.1.100"  # Desktop PC IP
        self.server_port = 8765
        self.websocket = None
        self.connected = False
        
        # Prediction settings - will be set by system detection
        self.current_prediction = "good"
        self.prediction_confidence = 0.0
        self.last_prediction_time = 0
        
        # Defect detection thresholds
        self.defect_confidence_threshold = 0.25
        self.good_confidence_threshold = 0.6
        
        # UI settings
        self.hud_height = 200
        self.hud_width = 800
        
        # Auto-detect system configuration
        self.system_config = self.detect_system_configuration()
        self.apply_system_optimizations()
        
        # Performance monitoring
        self.start_time = time.time()
        self.processing_times = []
        self.total_frames_processed = 0
        
        # Network data
        self.shared_data = {
            'inspection_results': [],
            'performance_stats': {},
            'system_info': {},
            'last_update': None
        }
        
        # Load model
        self.load_model()
        
        # Start network communication
        if self.mode == 'auto':
            self.mode = 'server' if self.system_config['is_notebook'] == False else 'client'
        
        print(f"[NETWORK] Mode: {self.mode.upper()}")
        if self.mode == 'server':
            print(f"[SERVER] Starting WebSocket server on {self.server_host}:{self.server_port}")
        else:
            print(f"[CLIENT] Connecting to server at {self.server_host}:{self.server_port}")
    
    def detect_system_configuration(self):
        """Detect system configuration and determine optimal settings"""
        try:
            # Get CPU information
            cpu_info = platform.processor()
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            
            # Get memory information
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024**3)
            
            # Get GPU information
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
            
            if cpu_count <= 4 and total_memory_gb <= 32:
                is_notebook = True
                print("[NOTEBOOK] Notebook PC detected")
            else:
                print("[DESKTOP] Desktop PC detected")
            
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
            self.prediction_history = deque(maxlen=3)
            self.stability_threshold = 0.4
            self.min_stable_count = 2
            self.prediction_interval = 1.5
            self.frame_skip_count = 0
            self.frame_skip_interval = 4
            self.processing_resolution = (480, 360)
            self.wait_key_interval = 150
            
        elif config['is_high_end']:
            print("[OPTIMIZE] Applying high-end desktop optimizations...")
            self.prediction_history = deque(maxlen=20)
            self.stability_threshold = 0.1
            self.min_stable_count = 8
            self.prediction_interval = 0.1
            self.frame_skip_count = 0
            self.frame_skip_interval = 1
            self.processing_resolution = (1280, 720)
            self.wait_key_interval = 1
            
        else:
            print("[OPTIMIZE] Applying standard desktop optimizations...")
            self.prediction_history = deque(maxlen=10)
            self.stability_threshold = 0.2
            self.min_stable_count = 4
            self.prediction_interval = 0.5
            self.frame_skip_count = 0
            self.frame_skip_interval = 2
            self.processing_resolution = (640, 480)
            self.wait_key_interval = 30
        
        print(f"Optimization settings applied:")
        print(f"  Prediction interval: {self.prediction_interval}s")
        print(f"  Frame skip interval: {self.frame_skip_interval}")
        print(f"  プロcessing resolution: {self.processing_resolution}")
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
    
    # WebSocket Server Functions
    async def websocket_server_handler(self, websocket, path):
        """Handle WebSocket connections for server mode"""
        print(f"[SERVER] Client connected: {websocket.remote_address}")
        self.connected = True
        
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.handle_client_message(data, websocket)
        except websockets.exceptions.ConnectionClosed:
            print("[SERVER] Client disconnected")
        finally:
            self.connected = False
    
    async def handle_client_message(self, data, websocket):
        """Handle messages from client"""
        message_type = data.get('type')
        
        if message_type == 'inspection_result':
            # Store inspection result
            self.shared_data['inspection_results'].append({
                'timestamp': datetime.now().isoformat(),
                'prediction': data.get('prediction'),
                'confidence': data.get('confidence'),
                'source': 'client'
            })
            
            # Keep only last 100 results
            if len(self.shared_data['inspection_results']) > 100:
                self.shared_data['inspection_results'].pop(0)
            
            print(f"[SERVER] Received inspection result: {data.get('prediction')} ({data.get('confidence'):.2f})")
            
        elif message_type == 'performance_stats':
            # Update performance stats
            self.shared_data['performance_stats'] = data.get('stats', {})
            self.shared_data['last_update'] = datetime.now().isoformat()
            
        elif message_type == 'request_data':
            # Send current data to client
            response = {
                'type': 'data_response',
                'data': self.shared_data
            }
            await websocket.send(json.dumps(response))
    
    async def start_server(self):
        """Start WebSocket server"""
        print(f"[SERVER] Starting WebSocket server on {self.server_host}:{self.server_port}")
        async with websockets.serve(self.websocket_server_handler, self.server_host, self.server_port):
            print("[SERVER] WebSocket server started")
            await asyncio.Future()  # Run forever
    
    # WebSocket Client Functions
    async def connect_to_server(self):
        """Connect to WebSocket server"""
        try:
            uri = f"ws://{self.server_host}:{self.server_port}"
            self.websocket = await websockets.connect(uri)
            self.connected = True
            print(f"[CLIENT] Connected to server at {uri}")
            
            # Start listening for messages
            asyncio.create_task(self.listen_for_messages())
            
        except Exception as e:
            print(f"[CLIENT] Failed to connect to server: {e}")
            self.connected = False
    
    async def listen_for_messages(self):
        """Listen for messages from server"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self.handle_server_message(data)
        except websockets.exceptions.ConnectionClosed:
            print("[CLIENT] Connection to server lost")
            self.connected = False
    
    async def handle_server_message(self, data):
        """Handle messages from server"""
        message_type = data.get('type')
        
        if message_type == 'data_response':
            # Update shared data
            self.shared_data = data.get('data', {})
            print(f"[CLIENT] Received data from server")
    
    async def send_to_server(self, message_type, data):
        """Send data to server"""
        if self.websocket and self.connected:
            message = {
                'type': message_type,
                'timestamp': datetime.now().isoformat(),
                **data
            }
            try:
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                print(f"[CLIENT] Failed to send message: {e}")
                self.connected = False
    
    # Inspection Functions (same as before)
    def detect_washer(self, frame):
        """Lightweight washer detection"""
        if frame is None:
            return False, 0.0, []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Simple Hough Circle Detection
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, 
                                     minRadius=20, maxRadius=150)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    if self._validate_circle(frame, x, y, r):
                        circle_contour = np.array([[[x-r, y-r]], [[x+r, y-r]], 
                                                [[x+r, y+r]], [[x-r, y+r]]], dtype=np.int32)
                        return True, 0.8, [circle_contour]
            
            # Simple contour analysis
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 50000:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.3:
                            return True, 0.6, [contour]
            
            return False, 0.0, []
            
        except Exception as e:
            print(f"Washer detection error: {e}")
            return False, 0.0, []
    
    def _validate_circle(self, frame, x, y, r):
        """Validate if a detected circle is likely a washer"""
        try:
            if x - r < 0 or y - r < 0 or x + r >= frame.shape[1] or y + r >= frame.shape[0]:
                return False
            
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            region = cv2.bitwise_and(gray, mask)
            
            mean_val = np.mean(region[region > 0])
            std_val = np.std(region[region > 0])
            
            if 50 < mean_val < 200 and std_val > 10:
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def predict_defect(self, frame):
        """Defect prediction with network synchronization"""
        if frame is None:
            return "good", 0.0
        
        # Check if washer is present
        has_washer, washer_confidence, washers = self.detect_washer(frame)
        if not has_washer:
            return "no_washer", 0.0
        
        # AI prediction
        if self.model is not None:
            try:
                img = cv2.resize(frame, (224, 224))
                img = np.expand_dims(img, axis=0)
                img = img.astype(np.float32) / 255.0
                
                predictions = self.model.predict(img, verbose=0)
                ai_scores = dict(zip(self.class_names, predictions[0]))
                
                max_ai_score = max(ai_scores.values())
                max_ai_class = max(ai_scores, key=ai_scores.get)
                
                if max_ai_class == "good" and max_ai_score > self.good_confidence_threshold:
                    final_prediction = "good"
                    final_confidence = max_ai_score
                elif max_ai_score > self.defect_confidence_threshold:
                    final_prediction = max_ai_class
                    final_confidence = max_ai_score
                else:
                    final_prediction = "good"
                    final_confidence = 0.5
                
                # Send result to network
                if self.mode == 'client' and self.connected:
                    asyncio.create_task(self.send_to_server('inspection_result', {
                        'prediction': final_prediction,
                        'confidence': final_confidence,
                        'system_type': self.system_config['system_type']
                    }))
                
                return final_prediction, final_confidence
                
            except Exception as e:
                print(f"AI prediction error: {e}")
                return "good", 0.0
        
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
    
    def update_performance_stats(self, processing_time):
        """Update performance statistics"""
        self.processing_times.append(processing_time)
        self.total_frames_processed += 1
        
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
        """Run the network-enabled inspection system"""
        # Start network communication in background
        if self.mode == 'server':
            # Start server in background thread
            server_thread = threading.Thread(target=lambda: asyncio.run(self.start_server()))
            server_thread.daemon = True
            server_thread.start()
            time.sleep(2)  # Wait for server to start
        else:
            # Connect to server
            asyncio.run(self.connect_to_server())
            time.sleep(2)  # Wait for connection
        
        # Camera detection and selection (simplified)
        print("Detecting cameras...")
        for cam_id in range(5):
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    self.camera_id = cam_id
                    print(f"Camera {cam_id} selected")
                    break
                cap.release()
        
        if self.cap is None:
            print("No camera found!")
            return
        
        # Initialize camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_title, 1280, 720)
        
        print(f"Network Washer Inspection System Started! ({self.mode.upper()} mode)")
        print("Press ESC to exit, 1-4 for feedback")
        
        frame_count = 0
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"Failed to read frame from Camera {self.camera_id}")
                    break
            except Exception as e:
                print(f"Error reading from Camera {self.camera_id}: {e}")
                break
            
            frame_count += 1
            
            # Skip frames for performance
            self.frame_skip_count += 1
            if self.frame_skip_count < self.frame_skip_interval:
                continue
            self.frame_skip_count = 0
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Resize for processing
            original_frame = frame.copy()
            processing_frame = cv2.resize(frame, self.processing_resolution)
            
            # Predict defect
            current_time = time.time()
            if current_time - self.last_prediction_time >= self.prediction_interval:
                prediction, confidence = self.predict_defect(processing_frame)
                
                # Add to prediction history
                self.prediction_history.append((prediction, confidence))
                
                # Stability check
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
            
            # Result banner
            if self.current_prediction == "no_washer":
                result_text = "Result: No Washer Detected"
                banner_color = (0, 165, 255)
            elif self.current_prediction == "good":
                result_text = "Result: OK - Good"
                banner_color = (0, 255, 0)
            else:
                reason = self.get_defect_reason(self.current_prediction)
                result_text = f"Result: NG - {reason}"
                banner_color = (0, 0, 255)
            
            # Draw banner
            text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            banner_width = text_size[0] + 40
            cv2.rectangle(original_frame, (10, 10), (banner_width, 80), banner_color, -1)
            cv2.putText(original_frame, result_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Network status
            network_status = f"Network: {'CONNECTED' if self.connected else 'DISCONNECTED'}"
            cv2.putText(original_frame, network_status, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.connected else (0, 0, 255), 2)
            
            # Performance stats
            stats = self.get_performance_stats()
            perf_text = f"FPS: {stats['fps']:.1f} | Frames: {stats['frames_processed']} | Runtime: {stats['total_runtime']:.1f}s"
            cv2.putText(original_frame, perf_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # System info
            system_text = f"Mode: {self.mode.upper()} | System: {self.system_config['system_type'].upper()}"
            cv2.putText(original_frame, system_text, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow(self.window_title, original_frame)
            
            # Handle key presses
            key = cv2.waitKey(self.wait_key_interval) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('1'):
                print("Feedback: Good")
            elif key == ord('2'):
                print("Feedback: Black Spot")
            elif key == ord('3'):
                print("Feedback: Chipped")
            elif key == ord('4'):
                print("Feedback: Scratched")
        
        # Cleanup
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Network Washer Inspection System stopped")

def main():
    """Main function"""
    import sys
    
    mode = 'auto'
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    
    system = NetworkWasherInspectionSystem(mode=mode)
    system.run()

if __name__ == "__main__":
    main()
