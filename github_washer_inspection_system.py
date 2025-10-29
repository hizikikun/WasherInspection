#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub-Integrated Washer Inspection System
- Automatically sends inspection results to GitHub Issues
- Uses GitHub Actions for automated processing
- Supports both desktop and notebook PC
"""

import cv2
import numpy as np
import tensorflow as tf
import os
import json
import time
import platform
import psutil
import requests
import base64
from datetime import datetime
from collections import deque

class GitHubWasherInspectionSystem:
    def __init__(self, github_config_file="github_config.json"):
        """
        Initialize the GitHub-integrated washer inspection system
        """
        self.cap = None
        self.model = None
        self.class_names = ['good', 'chipping', 'black_spot', 'scratch']
        self.window_title = "GitHub Washer Inspection System"
        self.camera_id = None
        
        # GitHub settings
        self.github_config = self.load_github_config(github_config_file)
        self.github_token = self.github_config.get('github_token')
        self.github_repo = self.github_config.get('github_repo')
        self.github_owner = self.github_config.get('github_owner')
        
        # Prediction settings
        self.current_prediction = "good"
        self.prediction_confidence = 0.0
        self.last_prediction_time = 0
        
        # Defect detection thresholds
        self.defect_confidence_threshold = 0.25
        self.good_confidence_threshold = 0.6
        
        # UI settings
        self.hud_height = 220
        self.hud_width = 800
        
        # Auto-detect system configuration
        self.system_config = self.detect_system_configuration()
        self.apply_system_optimizations()
        
        # Performance monitoring
        self.start_time = time.time()
        self.processing_times = []
        self.total_frames_processed = 0
        
        # Inspection history
        self.inspection_history = []
        self.ng_count = 0
        self.good_count = 0
        
        # Load model
        self.load_model()
        
        print(f"[GITHUB] Repository: {self.github_owner}/{self.github_repo}")
        print(f"[GITHUB] Auto-send enabled: {self.github_config.get('auto_send', True)}")
    
    def load_github_config(self, config_file):
        """Load GitHub configuration"""
        default_config = {
            "github_token": "",
            "github_owner": "your-username",
            "github_repo": "washer-inspection-results",
            "auto_send": True,
            "send_interval_minutes": 5,
            "include_images": True,
            "issue_labels": ["inspection", "automated"]
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"Error loading GitHub config: {e}")
                return default_config
        else:
            # Create default config file
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            print(f"Created default GitHub config file: {config_file}")
            print("Please edit the config file with your GitHub credentials")
            return default_config
    
    def detect_system_configuration(self):
        """Detect system configuration"""
        try:
            cpu_info = platform.processor()
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
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
            is_notebook = cpu_count <= 4 and total_memory_gb <= 32
            is_high_end = cpu_count >= 8 and total_memory_gb >= 32 and "RTX" in gpu_info
            
            if is_notebook:
                print("[NOTEBOOK] Notebook PC detected")
            else:
                print("[DESKTOP] Desktop PC detected")
            
            if is_high_end:
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
            self.prediction_interval = 2.0  # Longer interval for GitHub
            self.frame_skip_count = 0
            self.frame_skip_interval = 4
            self.processing_resolution = (480, 360)
            self.wait_key_interval = 150
            
        elif config['is_high_end']:
            print("[OPTIMIZE] Applying high-end desktop optimizations...")
            self.prediction_history = deque(maxlen=20)
            self.stability_threshold = 0.1
            self.min_stable_count = 8
            self.prediction_interval = 0.5
            self.frame_skip_count = 0
            self.frame_skip_interval = 1
            self.processing_resolution = (1280, 720)
            self.wait_key_interval = 1
            
        else:
            print("[OPTIMIZE] Applying standard desktop optimizations...")
            self.prediction_history = deque(maxlen=10)
            self.stability_threshold = 0.2
            self.min_stable_count = 4
            self.prediction_interval = 1.0
            self.frame_skip_count = 0
            self.frame_skip_interval = 2
            self.processing_resolution = (640, 480)
            self.wait_key_interval = 30
        
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
    
    # GitHub API Functions
    def create_github_issue(self, title, body, labels=None):
        """Create a GitHub issue"""
        if not self.github_token or not self.github_repo or not self.github_owner:
            print("[GITHUB] GitHub credentials not configured")
            return False
        
        url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/issues"
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        data = {
            "title": title,
            "body": body,
            "labels": labels or self.github_config.get('issue_labels', [])
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 201:
                issue_data = response.json()
                print(f"[GITHUB] Issue created: #{issue_data['number']} - {title}")
                return issue_data['number']
            else:
                print(f"[GITHUB] Failed to create issue: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"[GITHUB] Error creating issue: {e}")
            return False
    
    def upload_image_to_github(self, image_data, filename):
        """Upload image to GitHub repository"""
        if not self.github_token or not self.github_repo or not self.github_owner:
            return None
        
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', image_data)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/contents/images/{filename}"
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        data = {
            "message": f"Add inspection image: {filename}",
            "content": image_base64
        }
        
        try:
            response = requests.put(url, headers=headers, json=data)
            if response.status_code == 201:
                file_data = response.json()
                return file_data['content']['download_url']
            else:
                print(f"[GITHUB] Failed to upload image: {response.status_code}")
                return None
        except Exception as e:
            print(f"[GITHUB] Error uploading image: {e}")
            return None
    
    def send_inspection_result_to_github(self, prediction, confidence, frame=None):
        """Send inspection result to GitHub"""
        if not self.github_config.get('auto_send', True):
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Update counters
        if prediction == "good":
            self.good_count += 1
        else:
            self.ng_count += 1
        
        # Create issue title
        if prediction == "good":
            title = f"✅ Inspection Result: GOOD - {timestamp}"
            labels = ["inspection", "good", "automated"]
        else:
            title = f"❌ Inspection Result: NG - {prediction.upper()} - {timestamp}"
            labels = ["inspection", "ng", prediction, "automated"]
        
        # Create issue body
        body = f"""## Inspection Result

**Status**: {'✅ GOOD' if prediction == 'good' else '❌ NG'}
**Defect Type**: {prediction.replace('_', ' ').title()}
**Confidence**: {confidence:.2f}
**Timestamp**: {timestamp}

## System Information
- **System Type**: {self.system_config['system_type'].upper()}
- **CPU**: {self.system_config['cpu_info']} ({self.system_config['cpu_cores']} cores)
- **Memory**: {self.system_config['memory_gb']:.1f} GB
- **GPU**: {self.system_config['gpu_info']}

## Statistics
- **Total Inspections**: {self.good_count + self.ng_count}
- **Good Count**: {self.good_count}
- **NG Count**: {self.ng_count}
- **Success Rate**: {(self.good_count / max(1, self.good_count + self.ng_count) * 100):.1f}%

## Performance
- **FPS**: {self.get_performance_stats()['fps']:.1f}
- **Total Runtime**: {self.get_performance_stats()['total_runtime']:.1f}s
- **Frames Processed**: {self.get_performance_stats()['frames_processed']}
"""
        
        # Upload image if available and enabled
        image_url = None
        if frame is not None and self.github_config.get('include_images', True):
            filename = f"inspection_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
            image_url = self.upload_image_to_github(frame, filename)
            if image_url:
                body += f"\n## Inspection Image\n![Inspection Image]({image_url})"
        
        # Create GitHub issue
        issue_number = self.create_github_issue(title, body, labels)
        
        if issue_number:
            # Store in history
            self.inspection_history.append({
                'timestamp': timestamp,
                'prediction': prediction,
                'confidence': confidence,
                'issue_number': issue_number,
                'image_url': image_url
            })
            
            # Keep only last 50 results
            if len(self.inspection_history) > 50:
                self.inspection_history.pop(0)
    
    # Inspection Functions
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
        """Defect prediction with GitHub integration"""
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
        """Run the GitHub-integrated inspection system"""
        # Camera detection and selection
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
        
        print(f"GitHub Washer Inspection System Started!")
        print("Press ESC to exit, S to send manual report, 1-4 for feedback")
        
        frame_count = 0
        last_github_send = time.time()
        
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
                
                # Send to GitHub if NG detected or interval passed
                if (self.current_prediction != "good" and self.current_prediction != "no_washer") or \
                   (current_time - last_github_send >= self.github_config.get('send_interval_minutes', 5) * 60):
                    self.send_inspection_result_to_github(
                        self.current_prediction, 
                        self.prediction_confidence, 
                        original_frame
                    )
                    last_github_send = current_time
            
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
            
            # GitHub status
            github_status = f"GitHub: {'CONNECTED' if self.github_token else 'NOT CONFIGURED'}"
            cv2.putText(original_frame, github_status, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if self.github_token else (0, 0, 255), 2)
            
            # Statistics
            stats_text = f"Good: {self.good_count} | NG: {self.ng_count} | Total: {self.good_count + self.ng_count}"
            cv2.putText(original_frame, stats_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Performance stats
            stats = self.get_performance_stats()
            perf_text = f"FPS: {stats['fps']:.1f} | Runtime: {stats['total_runtime']:.1f}s"
            cv2.putText(original_frame, perf_text, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # System info
            system_text = f"System: {self.system_config['system_type'].upper()} | Repo: {self.github_repo}"
            cv2.putText(original_frame, system_text, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow(self.window_title, original_frame)
            
            # Handle key presses
            key = cv2.waitKey(self.wait_key_interval) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):
                # Manual send to GitHub
                self.send_inspection_result_to_github(
                    self.current_prediction, 
                    self.prediction_confidence, 
                    original_frame
                )
                print("Manual report sent to GitHub")
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
        print("GitHub Washer Inspection System stopped")

def main():
    """Main function"""
    system = GitHubWasherInspectionSystem()
    system.run()

if __name__ == "__main__":
    main()
