#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code and Training Data Auto-Sync System
- Automatically detects code changes and commits to GitHub
- Automatically uploads new training photos
- Organizes training data by class and date
- Generates training progress reports
"""

import os
import json
import time
import shutil
import hashlib
import requests
import base64
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

class CodeTrainingAutoSync:
    def __init__(self, config_file="auto_sync_config.json"):
        """
        Initialize the code and training data auto-sync system
        """
        self.config = self.load_config(config_file)
        self.github_token = self.config.get('github_token')
        self.github_owner = self.config.get('github_owner')
        self.github_repo = self.config.get('github_repo')
        
        # Paths
        self.code_path = self.config.get('code_path', '.')
        self.training_data_path = self.config.get('training_data_path', 'cs_AItraining_data')
        self.backup_path = self.config.get('backup_path', 'backup')
        
        # File tracking
        self.code_hashes = {}
        self.training_files = set()
        self.last_sync_time = time.time()
        
        # Create backup directory
        os.makedirs(self.backup_path, exist_ok=True)
        
        print(f"[AUTO-SYNC] Repository: {self.github_owner}/{self.github_repo}")
        print(f"[AUTO-SYNC] Code path: {self.code_path}")
        print(f"[AUTO-SYNC] Training data path: {self.training_data_path}")
    
    def load_config(self, config_file):
        """Load configuration"""
        default_config = {
            "github_token": "",
            "github_owner": "your-username",
            "github_repo": "washer-inspection-code",
            "code_path": ".",
            "training_data_path": "cs_AItraining_data",
            "backup_path": "backup",
            "sync_interval_minutes": 5,
            "auto_commit": True,
            "auto_organize_training": True,
            "include_code_files": [".py", ".json", ".md", ".yml", ".yaml"],
            "exclude_dirs": ["__pycache__", ".git", "node_modules", ".vscode"]
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"Error loading config: {e}")
                return default_config
        else:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            print(f"Created default config file: {config_file}")
            return default_config
    
    def calculate_file_hash(self, file_path):
        """Calculate MD5 hash of a file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"Error calculating hash for {file_path}: {e}")
            return None
    
    def scan_code_changes(self):
        """Scan for code changes"""
        changes = []
        
        for root, dirs, files in os.walk(self.code_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.config.get('exclude_dirs', [])]
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.code_path)
                
                # Check if file extension is included
                if any(file.endswith(ext) for ext in self.config.get('include_code_files', ['.py'])):
                    current_hash = self.calculate_file_hash(file_path)
                    
                    if current_hash:
                        if relative_path not in self.code_hashes:
                            # New file
                            changes.append({
                                'type': 'new',
                                'path': relative_path,
                                'hash': current_hash,
                                'size': os.path.getsize(file_path)
                            })
                            self.code_hashes[relative_path] = current_hash
                        elif self.code_hashes[relative_path] != current_hash:
                            # Modified file
                            changes.append({
                                'type': 'modified',
                                'path': relative_path,
                                'hash': current_hash,
                                'size': os.path.getsize(file_path)
                            })
                            self.code_hashes[relative_path] = current_hash
        
        return changes
    
    def scan_training_data(self):
        """Scan for new training data"""
        new_training_files = []
        
        if not os.path.exists(self.training_data_path):
            return new_training_files
        
        for root, dirs, files in os.walk(self.training_data_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.training_data_path)
                    
                    if relative_path not in self.training_files:
                        new_training_files.append({
                            'path': relative_path,
                            'full_path': file_path,
                            'class': self.extract_class_from_path(file_path),
                            'size': os.path.getsize(file_path),
                            'modified': os.path.getmtime(file_path)
                        })
                        self.training_files.add(relative_path)
        
        return new_training_files
    
    def extract_class_from_path(self, file_path):
        """Extract class name from file path"""
        path_parts = Path(file_path).parts
        
        # Look for class names in the path
        class_names = ['good', 'chipping', 'black_spot', 'scratch', 'resin']
        
        for part in path_parts:
            if part.lower() in class_names:
                return part.lower()
        
        return 'unknown'
    
    def upload_file_to_github(self, file_path, github_path, commit_message):
        """Upload file to GitHub repository"""
        if not self.github_token:
            print("[GITHUB] No token configured")
            return False
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Encode to base64
            content_base64 = base64.b64encode(content).decode('utf-8')
            
            url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/contents/{github_path}"
            
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            data = {
                "message": commit_message,
                "content": content_base64
            }
            
            response = requests.put(url, headers=headers, json=data)
            
            if response.status_code == 201:
                print(f"[GITHUB] Uploaded: {github_path}")
                return True
            else:
                print(f"[GITHUB] Failed to upload {github_path}: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[GITHUB] Error uploading {file_path}: {e}")
            return False
    
    def create_training_summary(self, training_files):
        """Create training data summary"""
        if not training_files:
            return None
        
        # Group by class
        class_counts = {}
        total_size = 0
        
        for file_info in training_files:
            class_name = file_info['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_size += file_info['size']
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(training_files),
            'total_size_mb': total_size / (1024 * 1024),
            'class_counts': class_counts,
            'files': training_files
        }
        
        return summary
    
    def organize_training_data(self, training_files):
        """Organize training data by class and date"""
        if not self.config.get('auto_organize_training', True):
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        organized_path = os.path.join(self.backup_path, f"organized_{timestamp}")
        os.makedirs(organized_path, exist_ok=True)
        
        for file_info in training_files:
            class_name = file_info['class']
            class_dir = os.path.join(organized_path, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Copy file to organized structure
            filename = os.path.basename(file_info['path'])
            dest_path = os.path.join(class_dir, filename)
            
            try:
                shutil.copy2(file_info['full_path'], dest_path)
                print(f"[ORGANIZE] Copied {file_info['path']} to {class_name}/")
            except Exception as e:
                print(f"[ORGANIZE] Error copying {file_info['path']}: {e}")
    
    def create_github_issue(self, title, body, labels=None):
        """Create GitHub issue for training progress"""
        if not self.github_token:
            return False
        
        url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/issues"
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        data = {
            "title": title,
            "body": body,
            "labels": labels or ["training", "automated"]
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 201:
                issue_data = response.json()
                print(f"[GITHUB] Issue created: #{issue_data['number']} - {title}")
                return issue_data['number']
            else:
                print(f"[GITHUB] Failed to create issue: {response.status_code}")
                return False
        except Exception as e:
            print(f"[GITHUB] Error creating issue: {e}")
            return False
    
    def sync_code_changes(self, changes):
        """Sync code changes to GitHub"""
        if not changes or not self.config.get('auto_commit', True):
            return
        
        print(f"[SYNC] Found {len(changes)} code changes")
        
        for change in changes:
            file_path = os.path.join(self.code_path, change['path'])
            github_path = change['path']
            
            commit_message = f"Auto-sync: {change['type']} {change['path']}"
            
            success = self.upload_file_to_github(file_path, github_path, commit_message)
            
            if success:
                # Backup the file
                backup_file = os.path.join(self.backup_path, change['path'].replace('/', '_'))
                os.makedirs(os.path.dirname(backup_file), exist_ok=True)
                shutil.copy2(file_path, backup_file)
    
    def sync_training_data(self, training_files):
        """Sync training data to GitHub"""
        if not training_files:
            return
        
        print(f"[SYNC] Found {len(training_files)} new training files")
        
        # Organize training data
        self.organize_training_data(training_files)
        
        # Create summary
        summary = self.create_training_summary(training_files)
        
        if summary:
            # Save summary to file
            summary_file = os.path.join(self.backup_path, f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            # Upload summary to GitHub
            github_summary_path = f"training_summaries/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.upload_file_to_github(summary_file, github_summary_path, "Auto-sync: Training data summary")
            
            # Create GitHub issue
            title = f"📊 Training Data Update - {summary['total_files']} new files"
            body = f"""## Training Data Update

**Timestamp**: {summary['timestamp']}
**Total Files**: {summary['total_files']}
**Total Size**: {summary['total_size_mb']:.2f} MB

## Class Distribution
"""
            
            for class_name, count in summary['class_counts'].items():
                body += f"- **{class_name.title()}**: {count} files\n"
            
            body += f"""
## Files Added
"""
            
            for file_info in training_files[:10]:  # Show first 10 files
                body += f"- `{file_info['path']}` ({file_info['size']} bytes)\n"
            
            if len(training_files) > 10:
                body += f"- ... and {len(training_files) - 10} more files\n"
            
            self.create_github_issue(title, body, ["training", "data-update", "automated"])
    
    def run_auto_sync(self):
        """Run automatic synchronization"""
        print("[AUTO-SYNC] Starting automatic synchronization...")
        
        while True:
            try:
                current_time = time.time()
                
                # Check if it's time to sync
                if current_time - self.last_sync_time >= self.config.get('sync_interval_minutes', 5) * 60:
                    print(f"[AUTO-SYNC] Running sync at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Scan for code changes
                    code_changes = self.scan_code_changes()
                    if code_changes:
                        self.sync_code_changes(code_changes)
                    
                    # Scan for training data
                    training_files = self.scan_training_data()
                    if training_files:
                        self.sync_training_data(training_files)
                    
                    self.last_sync_time = current_time
                    
                    if not code_changes and not training_files:
                        print("[AUTO-SYNC] No changes detected")
                
                # Wait before next check
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                print("\n[AUTO-SYNC] Stopped by user")
                break
            except Exception as e:
                print(f"[AUTO-SYNC] Error: {e}")
                time.sleep(60)
    
    def run_once(self):
        """Run synchronization once"""
        print("[AUTO-SYNC] Running one-time synchronization...")
        
        # Scan for code changes
        code_changes = self.scan_code_changes()
        if code_changes:
            print(f"Found {len(code_changes)} code changes")
            self.sync_code_changes(code_changes)
        else:
            print("No code changes detected")
        
        # Scan for training data
        training_files = self.scan_training_data()
        if training_files:
            print(f"Found {len(training_files)} new training files")
            self.sync_training_data(training_files)
        else:
            print("No new training data detected")

def main():
    """Main function"""
    import sys
    
    sync_system = CodeTrainingAutoSync()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'once':
        sync_system.run_once()
    else:
        sync_system.run_auto_sync()

if __name__ == "__main__":
    main()
