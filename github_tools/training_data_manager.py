#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Data Management System
- Automatically organizes training photos by class and date
- Generates training progress reports
- Manages dataset statistics
"""

import os
import json
import shutil
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

class TrainingDataManager:
    def __init__(self, training_data_path="cs_AItraining_data"):
        """
        Initialize the training data management system
        """
        self.training_data_path = training_data_path
        self.classes = ['good', 'chipping', 'black_spot', 'scratch']
        self.stats = {}
        
        print(f"[TRAINING] Training data path: {self.training_data_path}")
        self.scan_training_data()
    
    def scan_training_data(self):
        """Scan and analyze training data"""
        self.stats = {
            'total_files': 0,
            'total_size': 0,
            'class_counts': defaultdict(int),
            'class_sizes': defaultdict(int),
            'files_by_class': defaultdict(list),
            'date_groups': defaultdict(int)
        }
        
        if not os.path.exists(self.training_data_path):
            print(f"[TRAINING] Training data path does not exist: {self.training_data_path}")
            return
        
        for root, dirs, files in os.walk(self.training_data_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.training_data_path)
                    
                    # Extract class from path
                    class_name = self.extract_class_from_path(file_path)
                    
                    # Get file info
                    file_size = os.path.getsize(file_path)
                    file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
                    date_key = file_date.strftime('%Y-%m-%d')
                    
                    # Update statistics
                    self.stats['total_files'] += 1
                    self.stats['total_size'] += file_size
                    self.stats['class_counts'][class_name] += 1
                    self.stats['class_sizes'][class_name] += file_size
                    self.stats['files_by_class'][class_name].append({
                        'path': relative_path,
                        'size': file_size,
                        'date': file_date.isoformat()
                    })
                    self.stats['date_groups'][date_key] += 1
        
        print(f"[TRAINING] Scanned {self.stats['total_files']} training files")
        for class_name, count in self.stats['class_counts'].items():
            size_mb = self.stats['class_sizes'][class_name] / (1024 * 1024)
            print(f"  {class_name}: {count} files ({size_mb:.2f} MB)")
    
    def extract_class_from_path(self, file_path):
        """Extract class name from file path"""
        path_parts = Path(file_path).parts
        
        for part in path_parts:
            if part.lower() in self.classes:
                return part.lower()
        
        return 'unknown'
    
    def organize_by_date(self, output_path="organized_training_data"):
        """Organize training data by date"""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        print(f"[ORGANIZE] Organizing training data by date...")
        
        for class_name, files in self.stats['files_by_class'].items():
            class_dir = os.path.join(output_path, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for file_info in files:
                source_path = os.path.join(self.training_data_path, file_info['path'])
                filename = os.path.basename(file_info['path'])
                
                # Add date prefix to filename
                file_date = datetime.fromisoformat(file_info['date'])
                date_prefix = file_date.strftime('%Y%m%d_')
                new_filename = date_prefix + filename
                
                dest_path = os.path.join(class_dir, new_filename)
                
                try:
                    shutil.copy2(source_path, dest_path)
                    print(f"[ORGANIZE] Copied {file_info['path']} -> {class_name}/{new_filename}")
                except Exception as e:
                    print(f"[ORGANIZE] Error copying {file_info['path']}: {e}")
    
    def generate_training_report(self):
        """Generate training data report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files': self.stats['total_files'],
                'total_size_mb': self.stats['total_size'] / (1024 * 1024),
                'classes': len(self.stats['class_counts']),
                'date_range': {
                    'earliest': min(self.stats['date_groups'].keys()) if self.stats['date_groups'] else None,
                    'latest': max(self.stats['date_groups'].keys()) if self.stats['date_groups'] else None
                }
            },
            'class_distribution': dict(self.stats['class_counts']),
            'class_sizes_mb': {k: v / (1024 * 1024) for k, v in self.stats['class_sizes'].items()},
            'daily_counts': dict(self.stats['date_groups']),
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def generate_recommendations(self):
        """Generate training recommendations"""
        recommendations = []
        
        if not self.stats['class_counts']:
            return ["No training data found"]
        
        # Check class balance
        counts = list(self.stats['class_counts'].values())
        max_count = max(counts)
        min_count = min(counts)
        
        if max_count > min_count * 2:
            recommendations.append("Class imbalance detected - consider collecting more data for minority classes")
        
        # Check total data size
        total_mb = self.stats['total_size'] / (1024 * 1024)
        if total_mb < 100:
            recommendations.append("Consider collecting more training data (currently < 100MB)")
        elif total_mb > 1000:
            recommendations.append("Large dataset detected - consider data augmentation techniques")
        
        # Check recent activity
        recent_dates = [d for d in self.stats['date_groups'].keys() 
                       if datetime.fromisoformat(d) > datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)]
        
        if not recent_dates:
            recommendations.append("No training data added today - consider collecting new samples")
        
        return recommendations
    
    def create_balanced_dataset(self, output_path="balanced_dataset", target_count=None):
        """Create a balanced dataset"""
        if not target_count:
            # Use the median count as target
            counts = list(self.stats['class_counts'].values())
            target_count = sorted(counts)[len(counts) // 2]
        
        print(f"[BALANCE] Creating balanced dataset with {target_count} samples per class...")
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for class_name in self.classes:
            if class_name not in self.stats['files_by_class']:
                continue
            
            class_dir = os.path.join(output_path, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            files = self.stats['files_by_class'][class_name]
            
            # If we have more files than target, sample randomly
            if len(files) > target_count:
                import random
                selected_files = random.sample(files, target_count)
            else:
                selected_files = files
            
            for file_info in selected_files:
                source_path = os.path.join(self.training_data_path, file_info['path'])
                filename = os.path.basename(file_info['path'])
                dest_path = os.path.join(class_dir, filename)
                
                try:
                    shutil.copy2(source_path, dest_path)
                    print(f"[BALANCE] Copied {file_info['path']} -> {class_name}/{filename}")
                except Exception as e:
                    print(f"[BALANCE] Error copying {file_info['path']}: {e}")
    
    def validate_images(self):
        """Validate training images"""
        print("[VALIDATE] Validating training images...")
        
        invalid_files = []
        corrupted_files = []
        
        for class_name, files in self.stats['files_by_class'].items():
            for file_info in files:
                file_path = os.path.join(self.training_data_path, file_info['path'])
                
                try:
                    # Try to load image
                    img = cv2.imread(file_path)
                    if img is None:
                        invalid_files.append(file_info['path'])
                    else:
                        # Check image properties
                        height, width = img.shape[:2]
                        if height < 32 or width < 32:
                            invalid_files.append(file_info['path'])
                        elif height > 4000 or width > 4000:
                            invalid_files.append(file_info['path'])
                
                except Exception as e:
                    corrupted_files.append(file_info['path'])
        
        print(f"[VALIDATE] Found {len(invalid_files)} invalid files")
        print(f"[VALIDATE] Found {len(corrupted_files)} corrupted files")
        
        return {
            'invalid_files': invalid_files,
            'corrupted_files': corrupted_files
        }
    
    def export_statistics(self, output_file="training_statistics.json"):
        """Export training statistics to JSON"""
        report = self.generate_training_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"[EXPORT] Statistics exported to {output_file}")
        return report

def main():
    """Main function"""
    import sys
    
    manager = TrainingDataManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'scan':
            manager.scan_training_data()
            print("\nTraining Data Summary:")
            for class_name, count in manager.stats['class_counts'].items():
                size_mb = manager.stats['class_sizes'][class_name] / (1024 * 1024)
                print(f"  {class_name}: {count} files ({size_mb:.2f} MB)")
        
        elif command == 'organize':
            manager.organize_by_date()
        
        elif command == 'balance':
            target_count = int(sys.argv[2]) if len(sys.argv) > 2 else None
            manager.create_balanced_dataset(target_count=target_count)
        
        elif command == 'validate':
            manager.validate_images()
        
        elif command == 'report':
            report = manager.export_statistics()
            print("\nTraining Report:")
            print(f"Total files: {report['summary']['total_files']}")
            print(f"Total size: {report['summary']['total_size_mb']:.2f} MB")
            print(f"Classes: {report['summary']['classes']}")
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        else:
            print("Available commands: scan, organize, balance, validate, report")
    else:
        # Interactive mode
        while True:
            print("\nTraining Data Manager")
            print("1. Scan training data")
            print("2. Organize by date")
            print("3. Create balanced dataset")
            print("4. Validate images")
            print("5. Generate report")
            print("6. Exit")
            
            choice = input("Select option (1-6): ").strip()
            
            if choice == '1':
                manager.scan_training_data()
            elif choice == '2':
                manager.organize_by_date()
            elif choice == '3':
                target = input("Target count per class (press Enter for auto): ").strip()
                target_count = int(target) if target else None
                manager.create_balanced_dataset(target_count=target_count)
            elif choice == '4':
                manager.validate_images()
            elif choice == '5':
                manager.export_statistics()
            elif choice == '6':
                break
            else:
                print("Invalid choice")

if __name__ == "__main__":
    main()
