#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Code and Training Data Auto-Sync System
- Monitors code changes and automatically commits to GitHub
- Monitors training data and automatically uploads new photos
- Generates comprehensive reports
- Organizes data by class and date
"""

import os
import json
import time
import threading
from datetime import datetime
from auto_sync import CodeTrainingAutoSync
from training_data_manager import TrainingDataManager

class IntegratedAutoSync:
    def __init__(self):
        """
        Initialize the integrated auto-sync system
        """
        self.code_sync = CodeTrainingAutoSync()
        self.training_manager = TrainingDataManager()
        self.running = False
        
        print("[INTEGRATED] Integrated Auto-Sync System initialized")
        print("[INTEGRATED] Monitoring code changes and training data")
    
    def run_code_sync(self):
        """Run code synchronization in background thread"""
        try:
            self.code_sync.run_auto_sync()
        except Exception as e:
            print(f"[CODE-SYNC] Error: {e}")
    
    def run_training_monitor(self):
        """Run training data monitoring in background thread"""
        try:
            while self.running:
                # Scan for new training data
                new_files = self.training_manager.scan_training_data()
                
                if new_files:
                    print(f"[TRAINING] Found {len(new_files)} new training files")
                    
                    # Generate report
                    report = self.training_manager.generate_training_report()
                    
                    # Export statistics
                    self.training_manager.export_statistics()
                    
                    # Create GitHub issue
                    self.create_training_issue(report)
                
                # Wait before next scan
                time.sleep(300)  # Check every 5 minutes
                
        except Exception as e:
            print(f"[TRAINING-MONITOR] Error: {e}")
    
    def create_training_issue(self, report):
        """Create GitHub issue for training data update"""
        title = f"📊 Training Data Update - {report['summary']['total_files']} files"
        
        body = f"""## Training Data Update

**Timestamp**: {report['timestamp']}
**Total Files**: {report['summary']['total_files']}
**Total Size**: {report['summary']['total_size_mb']:.2f} MB
**Classes**: {report['summary']['classes']}

## Class Distribution
"""
        
        for class_name, count in report['class_distribution'].items():
            size_mb = report['class_sizes_mb'][class_name]
            body += f"- **{class_name.title()}**: {count} files ({size_mb:.2f} MB)\n"
        
        body += f"""
## Daily Activity
"""
        
        # Show last 7 days
        sorted_dates = sorted(report['daily_counts'].items(), reverse=True)[:7]
        for date, count in sorted_dates:
            body += f"- **{date}**: {count} files\n"
        
        body += f"""
## Recommendations
"""
        
        for rec in report['recommendations']:
            body += f"- {rec}\n"
        
        # Create issue
        self.code_sync.create_github_issue(title, body, ["training", "data-update", "automated"])
    
    def run(self):
        """Run the integrated system"""
        print("[INTEGRATED] Starting integrated auto-sync system...")
        print("[INTEGRATED] Press Ctrl+C to stop")
        
        self.running = True
        
        # Start code sync in background thread
        code_thread = threading.Thread(target=self.run_code_sync)
        code_thread.daemon = True
        code_thread.start()
        
        # Start training monitor in background thread
        training_thread = threading.Thread(target=self.run_training_monitor)
        training_thread.daemon = True
        training_thread.start()
        
        try:
            # Main loop
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n[INTEGRATED] Stopping system...")
            self.running = False
            
            # Wait for threads to finish
            code_thread.join(timeout=5)
            training_thread.join(timeout=5)
            
            print("[INTEGRATED] System stopped")
    
    def run_once(self):
        """Run synchronization once"""
        print("[INTEGRATED] Running one-time synchronization...")
        
        # Run code sync once
        self.code_sync.run_once()
        
        # Run training data scan
        self.training_manager.scan_training_data()
        report = self.training_manager.generate_training_report()
        self.training_manager.export_statistics()
        
        print("\n[INTEGRATED] Synchronization complete")
        print(f"Training data: {report['summary']['total_files']} files, {report['summary']['total_size_mb']:.2f} MB")

def main():
    """Main function"""
    import sys
    
    system = IntegratedAutoSync()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'once':
        system.run_once()
    else:
        system.run()

if __name__ == "__main__":
    main()
