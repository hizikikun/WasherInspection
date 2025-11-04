#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cursor Extension for GitHub Integration
- Provides GUI buttons in Cursor
- One-click send/receive
- Status indicators
- Auto-sync controls
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import json
import os
from cursor_integration import CursorGitHubIntegration

class CursorGitHubExtension:
    def __init__(self):
        """
        Initialize Cursor GitHub Extension
        """
        self.integration = CursorGitHubIntegration()
        self.root = None
        self.status_label = None
        self.sync_button = None
        self.auto_sync_var = None
        self.log_text = None
        
        # Status tracking
        self.last_sync_time = time.time()
        self.sync_status = "Stopped"
        
        print("[CURSOR-EXTENSION] GitHub Extension initialized")
    
    def create_gui(self):
        """Create the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Cursor GitHub Integration")
        self.root.geometry("900x600")
        self.root.resizable(True, True)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Cursor GitHub Integration", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Repository info
        repo_frame = ttk.LabelFrame(main_frame, text="Repository", padding="10")
        repo_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        repo_frame.columnconfigure(0, weight=1)
        
        repo_label = ttk.Label(repo_frame, text=f"[REPO] {self.integration.github_repo}", 
                               font=('Arial', 10))
        repo_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_label = ttk.Label(status_frame, text="[STATUS] Disconnected", font=('Arial', 10))
        self.status_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Control buttons
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)
        
        # Send button
        send_button = ttk.Button(control_frame, text="[SEND] Send to GitHub", 
                               command=self.send_to_github, width=30)
        send_button.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E))
        
        # Receive button
        receive_button = ttk.Button(control_frame, text="[RECEIVE] Receive from GitHub", 
                                  command=self.receive_from_github, width=30)
        receive_button.grid(row=0, column=1, padx=(5, 5), sticky=(tk.W, tk.E))
        
        # Sync button
        self.sync_button = ttk.Button(control_frame, text="[AUTO-SYNC] Start Auto-Sync", 
                                    command=self.toggle_auto_sync, width=30)
        self.sync_button.grid(row=0, column=2, padx=(5, 0), sticky=(tk.W, tk.E))
        
        # Auto-sync checkbox
        self.auto_sync_var = tk.BooleanVar(value=self.integration.auto_sync_enabled)
        auto_sync_check = ttk.Checkbutton(control_frame, text="Enable Auto-Sync", 
                                        variable=self.auto_sync_var,
                                        command=self.toggle_auto_sync_setting)
        auto_sync_check.grid(row=1, column=0, columnspan=3, pady=(10, 0))
        
        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="Activity Log", padding="10")
        log_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Log text widget
        self.log_text = scrolledtext.ScrolledText(log_frame, height=18, width=80, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure log frame grid weights
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Settings button
        settings_button = ttk.Button(main_frame, text="[SETTINGS] Settings", 
                                   command=self.open_settings, width=18)
        settings_button.grid(row=5, column=0, pady=(10, 0), sticky=tk.W)
        
        # Exit button
        exit_button = ttk.Button(main_frame, text="[EXIT] Exit", 
                               command=self.exit_application, width=18)
        exit_button.grid(row=5, column=2, pady=(10, 0), sticky=tk.E)
        
        # Handle window close event - don't stop auto-sync
        self.root.protocol("WM_DELETE_WINDOW", self.on_window_close)
        
        # Initial status update
        self.update_status()
        
        # Start status monitoring
        self.start_status_monitor()
        
        print("[GUI] Interface created successfully")
    
    def log_message(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        if self.log_text:
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
        
        print(f"[LOG] {message}")
    
    def update_status(self):
        """Update status display"""
        if not self.integration.github_token:
            status_text = "[STATUS] Not Connected"
            status_color = "red"
        elif self.sync_status == "Running":
            status_text = "[STATUS] Auto-Sync Running"
            status_color = "green"
        else:
            status_text = "[STATUS] Connected"
            status_color = "orange"
        
        if self.status_label:
            self.status_label.config(text=status_text)
    
    def send_to_github(self):
        """Send to GitHub button handler"""
        def send_worker():
            self.log_message("[SEND] Sending to GitHub...")
            
            # Check repository access first (will auto-create if needed)
            self.log_message("[CHECK] Verifying repository access...")
            access_ok, error_msg = self.integration.verify_repository_access()
            if not access_ok:
                self.log_message(f"[ERROR] Repository access failed: {error_msg}")
                self.show_notification(f"[ERROR] {error_msg}", show_popup=False)
                return
            else:
                if "created" in error_msg.lower():
                    self.log_message(f"[SUCCESS] {error_msg}")
                else:
                    self.log_message(f"[INFO] {error_msg}")
            
            success = self.integration.one_click_send("Manual send from Cursor")
            
            if success:
                self.log_message("[SUCCESS] Successfully sent to GitHub")
                self.show_notification("[SUCCESS] Successfully sent to GitHub", show_popup=True)
            else:
                self.log_message("[ERROR] Failed to send to GitHub (check console for details)")
                self.show_notification("[ERROR] Failed to send to GitHub", show_popup=False)
        
        # Run in background thread
        thread = threading.Thread(target=send_worker)
        thread.daemon = True
        thread.start()
    
    def receive_from_github(self):
        """Receive from GitHub button handler"""
        def receive_worker():
            self.log_message("[RECEIVE] Receiving from GitHub...")
            success = self.integration.one_click_receive()
            
            if success:
                self.log_message("[SUCCESS] Successfully received from GitHub")
                self.show_notification("[SUCCESS] Successfully received from GitHub", show_popup=True)
            else:
                self.log_message("[ERROR] Failed to receive from GitHub")
                self.show_notification("[ERROR] Failed to receive from GitHub", show_popup=False)
        
        # Run in background thread
        thread = threading.Thread(target=receive_worker)
        thread.daemon = True
        thread.start()
    
    def toggle_auto_sync(self):
        """Toggle auto-sync button handler"""
        if self.sync_status == "Running":
            self.integration.stop_auto_sync()
            self.sync_status = "Stopped"
            self.sync_button.config(text="[AUTO-SYNC] Start Auto-Sync")
            self.log_message("[STOP] Auto-sync stopped")
            self.show_notification("[STOP] Auto-sync stopped", show_popup=False)
        else:
            if not self.integration.github_token:
                messagebox.showerror("Error", "GitHub token not configured!")
                return
            
            self.integration.start_auto_sync()
            self.sync_status = "Running"
            self.sync_button.config(text="[STOP] Stop Auto-Sync")
            self.log_message("[START] Auto-sync started")
            self.show_notification("[START] Auto-sync started", show_popup=False)
        
        self.update_status()
    
    def toggle_auto_sync_setting(self):
        """Toggle auto-sync setting"""
        self.integration.auto_sync_enabled = self.auto_sync_var.get()
        self.integration.config['auto_sync_enabled'] = self.auto_sync_var.get()
        self.integration.save_config()
        
        if self.auto_sync_var.get():
            self.log_message("[SETTING] Auto-sync enabled in settings")
        else:
            self.log_message("[SETTING] Auto-sync disabled in settings")
    
    def open_settings(self):
        """Open settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.resizable(False, False)
        
        # Make window modal
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Settings frame
        settings_frame = ttk.Frame(settings_window, padding="20")
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Repository settings
        repo_frame = ttk.LabelFrame(settings_frame, text="Repository", padding="10")
        repo_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(repo_frame, text="Repository:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        repo_entry = ttk.Entry(repo_frame, width=20)
        repo_entry.insert(0, self.integration.github_repo)
        repo_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Sync settings
        sync_frame = ttk.LabelFrame(settings_frame, text="Sync Settings", padding="10")
        sync_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(sync_frame, text="Sync Interval (seconds):").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        interval_entry = ttk.Entry(sync_frame, width=10)
        interval_entry.insert(0, str(self.integration.sync_interval))
        interval_entry.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(sync_frame, text="Auto-commit Threshold:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        threshold_entry = ttk.Entry(sync_frame, width=10)
        threshold_entry.insert(0, str(self.integration.auto_commit_threshold))
        threshold_entry.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        # Token settings
        token_frame = ttk.LabelFrame(settings_frame, text="Authentication", padding="10")
        token_frame.pack(fill=tk.X, pady=(0, 10))
        
        token_status = "✅ Configured" if self.integration.github_token else "❌ Not configured"
        ttk.Label(token_frame, text=f"Token Status: {token_status}").pack(anchor=tk.W)
        
        if not self.integration.github_token:
            token_button = ttk.Button(token_frame, text="Setup Token", 
                                   command=self.setup_token)
            token_button.pack(anchor=tk.W, pady=(10, 0))
        
        # Save button
        def save_settings():
            # Update settings
            self.integration.github_repo = repo_entry.get()
            self.integration.sync_interval = int(interval_entry.get())
            self.integration.auto_commit_threshold = int(threshold_entry.get())
            
            # Save to config
            self.integration.config['github_repo'] = self.integration.github_repo
            self.integration.config['sync_interval'] = self.integration.sync_interval
            self.integration.config['auto_commit_threshold'] = self.integration.auto_commit_threshold
            
            self.integration.save_config()
            
            self.log_message("[SAVE] Settings saved")
            settings_window.destroy()
        
        save_button = ttk.Button(settings_frame, text="[SAVE] Save Settings", 
                               command=save_settings)
        save_button.pack(pady=(10, 0))
    
    def setup_token(self):
        """Setup GitHub token"""
        self.integration.setup_github_auth()
    
    def show_notification(self, message, show_popup=False):
        """Show notification"""
        if self.root:
            # Show in log
            self.log_message(message)
            
            # Show popup only if explicitly requested (for important notifications)
            # Errors are logged but don't show popup to avoid interrupting automatic sync
            if show_popup:
                try:
                    messagebox.showinfo("Cursor GitHub Sync", message)
                except:
                    pass
    
    def start_status_monitor(self):
        """Start status monitoring thread"""
        def status_worker():
            while True:
                try:
                    self.update_status()
                    time.sleep(5)  # Update every 5 seconds
                except:
                    break
        
        thread = threading.Thread(target=status_worker)
        thread.daemon = True
        thread.start()
    
    def on_window_close(self):
        """Handle window close event"""
        # Don't stop auto-sync when closing window
        # Auto-sync will continue in background
        print("[INFO] GUI closed, auto-sync continues in background")
        print("[INFO] Auto-sync is still running. To stop auto-sync, use the [STOP] button before closing.")
        self.root.quit()
        self.root.destroy()
    
    def exit_application(self):
        """Exit application - with confirmation"""
        # Ask user if they want to stop auto-sync
        if self.sync_status == "Running":
            response = messagebox.askyesno(
                "Confirm Exit",
                "Auto-sync is running. Do you want to stop auto-sync and exit?\n\n"
                "Yes: Stop auto-sync and exit\n"
                "No: Exit GUI but keep auto-sync running"
            )
            if response:
                self.integration.stop_auto_sync()
                self.sync_status = "Stopped"
                self.log_message("[STOP] Auto-sync stopped on exit")
        
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Run the extension"""
        self.create_gui()
        
        # Initial log message
        self.log_message("[START] Cursor GitHub Extension started")
        
        # Start auto-sync if enabled
        if self.integration.auto_sync_enabled and self.integration.github_token:
            self.toggle_auto_sync()
        
        # Run GUI
        self.root.mainloop()

def main():
    """Main function"""
    extension = CursorGitHubExtension()
    extension.run()

if __name__ == "__main__":
    main()
