#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Change History Viewer
- Monitors file changes and displays Git commit history
- Shows file diffs and commit details
- Auto-refreshes when changes are detected
"""

import os
import sys
import subprocess
import json
from datetime import datetime, timedelta
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time

class ChangeHistoryViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Change History Viewer - WasherInspection")
        self.root.geometry("1200x800")
        
        # Project path
        self.project_path = Path(__file__).parent.absolute()
        self.last_commit_hash = None
        self.state_file = self.project_path / ".history_viewer_state.json"
        self.config_file = self.project_path / ".history_viewer_config.json"
        self.last_check_time = self.load_last_check_time()
        
        # Auto-commit settings
        self.auto_commit_enabled = self.load_auto_commit_setting()
        self.auto_commit_interval = 60  # seconds
        self.last_auto_commit_check = datetime.now()
        
        # Setup UI
        self.setup_ui()
        
        # Start monitoring
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_changes, daemon=True)
        self.monitor_thread.start()
        
        # Start auto-commit monitoring if enabled
        if self.auto_commit_enabled:
            self.auto_commit_thread = threading.Thread(target=self.auto_commit_monitor, daemon=True)
            self.auto_commit_thread.start()
        
        # Initial load and check for missed changes
        self.check_missed_changes()
        self.refresh_history()
        
        # Save state on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_ui(self):
        """Setup user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Change History Viewer", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Status: Monitoring...", 
                                      foreground="green")
        self.status_label.pack(side=tk.LEFT)
        
        self.last_update_label = ttk.Label(status_frame, text="", foreground="gray")
        self.last_update_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W), pady=(0, 10))
        
        refresh_btn = ttk.Button(button_frame, text="Refresh", command=self.refresh_history)
        refresh_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        view_diff_btn = ttk.Button(button_frame, text="View Diff", command=self.view_diff)
        view_diff_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        view_files_btn = ttk.Button(button_frame, text="View Changed Files", command=self.view_changed_files)
        view_files_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        open_github_btn = ttk.Button(button_frame, text="Open GitHub", command=self.open_github)
        open_github_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        auto_commit_btn = ttk.Button(button_frame, text="Auto Commit & Push", command=self.manual_commit_push)
        auto_commit_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        test_commit_btn = ttk.Button(button_frame, text="Test Commit (UTF-8)", command=self.create_test_commit)
        test_commit_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        fix_garbled_btn = ttk.Button(button_frame, text="Fix Garbled Commits", command=self.fix_garbled_commits)
        fix_garbled_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Auto-commit toggle
        self.auto_commit_var = tk.BooleanVar(value=self.auto_commit_enabled)
        auto_commit_check = ttk.Checkbutton(button_frame, text="Enable Auto Commit", 
                                            variable=self.auto_commit_var,
                                            command=self.toggle_auto_commit)
        auto_commit_check.pack(side=tk.LEFT)
        
        # Commit list
        list_frame = ttk.LabelFrame(main_frame, text="Recent Commits", padding="5")
        list_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Treeview for commits
        columns = ("Date", "Author", "Message", "Hash")
        self.commit_tree = ttk.Treeview(list_frame, columns=columns, show="tree headings", height=10)
        self.commit_tree.heading("#0", text="#")
        self.commit_tree.heading("Date", text="Date")
        self.commit_tree.heading("Author", text="Author")
        self.commit_tree.heading("Message", text="Message")
        self.commit_tree.heading("Hash", text="Hash")
        
        self.commit_tree.column("#0", width=50)
        self.commit_tree.column("Date", width=150)
        self.commit_tree.column("Author", width=150)
        self.commit_tree.column("Message", width=400)
        self.commit_tree.column("Hash", width=100)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.commit_tree.yview)
        self.commit_tree.configure(yscrollcommand=scrollbar.set)
        
        self.commit_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Bind selection
        self.commit_tree.bind("<<TreeviewSelect>>", self.on_commit_select)
        
        # Details frame
        details_frame = ttk.LabelFrame(main_frame, text="Commit Details", padding="5")
        details_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        
        self.details_text = scrolledtext.ScrolledText(details_frame, height=15, wrap=tk.WORD)
        self.details_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def get_git_commits(self, limit=20):
        """Get recent Git commits"""
        try:
            os.chdir(self.project_path)
            cmd = [
                "git", "log",
                "--pretty=format:%H|%an|%ae|%ad|%s",
                "--date=format:%Y-%m-%d %H:%M:%S",
                f"-{limit}"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode != 0:
                return []
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split('|', 4)
                if len(parts) >= 5:
                    commits.append({
                        'hash': parts[0][:8],
                        'full_hash': parts[0],
                        'author': parts[1],
                        'email': parts[2],
                        'date': parts[3],
                        'message': parts[4]
                    })
            
            return commits
        except Exception as e:
            print(f"Error getting commits: {e}")
            return []
    
    def get_commits_since(self, since_datetime):
        """Get commits since a specific datetime"""
        try:
            os.chdir(self.project_path)
            since_str = since_datetime.strftime("%Y-%m-%d %H:%M:%S")
            cmd = [
                "git", "log",
                "--pretty=format:%H|%an|%ae|%ad|%s",
                "--date=format:%Y-%m-%d %H:%M:%S",
                f"--since=\"{since_str}\"",
                "--all"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode != 0:
                return []
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split('|', 4)
                if len(parts) >= 5:
                    commits.append({
                        'hash': parts[0][:8],
                        'full_hash': parts[0],
                        'author': parts[1],
                        'email': parts[2],
                        'date': parts[3],
                        'message': parts[4]
                    })
            
            return commits
        except Exception as e:
            print(f"Error getting commits since: {e}")
            return []
    
    def load_last_check_time(self):
        """Load last check time from state file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    if 'last_check_time' in state:
                        return datetime.fromisoformat(state['last_check_time'])
        except Exception as e:
            print(f"Error loading state: {e}")
        return datetime.now()
    
    def save_last_check_time(self):
        """Save current time as last check time"""
        try:
            state = {
                'last_check_time': datetime.now().isoformat(),
                'last_commit_hash': self.last_commit_hash
            }
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def check_missed_changes(self):
        """Check for changes that occurred while app was closed"""
        if not self.last_check_time:
            return
        
        # Get commits since last check
        missed_commits = self.get_commits_since(self.last_check_time)
        
        if missed_commits:
            time_diff = datetime.now() - self.last_check_time
            hours = time_diff.total_seconds() / 3600
            
            if hours < 1:
                time_str = f"{int(time_diff.total_seconds() / 60)}分"
            elif hours < 24:
                time_str = f"{int(hours)}時間"
            else:
                days = int(hours / 24)
                time_str = f"{days}日"
            
            message = f"アプリが閉じていた間に {len(missed_commits)} 件の新しいコミットを検出しました。\n"
            message += f"（約{time_str}間の変更）\n\n"
            message += "最近の変更:\n"
            for commit in missed_commits[:5]:
                message += f"  • {commit['date']}: {commit['message'][:50]}\n"
            
            if len(missed_commits) > 5:
                message += f"  ... 他 {len(missed_commits) - 5} 件\n"
            
            messagebox.showinfo("新しい変更を検出 / New Changes Detected", message)
    
    def on_closing(self):
        """Handle window closing"""
        self.save_last_check_time()
        self.monitoring = False
        self.root.destroy()
    
    def get_commit_details(self, commit_hash):
        """Get detailed information about a commit"""
        try:
            os.chdir(self.project_path)
            
            # Get commit info
            cmd = ["git", "show", "--stat", "--format=full", commit_hash]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                return result.stdout
            return None
        except Exception as e:
            print(f"Error getting commit details: {e}")
            return None
    
    def get_changed_files(self, commit_hash):
        """Get list of changed files in a commit"""
        try:
            os.chdir(self.project_path)
            cmd = ["git", "show", "--name-status", "--pretty=format:", commit_hash]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                return result.stdout.strip().split('\n')
            return []
        except Exception as e:
            print(f"Error getting changed files: {e}")
            return []
    
    def get_current_changes(self):
        """Get current uncommitted changes"""
        try:
            os.chdir(self.project_path)
            cmd = ["git", "status", "--porcelain"]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                return result.stdout.strip().split('\n') if result.stdout.strip() else []
            return []
        except Exception as e:
            print(f"Error getting current changes: {e}")
            return []
    
    def refresh_history(self):
        """Refresh commit history"""
        commits = self.get_git_commits()
        
        # Get missed commits (since last check)
        missed_hashes = set()
        if self.last_check_time:
            missed_commits = self.get_commits_since(self.last_check_time)
            missed_hashes = {c['full_hash'] for c in missed_commits}
        
        # Clear existing items
        for item in self.commit_tree.get_children():
            self.commit_tree.delete(item)
        
        # Configure tags for highlighting missed commits
        self.commit_tree.tag_configure("missed", background="#fff3cd", foreground="#856404")
        
        # Add commits
        for idx, commit in enumerate(commits, 1):
            tags = (commit['full_hash'],)
            if commit['full_hash'] in missed_hashes:
                tags = (commit['full_hash'], "missed")
            
            self.commit_tree.insert("", "end", 
                                   text=str(idx),
                                   values=(commit['date'], commit['author'], 
                                          commit['message'], commit['hash']),
                                   tags=tags)
        
        # Check for new commits
        if commits:
            latest_hash = commits[0]['full_hash']
            if self.last_commit_hash and self.last_commit_hash != latest_hash:
                self.show_new_commit_notification(commits[0])
            self.last_commit_hash = latest_hash
        
        # Update status
        self.last_update_label.config(text=f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check for uncommitted changes
        uncommitted = self.get_current_changes()
        if uncommitted:
            self.status_label.config(text=f"Status: {len(uncommitted)} uncommitted changes", 
                                   foreground="orange")
        else:
            missed_count = len(missed_hashes)
            if missed_count > 0:
                self.status_label.config(text=f"Status: {missed_count} missed commits (highlighted)", 
                                       foreground="blue")
            else:
                self.status_label.config(text="Status: No uncommitted changes", 
                                       foreground="green")
        
        # Save state after refresh
        self.save_last_check_time()
    
    def on_commit_select(self, event):
        """Handle commit selection"""
        selection = self.commit_tree.selection()
        if not selection:
            return
        
        item = self.commit_tree.item(selection[0])
        tags = item['tags']
        if tags:
            commit_hash = tags[0]
            details = self.get_commit_details(commit_hash)
            if details:
                self.details_text.delete(1.0, tk.END)
                self.details_text.insert(1.0, details)
    
    def view_diff(self):
        """View diff of selected commit"""
        selection = self.commit_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a commit first")
            return
        
        item = self.commit_tree.item(selection[0])
        tags = item['tags']
        if tags:
            commit_hash = tags[0]
            self.show_diff_window(commit_hash)
    
    def analyze_diff(self, diff_output):
        """Analyze diff and create Japanese description"""
        lines = diff_output.split('\n')
        summary = {
            'files_changed': 0,
            'insertions': 0,
            'deletions': 0,
            'files': []
        }
        
        current_file = None
        file_stats = {'additions': 0, 'deletions': 0}
        
        for line in lines:
            # File header
            if line.startswith('diff --git'):
                if current_file:
                    summary['files'].append({
                        'name': current_file,
                        'additions': file_stats['additions'],
                        'deletions': file_stats['deletions']
                    })
                summary['files_changed'] += 1
                # Extract filename
                parts = line.split()
                if len(parts) >= 4:
                    current_file = parts[3].replace('b/', '')
                    file_stats = {'additions': 0, 'deletions': 0}
            # Stats line
            elif line.startswith('+') and not line.startswith('+++'):
                summary['insertions'] += 1
                file_stats['additions'] += 1
            elif line.startswith('-') and not line.startswith('---'):
                summary['deletions'] += 1
                file_stats['deletions'] += 1
        
        # Add last file
        if current_file:
            summary['files'].append({
                'name': current_file,
                'additions': file_stats['additions'],
                'deletions': file_stats['deletions']
            })
        
        return summary
    
    def format_diff_with_japanese(self, diff_output, summary):
        """Format diff output with Japanese descriptions"""
        japanese_header = f"""
{'='*80}
【変更の概要 / Change Summary】
{'='*80}
変更されたファイル数 / Files Changed: {summary['files_changed']}
追加された行数 / Lines Added: +{summary['insertions']}
削除された行数 / Lines Deleted: -{summary['deletions']}

【変更されたファイル一覧 / Changed Files】
{'='*80}
"""
        
        for file_info in summary['files']:
            japanese_header += f"  • {file_info['name']}\n"
            japanese_header += f"    (追加: +{file_info['additions']}行, 削除: -{file_info['deletions']}行)\n"
        
        japanese_header += f"\n{'='*80}\n【詳細な変更内容 / Detailed Changes】\n{'='*80}\n\n"
        
        return japanese_header + diff_output
    
    def show_diff_window(self, commit_hash):
        """Show diff in a new window with Japanese descriptions"""
        diff_window = tk.Toplevel(self.root)
        diff_window.title(f"Diff - {commit_hash[:8]}")
        diff_window.geometry("1200x700")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(diff_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Diff with Japanese description
        diff_frame = ttk.Frame(notebook)
        notebook.add(diff_frame, text="変更内容 (日本語説明付き)")
        
        diff_text = scrolledtext.ScrolledText(diff_frame, wrap=tk.NONE, font=("Courier", 9))
        diff_text.pack(fill=tk.BOTH, expand=True)
        
        # Tab 2: Original diff only
        original_frame = ttk.Frame(notebook)
        notebook.add(original_frame, text="元のDiff (Original)")
        
        original_text = scrolledtext.ScrolledText(original_frame, wrap=tk.NONE, font=("Courier", 9))
        original_text.pack(fill=tk.BOTH, expand=True)
        
        try:
            os.chdir(self.project_path)
            cmd = ["git", "show", commit_hash]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                diff_output = result.stdout
                
                # Analyze diff
                summary = self.analyze_diff(diff_output)
                
                # Format with Japanese
                formatted_diff = self.format_diff_with_japanese(diff_output, summary)
                
                # Insert into tabs
                diff_text.insert(1.0, formatted_diff)
                diff_text.config(state=tk.DISABLED)
                
                original_text.insert(1.0, diff_output)
                original_text.config(state=tk.DISABLED)
            else:
                error_msg = f"Error: Could not get diff for commit {commit_hash[:8]}"
                diff_text.insert(1.0, error_msg)
                original_text.insert(1.0, error_msg)
        except Exception as e:
            error_msg = f"Error: {e}"
            diff_text.insert(1.0, error_msg)
            original_text.insert(1.0, error_msg)
    
    def view_changed_files(self):
        """View changed files in selected commit"""
        selection = self.commit_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a commit first")
            return
        
        item = self.commit_tree.item(selection[0])
        tags = item['tags']
        if tags:
            commit_hash = tags[0]
            files = self.get_changed_files(commit_hash)
            
            files_window = tk.Toplevel(self.root)
            files_window.title(f"Changed Files - {commit_hash[:8]}")
            files_window.geometry("600x400")
            
            files_text = scrolledtext.ScrolledText(files_window, wrap=tk.WORD)
            files_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            for file_info in files:
                files_text.insert(tk.END, file_info + "\n")
            
            files_text.config(state=tk.DISABLED)
    
    def open_github(self):
        """Open GitHub repository in browser"""
        try:
            os.chdir(self.project_path)
            cmd = ["git", "remote", "get-url", "origin"]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                url = result.stdout.strip()
                # Convert SSH to HTTPS if needed
                if url.startswith("git@"):
                    url = url.replace("git@github.com:", "https://github.com/").replace(".git", "")
                
                import webbrowser
                webbrowser.open(url)
            else:
                messagebox.showwarning("Warning", "Could not find GitHub remote URL")
        except Exception as e:
            messagebox.showerror("Error", f"Error opening GitHub: {e}")
    
    def load_auto_commit_setting(self):
        """Load auto-commit setting from config file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('auto_commit_enabled', False)
        except Exception as e:
            print(f"Error loading config: {e}")
        return False
    
    def save_auto_commit_setting(self):
        """Save auto-commit setting to config file"""
        try:
            config = {
                'auto_commit_enabled': self.auto_commit_enabled,
                'auto_commit_interval': self.auto_commit_interval
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def toggle_auto_commit(self):
        """Toggle auto-commit on/off"""
        self.auto_commit_enabled = self.auto_commit_var.get()
        self.save_auto_commit_setting()
        
        if self.auto_commit_enabled:
            if not hasattr(self, 'auto_commit_thread') or not self.auto_commit_thread.is_alive():
                self.auto_commit_thread = threading.Thread(target=self.auto_commit_monitor, daemon=True)
                self.auto_commit_thread.start()
            self.status_label.config(text="Status: Auto-commit enabled", foreground="blue")
        else:
            self.status_label.config(text="Status: Auto-commit disabled", foreground="gray")
    
    def commit_and_push(self, message=None):
        """Commit and push changes"""
        try:
            os.chdir(self.project_path)
            
            # Check for changes
            cmd = ["git", "status", "--porcelain"]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if not result.stdout.strip():
                return False, "No changes to commit"
            
            # Add all changes
            cmd = ["git", "add", "-A"]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode != 0:
                return False, f"Error adding files: {result.stderr}"
            
            # Create commit message if not provided
            if not message:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                message = f"Auto-commit: {timestamp}"
            
            # Commit
            cmd = ["git", "commit", "-m", message]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode != 0:
                return False, f"Error committing: {result.stderr}"
            
            # Push
            cmd = ["git", "push", "origin", "master"]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode != 0:
                # Try 'main' branch if 'master' fails
                cmd = ["git", "push", "origin", "main"]
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                
                if result.returncode != 0:
                    return False, f"Error pushing: {result.stderr}"
            
            return True, "Successfully committed and pushed"
            
        except Exception as e:
            return False, f"Error: {e}"
    
    def manual_commit_push(self):
        """Manually trigger commit and push"""
        success, message = self.commit_and_push()
        
        if success:
            messagebox.showinfo("Success", message)
            self.refresh_history()
        else:
            messagebox.showwarning("Warning", message)
    
    def auto_commit_monitor(self):
        """Monitor for changes and auto-commit/push"""
        while self.monitoring and self.auto_commit_enabled:
            try:
                time.sleep(self.auto_commit_interval)
                
                if not self.auto_commit_enabled:
                    break
                
                # Check for changes
                os.chdir(self.project_path)
                cmd = ["git", "status", "--porcelain"]
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                
                if result.stdout.strip():
                    # Changes detected, wait a bit more to ensure file is saved
                    time.sleep(5)
                    
                    # Check again to make sure changes are stable
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                    
                    if result.stdout.strip():
                        # Commit and push
                        success, msg = self.commit_and_push()
                        
                        if success:
                            # Update UI in main thread
                            self.root.after(0, lambda: self.status_label.config(
                                text=f"Status: Auto-committed and pushed", foreground="green"))
                            self.root.after(0, self.refresh_history)
                            
                            # Reset status after 5 seconds
                            self.root.after(5000, lambda: self.status_label.config(
                                text="Status: Auto-commit enabled", foreground="blue"))
                        else:
                            # Update UI with error
                            self.root.after(0, lambda: self.status_label.config(
                                text=f"Status: Auto-commit failed", foreground="red"))
                            
            except Exception as e:
                print(f"Auto-commit monitor error: {e}")
                time.sleep(self.auto_commit_interval)
    
    def show_new_commit_notification(self, commit):
        """Show notification for new commit"""
        self.status_label.config(text=f"New commit detected: {commit['message'][:50]}...", 
                               foreground="blue")
        self.root.after(5000, lambda: self.status_label.config(
            text="Status: Monitoring...", foreground="green"))
    
    def monitor_changes(self):
        """Monitor for changes in background"""
        while self.monitoring:
            try:
                time.sleep(30)  # Check every 30 seconds
                if self.monitoring:
                    self.root.after(0, self.refresh_history)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(30)

def main():
    root = tk.Tk()
    app = ChangeHistoryViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()

