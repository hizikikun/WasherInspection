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
        
        # Buttons - organized by category
        button_container = ttk.Frame(main_frame)
        button_container.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Category 1: 基本操作 (Basic Operations)
        basic_frame = ttk.LabelFrame(button_container, text="基本操作", padding=5)
        basic_frame.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X)
        
        refresh_btn = ttk.Button(basic_frame, text="更新", command=self.refresh_history)
        refresh_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        view_diff_btn = ttk.Button(basic_frame, text="差分表示", command=self.view_diff)
        view_diff_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        view_files_btn = ttk.Button(basic_frame, text="変更ファイル", command=self.view_changed_files)
        view_files_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        open_github_btn = ttk.Button(basic_frame, text="GitHubを開く", command=self.open_github)
        open_github_btn.pack(side=tk.LEFT)
        
        # Category 2: コミット操作 (Commit Operations)
        commit_frame = ttk.LabelFrame(button_container, text="コミット操作", padding=5)
        commit_frame.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X)
        
        auto_commit_btn = ttk.Button(commit_frame, text="コミット & プッシュ", command=self.manual_commit_push)
        auto_commit_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        test_commit_btn = ttk.Button(commit_frame, text="テストコミット (UTF-8)", command=self.create_test_commit)
        test_commit_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Auto-commit toggle
        self.auto_commit_var = tk.BooleanVar(value=self.auto_commit_enabled)
        auto_commit_check = ttk.Checkbutton(commit_frame, text="自動コミット有効", 
                                            variable=self.auto_commit_var,
                                            command=self.toggle_auto_commit)
        auto_commit_check.pack(side=tk.LEFT, padx=(5, 0))
        
        # Category 3: 文字化け修正 (Fix Garbled Text)
        fix_frame = ttk.LabelFrame(button_container, text="文字化け修正", padding=5)
        fix_frame.pack(side=tk.LEFT, fill=tk.X)
        
        fix_commit_btn = ttk.Button(fix_frame, text="選択コミット修正", command=self.fix_selected_commit)
        fix_commit_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        fix_files_btn = ttk.Button(fix_frame, text="ファイル修正", command=self.fix_garbled_files)
        fix_files_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        fix_all_commits_btn = ttk.Button(fix_frame, text="全コミット修正", command=self.fix_all_garbled_commits)
        fix_all_commits_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        reupload_all_btn = ttk.Button(fix_frame, text="全ファイル再アップロード", command=self.reupload_all_files)
        reupload_all_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        scan_github_btn = ttk.Button(fix_frame, text="GitHub文字化け検出", command=self.scan_github_garbled_files)
        scan_github_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        update_commit_messages_btn = ttk.Button(fix_frame, text="コミットメッセージ更新", command=self.update_commit_messages)
        update_commit_messages_btn.pack(side=tk.LEFT)
        
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
    
    # ===== 共通ヘルパー関数 =====
    
    def create_progress_dialog(self, title, width=700, height=600):
        """共通プログレスダイアログを作成"""
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title(title)
        progress_dialog.geometry(f"{width}x{height}")
        progress_dialog.transient(self.root)
        progress_dialog.grab_set()
        
        progress_label = ttk.Label(progress_dialog, text="処理中...")
        progress_label.pack(pady=10)
        
        progress_text = scrolledtext.ScrolledText(progress_dialog, height=25, width=80)
        progress_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        def log(message):
            progress_text.insert(tk.END, message + "\n")
            progress_text.see(tk.END)
            progress_dialog.update()
        
        close_btn = ttk.Button(progress_dialog, text="閉じる", 
                              command=progress_dialog.destroy)
        close_btn.pack(pady=10)
        
        return progress_dialog, log, close_btn
    
    def git_add_files(self, files=None, log_func=None):
        """Git add操作の共通処理"""
        try:
            os.chdir(self.project_path)
            if files is None:
                cmd = ["git", "add", "-A"]
            else:
                cmd = ["git", "add"] + (files if isinstance(files, list) else [files])
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                if log_func:
                    if files is None:
                        log_func("✓ 全てのファイルをステージングしました")
                    else:
                        file_list = files if isinstance(files, list) else [files]
                        for f in file_list:
                            log_func(f"✓ ステージング: {f}")
                return True, None
            else:
                error_msg = result.stderr.strip()
                if log_func:
                    log_func(f"✗ git add失敗: {error_msg}")
                return False, error_msg
        except Exception as e:
            error_msg = str(e)
            if log_func:
                log_func(f"✗ エラー: {error_msg}")
            return False, error_msg
    
    def git_commit(self, message, allow_empty=False, log_func=None):
        """Git commit操作の共通処理"""
        try:
            os.chdir(self.project_path)
            cmd = ["git", "commit", "-m", message]
            if allow_empty:
                cmd.append("--allow-empty")
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                if log_func:
                    log_func(f"✓ コミット成功: {message}")
                return True, None
            else:
                error_msg = result.stderr.strip()
                if log_func:
                    log_func(f"✗ コミット失敗: {error_msg}")
                return False, error_msg
        except Exception as e:
            error_msg = str(e)
            if log_func:
                log_func(f"✗ エラー: {error_msg}")
            return False, error_msg
    
    def git_push(self, branch="master", force=False, log_func=None):
        """Git push操作の共通処理"""
        try:
            os.chdir(self.project_path)
            cmd = ["git", "push", "origin", branch]
            if force:
                cmd.append("--force")
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                if log_func:
                    log_func("✓ プッシュ成功")
                return True, None
            else:
                # Try 'main' branch if 'master' fails
                if branch == "master":
                    cmd = ["git", "push", "origin", "main"]
                    if force:
                        cmd.append("--force")
                    result = subprocess.run(cmd, capture_output=True, text=True, 
                                          encoding='utf-8', errors='replace')
                    if result.returncode == 0:
                        if log_func:
                            log_func("✓ プッシュ成功 (mainブランチ)")
                        return True, None
                
                error_msg = result.stderr.strip()
                if log_func:
                    log_func(f"✗ プッシュ失敗: {error_msg}")
                return False, error_msg
        except Exception as e:
            error_msg = str(e)
            if log_func:
                log_func(f"✗ エラー: {error_msg}")
            return False, error_msg
    
    def fix_file_encoding(self, file_path, log_func=None):
        """ファイルのエンコーディングを修正（共通処理）"""
        try:
            if not os.path.exists(file_path):
                return False, "ファイルが存在しません"
            
            # Read file as binary first
            with open(file_path, 'rb') as f:
                raw_bytes = f.read()
            
            # Try to decode as UTF-8
            try:
                content = raw_bytes.decode('utf-8')
                is_utf8 = True
            except UnicodeDecodeError:
                # Try Shift-JIS
                try:
                    content = raw_bytes.decode('shift_jis')
                    is_utf8 = False
                    if log_func:
                        log_func(f"✓ {file_path}: Shift-JISとして読み込みました")
                except:
                    if log_func:
                        log_func(f"✗ {file_path}: エンコーディング不明（スキップ）")
                    return False, "エンコーディング不明"
            
            # Check if content is garbled or needs conversion
            needs_fix = False
            
            if is_utf8 and self.is_garbled(content):
                # UTF-8として読めたが文字化けしている
                needs_fix = True
                if log_func:
                    log_func(f"文字化け検出: {file_path}")
            elif not is_utf8:
                # Shift-JISとしてしか読めない - UTF-8に変換
                needs_fix = True
                if log_func:
                    log_func(f"Shift-JIS検出: {file_path}")
            
            if needs_fix:
                # Try to fix by decoding as Shift-JIS and converting to UTF-8
                try:
                    decoded = raw_bytes.decode('shift_jis')
                    # Write back as UTF-8
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(decoded)
                    if log_func:
                        log_func(f"  → 修正完了: {file_path}")
                    return True, "修正完了"
                except UnicodeDecodeError:
                    if log_func:
                        log_func(f"  → 修正失敗: Shift-JISとして解釈できませんでした")
                    return False, "Shift-JISとして解釈できません"
                except Exception as e:
                    if log_func:
                        log_func(f"  → エラー: {e}")
                    return False, str(e)
            else:
                # Already UTF-8 and not garbled - ensure it's saved as UTF-8
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    return True, "既にUTF-8"
                except:
                    pass
            
            return False, "修正不要"
        except Exception as e:
            if log_func:
                log_func(f"✗ {file_path}: エラー - {e}")
            return False, str(e)
    
    def get_text_files(self, files, log_func=None):
        """テキストファイルをフィルタリング（共通処理）"""
        text_extensions = {'.py', '.md', '.txt', '.json', '.yml', '.yaml', '.js', '.ts', 
                         '.html', '.css', '.xml', '.ps1', '.bat', '.sh', '.cfg', '.ini'}
        
        text_files = []
        for file in files:
            if any(file.endswith(ext) for ext in text_extensions):
                text_files.append(file)
        
        if log_func:
            log_func(f"テキストファイル数: {len(text_files)}")
        
        return text_files
    
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
    
    def create_test_commit(self):
        """Create a test commit with UTF-8 encoding to verify encoding settings"""
        try:
            os.chdir(self.project_path)
            
            # Create a test file with Japanese content
            test_file = self.project_path / ".test_encoding_check.txt"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            test_content = f"""テストコミット: UTF-8エンコーディング確認
Test Commit: UTF-8 Encoding Verification
作成日時: {timestamp}

このファイルは文字化けしないことを確認するためのテストファイルです。
This file is for testing UTF-8 encoding to ensure no garbled characters.
"""
            
            # Write test file with UTF-8 encoding
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            # Set Git environment for UTF-8
            env = os.environ.copy()
            env['GIT_COMMITTER_NAME'] = 'Test User'
            env['GIT_COMMITTER_EMAIL'] = 'test@example.com'
            env['LANG'] = 'en_US.UTF-8'
            env['LC_ALL'] = 'en_US.UTF-8'
            
            # Add the test file
            cmd = ["git", "add", str(test_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)
            
            if result.returncode != 0:
                messagebox.showerror("Error", f"ファイルの追加に失敗しました: {result.stderr}")
                return
            
            # Create commit with UTF-8 message
            commit_message = f"テストコミット (UTF-8): {timestamp}\nTest commit to verify UTF-8 encoding is working correctly."
            
            # Use git commit with explicit UTF-8 encoding
            cmd = ["git", "-c", "i18n.commitEncoding=utf-8", "commit", "-m", commit_message]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)
            
            if result.returncode != 0:
                messagebox.showerror("Error", f"コミットに失敗しました: {result.stderr}")
                return
            
            # Push to GitHub
            cmd = ["git", "push", "origin", "master"]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)
            
            if result.returncode != 0:
                # Try 'main' branch if 'master' fails
                cmd = ["git", "push", "origin", "main"]
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)
                
                if result.returncode != 0:
                    messagebox.showwarning("Warning", 
                        f"コミットは成功しましたが、プッシュに失敗しました: {result.stderr}\n\n"
                        "手動でプッシュしてください。")
                    self.refresh_history()
                    return
            
            messagebox.showinfo("Success", 
                f"テストコミットが正常に作成され、GitHubにプッシュされました！\n\n"
                f"コミットメッセージ: {commit_message[:50]}...\n\n"
                "GitHubで文字化けしていないか確認してください。")
            
            # Refresh history
            self.refresh_history()
            
        except Exception as e:
            messagebox.showerror("Error", f"エラーが発生しました: {e}")
    
    def is_garbled(self, text):
        """Check if text contains garbled characters (Shift-JIS misread as UTF-8)"""
        garbled_chars = ['縺', '繧', '繝', '謨', '譁', '邨', '蜷']
        return any(char in text for char in garbled_chars)
    
    def fix_selected_commit(self):
        """Fix the message of the selected garbled commit"""
        selection = self.commit_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "コミットを選択してください")
            return
        
        item = self.commit_tree.item(selection[0])
        tags = item['tags']
        if not tags:
            return
        
        commit_hash = tags[0]
        current_message = item['values'][2]  # Message column
        
        # Check if message is garbled
        if not self.is_garbled(current_message):
            response = messagebox.askyesno("確認", 
                f"このコミットメッセージは文字化けしていないようです。\n\n"
                f"現在のメッセージ: {current_message[:50]}...\n\n"
                "それでも修正しますか？")
            if not response:
                return
        
        # Get full commit message
        try:
            os.chdir(self.project_path)
            cmd = ["git", "log", "-1", "--pretty=format:%B", commit_hash]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            full_message = result.stdout.strip() if result.returncode == 0 else current_message
        except:
            full_message = current_message
        
        # Create dialog to input new message
        dialog = tk.Toplevel(self.root)
        dialog.title("コミットメッセージを修正")
        dialog.geometry("600x400")
        
        ttk.Label(dialog, text="現在のメッセージ（文字化けしている可能性）:").pack(padx=10, pady=5, anchor=tk.W)
        old_text = scrolledtext.ScrolledText(dialog, height=5, wrap=tk.WORD)
        old_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        old_text.insert(1.0, full_message)
        old_text.config(state=tk.DISABLED)
        
        ttk.Label(dialog, text="新しいメッセージ（英語で入力してください）:").pack(padx=10, pady=5, anchor=tk.W)
        new_text = scrolledtext.ScrolledText(dialog, height=5, wrap=tk.WORD)
        new_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        new_text.insert(1.0, self.suggest_fixed_message(full_message))
        
        def do_fix():
            new_message = new_text.get(1.0, tk.END).strip()
            if not new_message:
                messagebox.showwarning("Warning", "新しいメッセージを入力してください")
                return
            
            dialog.destroy()
            self.apply_commit_fix(commit_hash, new_message)
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(padx=10, pady=10)
        
        ttk.Button(button_frame, text="修正", command=do_fix).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="キャンセル", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def suggest_fixed_message(self, garbled_message):
        """Suggest a fixed message based on garbled message patterns"""
        # Common patterns for garbled messages
        suggestions = {
            "Update: GitHub": "Update: GitHub integration setup",
            "Reorganize": "Reorganize: Clean up project structure",
            "Initial": "Initial: Add files to repository",
            "docs": "Update: Add documentation files",
            "繧繝・": "Update: ",
            "繝・": "Update: ",
        }
        
        # Try to extract meaningful parts
        if "Update:" in garbled_message or "繧繝・" in garbled_message:
            return "Update: Fix encoding and update files"
        elif "Reorganize" in garbled_message or "謨" in garbled_message:
            return "Reorganize: Clean up project structure"
        elif "Initial" in garbled_message or "譖" in garbled_message:
            return "Initial: Add initial project files"
        else:
            return "Update: Fix commit message encoding"
    
    def apply_commit_fix(self, commit_hash, new_message):
        """Apply the fix to a commit message using git rebase"""
        try:
            # Warning dialog
            response = messagebox.askyesno("警告", 
                "コミットメッセージを修正すると、Git履歴が書き換えられます。\n\n"
                "既にGitHubにpushされている場合は、force pushが必要になります。\n\n"
                "他の人と共同作業している場合は、事前に相談してください。\n\n"
                "続行しますか？")
            
            if not response:
                return
            
            os.chdir(self.project_path)
            
            # Get commit position in history
            cmd = ["git", "log", "--oneline", "--all"]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            commits = result.stdout.strip().split('\n')
            
            # Find commit position
            commit_index = None
            for idx, line in enumerate(commits):
                if line.startswith(commit_hash[:8]) or commit_hash in line:
                    commit_index = len(commits) - idx
                    break
            
            if commit_index is None:
                messagebox.showerror("Error", "コミットが見つかりませんでした")
                return
            
            # Use git commit --amend for the latest commit, or rebase for older ones
            if commit_index == 1:
                # Latest commit - use amend
                env = os.environ.copy()
                env['GIT_COMMITTER_NAME'] = 'Fixed User'
                env['GIT_COMMITTER_EMAIL'] = 'fixed@example.com'
                env['LANG'] = 'en_US.UTF-8'
                env['LC_ALL'] = 'en_US.UTF-8'
                
                cmd = ["git", "-c", "i18n.commitEncoding=utf-8", "commit", "--amend", "-m", new_message]
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)
                
                if result.returncode != 0:
                    messagebox.showerror("Error", f"コミットの修正に失敗しました: {result.stderr}")
                    return
                
                # Ask if user wants to push
                push_response = messagebox.askyesno("確認", 
                    "コミットを修正しました。GitHubにpushしますか？\n\n"
                    "（force pushが必要になります）")
                
                if push_response:
                    # Force push
                    cmd = ["git", "push", "origin", "master", "--force"]
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)
                    
                    if result.returncode != 0:
                        cmd = ["git", "push", "origin", "main", "--force"]
                        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)
                        
                        if result.returncode != 0:
                            messagebox.showwarning("Warning", 
                                f"プッシュに失敗しました: {result.stderr}\n\n"
                                "手動で force push してください:\n"
                                "git push origin master --force")
                            return
                    
                    messagebox.showinfo("Success", "コミットメッセージを修正し、GitHubにpushしました！")
                    self.refresh_history()
                else:
                    messagebox.showinfo("情報", 
                        "コミットは修正されましたが、まだpushされていません。\n\n"
                        "後で手動でpushしてください:\n"
                        "git push origin master --force")
            else:
                messagebox.showinfo("情報", 
                    f"このコミットは履歴の{commit_index}番目にあります。\n\n"
                    "古いコミットを修正するには、git rebase を使用する必要があります。\n\n"
                    "手動で修正してください:\n"
                    f"git rebase -i HEAD~{commit_index}\n\n"
                    "エディタで該当コミットの 'pick' を 'reword' に変更してください。")
        
        except Exception as e:
            messagebox.showerror("Error", f"エラーが発生しました: {e}")
    
    def fix_garbled_files(self):
        """Scan and fix garbled files in the repository"""
        try:
            os.chdir(self.project_path)
            
            # Use common progress dialog
            progress_dialog, log, close_btn = self.create_progress_dialog("ファイル文字化け修正", width=600, height=500)
            
            def scan_and_fix():
                try:
                    # Get list of files in repository
                    cmd = ["git", "ls-files"]
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
                    if result.returncode != 0:
                        log("エラー: Gitリポジトリではありません")
                        return
                    
                    files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
                    log(f"スキャン対象ファイル数: {len(files)}")
                    log("")
                    
                    # Filter text files
                    text_files = self.get_text_files(files, log_func=log)
                    log("")
                    
                    garbled_files = []
                    fixed_files = []
                    
                    for file_path in text_files:
                        # Skip if file doesn't exist
                        if not os.path.exists(file_path):
                            continue
                        
                        # Use common encoding fix function
                        success, status = self.fix_file_encoding(file_path, log_func=log)
                        if success and status == "修正完了":
                            garbled_files.append(file_path)
                            fixed_files.append(file_path)
                    
                    log("")
                    log("=" * 60)
                    log(f"スキャン完了")
                    log(f"文字化けファイル: {len(garbled_files)}個")
                    log(f"修正完了ファイル: {len(fixed_files)}個")
                    log("")
                    
                    if fixed_files:
                        log("修正したファイルをコミットしますか？")
                        log("")
                        
                        # Ask user if they want to commit
                        progress_dialog.update()
                        response = messagebox.askyesno("確認", 
                            f"{len(fixed_files)}個のファイルを修正しました。\n\n"
                            f"修正したファイルをコミット・プッシュしますか？")
                        
                        if response:
                            log("")
                            log("コミット中...")
                            
                            # Stage all fixed files
                            success, error = self.git_add_files(fixed_files, log_func=log)
                            
                            if success:
                                # Commit
                                commit_message = f"Fix: 文字化けファイルをUTF-8に変換 ({len(fixed_files)}ファイル)"
                                success, error = self.git_commit(commit_message, log_func=log)
                                
                                if success:
                                    # Push
                                    log("")
                                    log("プッシュ中...")
                                    success, error = self.git_push("master", force=False, log_func=log)
                                    
                                    if success:
                                        messagebox.showinfo("成功", 
                                            f"{len(fixed_files)}個のファイルを修正してコミット・プッシュしました。")
                                    else:
                                        messagebox.showerror("エラー", f"プッシュに失敗しました:\n{error}")
                                else:
                                    messagebox.showerror("エラー", f"コミットに失敗しました:\n{error}")
                    else:
                        log("修正が必要なファイルは見つかりませんでした。")
                        messagebox.showinfo("完了", "文字化けファイルは見つかりませんでした。")
                    
                except Exception as e:
                    log(f"エラー: {e}")
                    messagebox.showerror("Error", f"エラーが発生しました: {e}")
                    progress_dialog.destroy()
            
            # Run scan in background thread
            threading.Thread(target=scan_and_fix, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"エラーが発生しました: {e}")
    
    def fix_all_garbled_commits(self):
        """Scan and fix all garbled commit messages in the repository"""
        try:
            os.chdir(self.project_path)
            
            # Show warning first
            response = messagebox.askyesno("警告", 
                "この操作は、Gitの履歴を書き換えます。\n\n"
                "⚠️ 注意事項:\n"
                "• 既にpushされたコミットを修正するには force push が必要です\n"
                "• 他の人と共同作業している場合は、事前に相談してください\n"
                "• この操作は取り消せません\n\n"
                "続行しますか？")
            
            if not response:
                return
            
            # Show progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("全コミット文字化け修正")
            progress_dialog.geometry("700x600")
            progress_dialog.transient(self.root)
            progress_dialog.grab_set()
            
            progress_label = ttk.Label(progress_dialog, text="コミット履歴をスキャン中...")
            progress_label.pack(pady=10)
            
            progress_text = scrolledtext.ScrolledText(progress_dialog, height=25, width=80)
            progress_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            
            def log(message):
                progress_text.insert(tk.END, message + "\n")
                progress_text.see(tk.END)
                progress_dialog.update()
            
            def scan_and_fix():
                try:
                    # Get all commits
                    cmd = ["git", "log", "--all", "--pretty=format:%H|%s", "--reverse"]
                    result = subprocess.run(cmd, capture_output=True, text=True, 
                                          encoding='utf-8', errors='replace')
                    
                    if result.returncode != 0:
                        log("エラー: Gitリポジトリではありません")
                        return
                    
                    commits = []
                    for line in result.stdout.strip().split('\n'):
                        if '|' in line:
                            hash_part, message = line.split('|', 1)
                            commits.append((hash_part.strip(), message.strip()))
                    
                    log(f"総コミット数: {len(commits)}")
                    log("")
                    
                    garbled_commits = []
                    
                    for commit_hash, message in commits:
                        if self.is_garbled(message):
                            garbled_commits.append((commit_hash, message))
                            log(f"文字化け検出: {commit_hash[:8]} - {message[:50]}...")
                    
                    log("")
                    log("=" * 60)
                    log(f"文字化けコミット数: {len(garbled_commits)}")
                    log("")
                    
                    if not garbled_commits:
                        log("文字化けしているコミットは見つかりませんでした。")
                        close_btn = ttk.Button(progress_dialog, text="閉じる", 
                                              command=progress_dialog.destroy)
                        close_btn.pack(pady=10)
                        return
                    
                    log("修正方法を選択してください:")
                    log("1. 各コミットメッセージを手動で入力して修正")
                    log("2. 文字化けを自動検出して修正（推奨）")
                    log("")
                    
                    # Ask user for method
                    progress_dialog.update()
                    
                    # Create method selection dialog
                    method_dialog = tk.Toplevel(progress_dialog)
                    method_dialog.title("修正方法を選択")
                    method_dialog.geometry("400x200")
                    method_dialog.transient(progress_dialog)
                    method_dialog.grab_set()
                    
                    ttk.Label(method_dialog, 
                            text=f"{len(garbled_commits)}個の文字化けコミットが見つかりました。\n\n修正方法を選択してください:",
                            justify=tk.LEFT).pack(pady=10)
                    
                    method_choice = tk.StringVar(value="auto")
                    
                    def on_method_selected():
                        method_dialog.destroy()
                    
                    ttk.Radiobutton(method_dialog, text="自動修正（文字化けを検出して修正）", 
                                   variable=method_choice, value="auto").pack(anchor=tk.W, padx=20)
                    ttk.Radiobutton(method_dialog, text="手動修正（各コミットメッセージを入力）", 
                                   variable=method_choice, value="manual").pack(anchor=tk.W, padx=20)
                    
                    ttk.Button(method_dialog, text="続行", command=on_method_selected).pack(pady=10)
                    
                    # Wait for method selection
                    method_dialog.wait_window()
                    
                    use_auto = method_choice.get() == "auto"
                    
                    log("")
                    if use_auto:
                        log("自動修正モードで修正します...")
                        log("⚠️ 注意: 自動修正は正確ではない場合があります。")
                        log("修正後、コミット履歴を確認してください。")
                        log("")
                    else:
                        log("手動修正モードで修正します...")
                        log("各コミットの新しいメッセージを入力してください。")
                        log("")
                    
                    fixed_count = 0
                    
                    for idx, (commit_hash, old_message) in enumerate(garbled_commits, 1):
                        log(f"[{idx}/{len(garbled_commits)}] {commit_hash[:8]}")
                        log(f"  現在: {old_message[:60]}...")
                        
                        if use_auto:
                            # Try to auto-fix by detecting common patterns
                            new_message = self.auto_fix_commit_message(old_message)
                            log(f"  自動修正: {new_message[:60]}...")
                        else:
                            # Manual input
                            input_dialog = tk.Toplevel(progress_dialog)
                            input_dialog.title(f"コミット {commit_hash[:8]} を修正")
                            input_dialog.geometry("600x300")
                            input_dialog.transient(progress_dialog)
                            input_dialog.grab_set()
                            
                            ttk.Label(input_dialog, text=f"コミット: {commit_hash[:8]}").pack(pady=5)
                            ttk.Label(input_dialog, text="現在のメッセージ:", font=("", 9)).pack(anchor=tk.W, padx=10)
                            
                            old_text = scrolledtext.ScrolledText(input_dialog, height=3, width=70)
                            old_text.insert('1.0', old_message)
                            old_text.config(state=tk.DISABLED)
                            old_text.pack(padx=10, pady=5, fill=tk.X)
                            
                            ttk.Label(input_dialog, text="新しいメッセージ:", font=("", 9)).pack(anchor=tk.W, padx=10)
                            
                            new_text = scrolledtext.ScrolledText(input_dialog, height=5, width=70)
                            new_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
                            
                            new_message = None
                            
                            def on_ok():
                                nonlocal new_message
                                new_message = new_text.get('1.0', tk.END).strip()
                                input_dialog.destroy()
                            
                            def on_skip():
                                nonlocal new_message
                                new_message = None
                                input_dialog.destroy()
                            
                            btn_frame = ttk.Frame(input_dialog)
                            btn_frame.pack(pady=10)
                            ttk.Button(btn_frame, text="修正", command=on_ok).pack(side=tk.LEFT, padx=5)
                            ttk.Button(btn_frame, text="スキップ", command=on_skip).pack(side=tk.LEFT, padx=5)
                            
                            input_dialog.wait_window()
                            
                            if not new_message:
                                log("  → スキップ")
                                continue
                        
                        # Try to fix commit using git rebase
                        log(f"  → 修正中...")
                        
                        try:
                            # First, verify the commit exists
                            cmd = ["git", "cat-file", "-e", commit_hash]
                            result = subprocess.run(cmd, capture_output=True, text=True, 
                                                  encoding='utf-8', errors='replace')
                            if result.returncode != 0:
                                # Try with short hash
                                cmd = ["git", "cat-file", "-e", commit_hash[:8]]
                                result = subprocess.run(cmd, capture_output=True, text=True, 
                                                      encoding='utf-8', errors='replace')
                                if result.returncode != 0:
                                    log(f"  ✗ エラー: コミットが見つかりません (ハッシュ: {commit_hash[:8]})")
                                    continue
                            
                            # Get commit position
                            cmd = ["git", "log", "--oneline", "--all", "--reverse"]
                            result = subprocess.run(cmd, capture_output=True, text=True, 
                                                  encoding='utf-8', errors='replace')
                            if result.returncode != 0:
                                log(f"  ✗ エラー: コミット履歴の取得に失敗")
                                continue
                            
                            commits = result.stdout.strip().split('\n')
                            commit_index = None
                            for idx, line in enumerate(commits):
                                # Check both full hash and short hash
                                if commit_hash in line or commit_hash[:8] in line.split()[0]:
                                    commit_index = idx + 1
                                    break
                            
                            if commit_index is None:
                                log(f"  ✗ エラー: コミットが見つかりません (ハッシュ: {commit_hash[:8]})")
                                log(f"     ヒント: このコミットは別のブランチにある可能性があります")
                                continue
                            
                            # Use git commit --amend for latest, rebase for others
                            if commit_index == len(commits):
                                # Latest commit - use amend
                                cmd = ["git", "commit", "--amend", "-m", new_message]
                                result = subprocess.run(cmd, capture_output=True, text=True, 
                                                      encoding='utf-8', errors='replace',
                                                      env={**os.environ, 'GIT_EDITOR': 'true'})
                                if result.returncode == 0:
                                    log(f"  ✓ 修正完了 (amend)")
                                    fixed_count += 1
                                else:
                                    log(f"  ✗ 修正失敗: {result.stderr}")
                            else:
                                # Older commit - try to use git filter-branch or rebase
                                # For Windows, we'll use git filter-branch with a script
                                log(f"  → 古いコミットの修正を試行中...")
                                
                                # Try using git filter-branch with --msg-filter
                                # This is safer than rebase for automation
                                import tempfile
                                import os
                                
                                # Create a script that will be used by git filter-branch
                                filter_script = f'''if [ "$GIT_COMMIT" = "{commit_hash}" ]; then
    echo "{new_message}"
else
    cat
fi
'''
                                
                                with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False, encoding='utf-8') as f:
                                    f.write(filter_script)
                                    filter_script_path = f.name
                                
                                try:
                                    # On Windows, we need to use git filter-branch or git rebase
                                    # Let's try a simpler approach: use git rebase with environment variables
                                    # But first, save the script path
                                    os.chmod(filter_script_path, 0o755)
                                    
                                    # For now, provide manual instructions but with better details
                                    log(f"  ⚠️ 古いコミットの修正には手動操作が必要です:")
                                    log(f"     方法1: git rebase -i を使用")
                                    log(f"     1. git rebase -i {commit_hash}^")
                                    log(f"     2. エディタで 'pick {commit_hash[:8]}' を 'reword {commit_hash[:8]}' に変更")
                                    log(f"     3. 保存して閉じる")
                                    log(f"     4. 新しいメッセージを入力: {new_message}")
                                    log(f"")
                                    log(f"     方法2: git filter-branch を使用（上級者向け）")
                                    log(f"     git filter-branch -f --msg-filter '")
                                    log(f"       if [ \"$GIT_COMMIT\" = \"{commit_hash}\" ]; then")
                                    log(f"         echo \"{new_message}\"")
                                    log(f"       else")
                                    log(f"         cat")
                                    log(f"       fi")
                                    log(f"     ' HEAD")
                                    log("")
                                    
                                    # Count as manual intervention needed
                                    # Don't increment fixed_count
                                    
                                finally:
                                    # Clean up temp file
                                    try:
                                        os.unlink(filter_script_path)
                                    except:
                                        pass
                        
                        except Exception as e:
                            log(f"  ✗ エラー: {e}")
                            import traceback
                            log(traceback.format_exc())
                    
                    log("")
                    log("=" * 60)
                    log(f"修正完了: {fixed_count}個")
                    log("")
                    
                    if fixed_count > 0:
                        log("⚠️ 次のステップ:")
                        log("1. 修正したコミットを確認してください")
                        log("2. 既にpushされたコミットを修正した場合は、force pushが必要です:")
                        log("   git push origin master --force")
                        log("")
                        log("⚠️ 警告: force pushは他の人と共同作業している場合は危険です。")
                        log("   事前にチームメンバーに相談してください。")
                        log("")
                        
                        # Ask if user wants to push
                        progress_dialog.update()
                        response = messagebox.askyesno("確認", 
                            f"{fixed_count}個のコミットを修正しました。\n\n"
                            "GitHubにプッシュしますか？\n\n"
                            "⚠️ 注意: force pushが必要な場合があります。")
                        
                        if response:
                            log("")
                            log("プッシュ中...")
                            cmd = ["git", "push", "origin", "master", "--force"]
                            result = subprocess.run(cmd, capture_output=True, text=True, 
                                                  encoding='utf-8', errors='replace')
                            
                            if result.returncode == 0:
                                log("✓ プッシュ成功")
                                messagebox.showinfo("成功", "コミットを修正してプッシュしました。")
                            else:
                                log(f"✗ プッシュ失敗: {result.stderr}")
                                messagebox.showerror("エラー", f"プッシュに失敗しました:\n{result.stderr}")
                    else:
                        log("⚠️ 重要:")
                        log("古いコミットの修正には手動でgit rebaseが必要です。")
                        log("既にpushされたコミットを修正する場合は、force pushが必要です:")
                        log("  git push origin master --force")
                        log("")
                        log("詳細な手順については、上記の各コミットの修正方法を参照してください。")
                    
                    close_btn = ttk.Button(progress_dialog, text="閉じる", 
                                          command=progress_dialog.destroy)
                    close_btn.pack(pady=10)
                    
                except Exception as e:
                    log(f"エラー: {e}")
                    import traceback
                    log(traceback.format_exc())
                    messagebox.showerror("Error", f"エラーが発生しました: {e}")
                    progress_dialog.destroy()
            
            # Run scan in background thread
            threading.Thread(target=scan_and_fix, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"エラーが発生しました: {e}")
    
    def auto_fix_commit_message(self, garbled_message):
        """Try to auto-fix a garbled commit message"""
        # Common patterns for fixing
        # This is a simple heuristic - may need improvement
        
        # Try to decode as Shift-JIS if it looks garbled
        try:
            # If message contains garbled chars, try to reconstruct
            # This is tricky - we can't easily reverse the garbling
            # So we'll just suggest a generic message
            if "Update" in garbled_message or "菫ｮ豁｣" in garbled_message:
                return "Update: ファイル更新"
            elif "Initial" in garbled_message or "譖ｴ譁ｰ" in garbled_message:
                return "Initial: 初期コミット"
            elif "Auto-commit" in garbled_message:
                return f"Auto-commit: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            else:
                return "Update: コミットメッセージ修正"
        except:
            return "Update: コミットメッセージ修正"
    
    def reupload_all_files(self):
        """Re-upload all files to GitHub with encoding fixes"""
        try:
            os.chdir(self.project_path)
            
            # Show warning first
            response = messagebox.askyesno("警告", 
                "この操作は、全てのファイルを再アップロードします。\n\n"
                "⚠️ 注意事項:\n"
                "• 全てのファイルを新しいコミットとして追加します\n"
                "• 文字化けしているファイルを自動修正します\n"
                "• 既にpushされたコミットを上書きするため、force pushが必要です\n"
                "• 他の人と共同作業している場合は、事前に相談してください\n\n"
                "続行しますか？")
            
            if not response:
                return
            
            # Show progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("全ファイル再アップロード")
            progress_dialog.geometry("700x600")
            progress_dialog.transient(self.root)
            progress_dialog.grab_set()
            
            progress_label = ttk.Label(progress_dialog, text="ファイルを処理中...")
            progress_label.pack(pady=10)
            
            progress_text = scrolledtext.ScrolledText(progress_dialog, height=25, width=80)
            progress_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            
            def log(message):
                progress_text.insert(tk.END, message + "\n")
                progress_text.see(tk.END)
                progress_dialog.update()
            
            def process_files():
                try:
                    # Step 1: Pull latest from GitHub
                    log("=" * 60)
                    log("ステップ1: GitHubから最新のファイルを取得中...")
                    log("")
                    
                    cmd = ["git", "pull", "origin", "master"]
                    result = subprocess.run(cmd, capture_output=True, text=True, 
                                          encoding='utf-8', errors='replace')
                    
                    if result.returncode == 0:
                        log("✓ 最新ファイルを取得しました")
                    else:
                        log(f"⚠️ 警告: pullに失敗しましたが、続行します: {result.stderr}")
                    
                    log("")
                    log("=" * 60)
                    log("ステップ2: ファイルをスキャン・修正中...")
                    log("")
                    
                    # Step 2: Get all files and fix garbled ones
                    cmd = ["git", "ls-files"]
                    result = subprocess.run(cmd, capture_output=True, text=True, 
                                          encoding='utf-8', errors='replace')
                    
                    if result.returncode != 0:
                        log("✗ エラー: Gitリポジトリではありません")
                        return
                    
                    files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
                    log(f"スキャン対象ファイル数: {len(files)}")
                    log("")
                    
                    # Filter text files
                    text_files = self.get_text_files(files, log_func=log)
                    log("")
                    
                    fixed_files = []
                    
                    for file_path in text_files:
                        if not os.path.exists(file_path):
                            continue
                        
                        # Use common encoding fix function
                        success, status = self.fix_file_encoding(file_path, log_func=log)
                        if success and (status == "修正完了" or status == "既にUTF-8"):
                            if status == "修正完了":
                                fixed_files.append(file_path)
                    
                    log("")
                    log("=" * 60)
                    log(f"修正完了ファイル: {len(fixed_files)}個")
                    log("")
                    
                    # Step 3: Stage all files
                    log("=" * 60)
                    log("ステップ3: 全てのファイルをステージング中...")
                    log("")
                    
                    success, error = self.git_add_files(None, log_func=log)
                    if not success:
                        log(f"✗ エラー: {error}")
                        return
                    
                    # Step 4: Commit
                    log("")
                    log("=" * 60)
                    log("ステップ4: コミット中...")
                    log("")
                    
                    commit_message = f"Fix: 全ファイルをUTF-8エンコーディングに統一 ({len(fixed_files)}ファイル修正)"
                    success, error = self.git_commit(commit_message, log_func=log)
                    
                    if success:
                        # Step 5: Push
                        log("")
                        log("=" * 60)
                        log("ステップ5: GitHubにプッシュ中...")
                        log("")
                        
                        # Ask user if they want to force push
                        progress_dialog.update()
                        response = messagebox.askyesno("確認", 
                            f"{len(fixed_files)}個のファイルを修正しました。\n\n"
                            "GitHubにプッシュしますか？\n\n"
                            "⚠️ 注意: force pushが必要な場合があります。")
                        
                        if response:
                            success, error = self.git_push("master", force=True, log_func=log)
                            
                            if success:
                                log("")
                                log("=" * 60)
                                log("完了！")
                                log("全てのファイルを再アップロードし、文字化けを修正しました。")
                                messagebox.showinfo("成功", 
                                    f"{len(fixed_files)}個のファイルを修正してGitHubに再アップロードしました。")
                            else:
                                messagebox.showerror("エラー", f"プッシュに失敗しました:\n{error}")
                        else:
                            log("")
                            log("プッシュはキャンセルされました。")
                            log("後で手動でプッシュしてください:")
                            log("  git push origin master --force")
                    else:
                        messagebox.showerror("エラー", f"コミットに失敗しました:\n{error}")
                    
                except Exception as e:
                    log(f"エラー: {e}")
                    import traceback
                    log(traceback.format_exc())
                    messagebox.showerror("Error", f"エラーが発生しました: {e}")
                    progress_dialog.destroy()
            
            # Run in background thread
            threading.Thread(target=process_files, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"エラーが発生しました: {e}")
    
    def scan_github_garbled_files(self):
        """Scan GitHub remote repository for garbled files"""
        try:
            os.chdir(self.project_path)
            
            # Use common progress dialog
            progress_dialog, log, close_btn = self.create_progress_dialog("GitHub文字化けファイル検出", width=800, height=700)
            
            def scan_files():
                try:
                    log("=" * 60)
                    log("GitHub上の文字化けファイルを検出中...")
                    log("")
                    
                    # Step 1: Pull latest from GitHub
                    log("ステップ1: GitHubから最新のファイル情報を取得中...")
                    log("")
                    
                    cmd = ["git", "fetch", "origin"]
                    result = subprocess.run(cmd, capture_output=True, text=True, 
                                          encoding='utf-8', errors='replace')
                    
                    if result.returncode != 0:
                        log(f"⚠️ 警告: fetchに失敗しましたが、続行します: {result.stderr}")
                    
                    log("")
                    log("=" * 60)
                    log("ステップ2: リモートブランチのファイル一覧を取得中...")
                    log("")
                    
                    # Step 2: Get all files from remote
                    cmd = ["git", "ls-tree", "-r", "--name-only", "origin/master"]
                    result = subprocess.run(cmd, capture_output=True, text=True, 
                                          encoding='utf-8', errors='replace')
                    
                    if result.returncode != 0:
                        # Try origin/main
                        cmd = ["git", "ls-tree", "-r", "--name-only", "origin/main"]
                        result = subprocess.run(cmd, capture_output=True, text=True, 
                                              encoding='utf-8', errors='replace')
                    
                    if result.returncode != 0:
                        log("✗ エラー: リモートブランチにアクセスできません")
                        log("ローカルファイルをスキャンします...")
                        # Fallback to local files
                        cmd = ["git", "ls-files"]
                        result = subprocess.run(cmd, capture_output=True, text=True, 
                                              encoding='utf-8', errors='replace')
                        
                        if result.returncode != 0:
                            log("✗ エラー: Gitリポジトリではありません")
                            return
                    
                    files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
                    log(f"スキャン対象ファイル数: {len(files)}")
                    log("")
                    
                    # Filter text files
                    text_files = self.get_text_files(files, log_func=log)
                    log("")
                    
                    log("=" * 60)
                    log("ステップ3: 各ファイルの内容をチェック中...")
                    log("")
                    
                    garbled_files = []
                    checked_count = 0
                    
                    for file_path in text_files:
                        checked_count += 1
                        if checked_count % 10 == 0:
                            log(f"チェック中... ({checked_count}/{len(text_files)})")
                            progress_dialog.update()
                        
                        try:
                            # Get file content from remote (or HEAD if remote fails)
                            cmd = ["git", "show", f"origin/master:{file_path}"]
                            result = subprocess.run(cmd, capture_output=True, text=True, 
                                                  encoding='utf-8', errors='replace')
                            
                            if result.returncode != 0:
                                # Try origin/main
                                cmd = ["git", "show", f"origin/main:{file_path}"]
                                result = subprocess.run(cmd, capture_output=True, text=True, 
                                                      encoding='utf-8', errors='replace')
                            
                            if result.returncode != 0:
                                # Fallback to HEAD
                                cmd = ["git", "show", f"HEAD:{file_path}"]
                                result = subprocess.run(cmd, capture_output=True, text=True, 
                                                      encoding='utf-8', errors='replace')
                            
                            if result.returncode == 0:
                                content = result.stdout
                                
                                # Check if content is garbled
                                if self.is_garbled(content):
                                    garbled_files.append(file_path)
                                    log(f"✗ 文字化け検出: {file_path}")
                        except Exception as e:
                            # Skip binary files or files that can't be read
                            pass
                    
                    log("")
                    log("=" * 60)
                    log(f"スキャン完了")
                    log(f"チェックしたファイル数: {checked_count}")
                    log(f"文字化けファイル数: {len(garbled_files)}")
                    log("")
                    
                    if garbled_files:
                        log("=" * 60)
                        log("文字化けしているファイル:")
                        log("=" * 60)
                        for file_path in garbled_files:
                            log(f"  • {file_path}")
                        log("")
                        log("=" * 60)
                        log("")
                        log("これらのファイルを修正しますか？")
                        log("「ファイル修正」ボタンを使用して修正できます。")
                        
                        # Show result window
                        result_window = tk.Toplevel(progress_dialog)
                        result_window.title("文字化けファイル検出結果")
                        result_window.geometry("600x400")
                        result_window.transient(progress_dialog)
                        
                        result_label = ttk.Label(result_window, 
                                                text=f"{len(garbled_files)}個の文字化けファイルが見つかりました",
                                                font=("", 10, "bold"))
                        result_label.pack(pady=10)
                        
                        result_text = scrolledtext.ScrolledText(result_window, height=15, width=70)
                        result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
                        
                        for file_path in garbled_files:
                            result_text.insert(tk.END, f"{file_path}\n")
                        
                        result_text.config(state=tk.DISABLED)
                        
                        def fix_files():
                            result_window.destroy()
                            progress_dialog.destroy()
                            self.fix_garbled_files()
                        
                        fix_btn = ttk.Button(result_window, text="修正する", command=fix_files)
                        fix_btn.pack(pady=10)
                        
                        close_result_btn = ttk.Button(result_window, text="閉じる", 
                                                      command=result_window.destroy)
                        close_result_btn.pack(pady=5)
                        
                        messagebox.showinfo("検出完了", 
                            f"{len(garbled_files)}個の文字化けファイルが見つかりました。\n\n"
                            "結果ウィンドウで詳細を確認できます。")
                    else:
                        log("文字化けファイルは見つかりませんでした。")
                        log("すべてのファイルは正常なエンコーディングです。")
                        messagebox.showinfo("検出完了", "文字化けファイルは見つかりませんでした。")
                    
                except Exception as e:
                    log(f"エラー: {e}")
                    import traceback
                    log(traceback.format_exc())
                    messagebox.showerror("Error", f"エラーが発生しました: {e}")
            
            # Run scan in background thread
            threading.Thread(target=scan_files, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"エラーが発生しました: {e}")
    
    def update_commit_messages(self):
        """Update commit messages for files with descriptive messages"""
        try:
            os.chdir(self.project_path)
            
            # File descriptions mapping
            file_descriptions = {
                # Folders
                '.github/workflows': 'GitHub Actions ワークフロー設定',
                'ai_learning_analysis': 'AI学習データ分析モジュール',
                'backup': 'バックアップファイル',
                'batch_files': 'バッチファイル集',
                'docs': 'ドキュメント',
                'github_tools': 'GitHub連携ツール',
                'old': '旧ファイル',
                'resin_washer_model': '樹脂ワッシャー検査モデル',
                'scripts': 'スクリプト集',
                
                # Main Files (改良版)
                'main.py': 'メイン検査システム',
                'camera_inspection.py': 'カメラ検査システム（改良版）',
                'scripts/train_4class_sparse_ensemble.py': '4クラス分類学習スクリプト（改良版）',
                'auto-commit.ps1': '自動コミットスクリプト（改良版）',
                
                # Files
                '.gitignore': 'Git除外設定ファイル',
                '.test_encoding_check.txt': 'エンコーディング確認テストファイル',
                'AUTO_COMMIT_README.md': '自動コミット機能の説明',
                'CHANGE_HISTORY_VIEWER_README.md': '変更履歴ビューアーの説明',
                'COMMIT_FIX_GUIDE.md': 'コミット修正ガイド',
                'CURSOR_GITHUB_GUIDE.md': 'CursorとGitHub連携ガイド',
                'GITHUB_AUTO_COMMIT_GUIDE.md': 'GitHub自動コミットガイド',
                'NETWORK_SETUP_GUIDE.md': 'ネットワーク設定ガイド',
                'README.md': 'プロジェクト説明書',
                'change_history_viewer.py': '変更履歴ビューアーアプリ',
                'test_auto_commit.ps1': '自動コミットテストスクリプト',
                'start_history_viewer.bat': '履歴ビューアー起動バッチ',
                'fix_git_encoding.md': 'Gitエンコーディング修正ガイド',
                'organize_duplicate_files.py': '重複ファイル整理スクリプト',
                
                # Duplicate files (old versions - will be moved to old/duplicate_files/)
                'all_cameras_inspection.py': '全カメラ検査システム（旧版）',
                'camera_selector_inspection.py': 'カメラ選択システム（旧版）',
                'fundamental_fix_inspection.py': '基礎修正検査システム（旧版）',
                'fixed_inspection.py': '修正済み検査システム（旧版）',
                'perfect_focus_restored.py': 'ピント復元検査システム（旧版）',
                'focus_enhanced_inspection.py': 'ピント強化検査システム（旧版）',
                'high_accuracy_inspection.py': '高精度検査システム（旧版）',
                'network_washer_inspection_system.py': 'ネットワーク検査システム（旧版）',
                'enhanced_trainer.py': '強化学習スクリプト（旧版）',
                'improved_trainer.py': '改良学習スクリプト（旧版）',
                'high_accuracy_trainer.py': '高精度学習スクリプト（旧版）',
            }
            
            # Use common progress dialog
            progress_dialog, log, close_btn = self.create_progress_dialog("コミットメッセージ更新", width=700, height=600)
            
            def process_files():
                try:
                    log("=" * 60)
                    log("コミットメッセージを更新中...")
                    log("")
                    
                    # Get all tracked files
                    cmd = ["git", "ls-files"]
                    result = subprocess.run(cmd, capture_output=True, text=True, 
                                          encoding='utf-8', errors='replace')
                    
                    if result.returncode != 0:
                        log("✗ エラー: Gitリポジトリではありません")
                        return
                    
                    files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
                    
                    # Get directories too
                    directories = set()
                    for file in files:
                        dir_path = os.path.dirname(file) if os.path.dirname(file) else '.'
                        if dir_path != '.':
                            # Add all parent directories
                            parts = dir_path.split(os.sep)
                            for i in range(len(parts)):
                                directories.add(os.sep.join(parts[:i+1]) if i > 0 else parts[0])
                    
                    # Combine files and directories
                    all_items = list(set(files) | directories)
                    all_items.sort()
                    
                    log(f"処理対象: {len(all_items)}個のファイル/フォルダ")
                    log("")
                    
                    updated_count = 0
                    
                    # Process each file/directory individually
                    for item_path in all_items:
                        # Get description
                        description = file_descriptions.get(item_path, None)
                        
                        # If no description, generate one
                        if not description:
                            if item_path == '.':
                                description = 'プロジェクトファイル'
                            elif os.path.isdir(item_path) if os.path.exists(item_path) else item_path.endswith('/'):
                                # It's a directory
                                dir_name = os.path.basename(item_path) if item_path != '.' else 'プロジェクト'
                                description = f'{dir_name}フォルダ'
                            else:
                                # It's a file
                                file_name = os.path.basename(item_path)
                                name_without_ext = os.path.splitext(file_name)[0]
                                description = f'{name_without_ext}ファイル'
                        
                        # Check if this is a file or directory
                        is_file = os.path.isfile(item_path) if os.path.exists(item_path) else item_path in files
                        
                        if is_file:
                            # Process individual file
                            log(f"処理中: {item_path}")
                            log(f"  説明: {description}")
                            
                            try:
                                cmd = ["git", "add", item_path]
                                result = subprocess.run(cmd, capture_output=True, text=True, 
                                                      encoding='utf-8', errors='replace')
                                if result.returncode == 0:
                                    log(f"  ✓ ステージング完了")
                                else:
                                    log(f"  ✗ ステージング失敗: {result.stderr}")
                                    continue
                            except Exception as e:
                                log(f"  ✗ エラー: {e}")
                                continue
                            
                            # Check current commit message for this file
                            cmd = ["git", "log", "-1", "--format=%s", "--", item_path]
                            current_msg_result = subprocess.run(cmd, capture_output=True, text=True, 
                                                               encoding='utf-8', errors='replace')
                            current_message = current_msg_result.stdout.strip() if current_msg_result.returncode == 0 else ""
                            
                            # If message is already correct, skip
                            if current_message == description:
                                log(f"  → コミットメッセージは既に正しい（スキップ）")
                                continue
                            
                            # Commit with description (allow empty if no changes)
                            commit_message = description
                            cmd = ["git", "commit", "--allow-empty", "-m", commit_message]
                            result = subprocess.run(cmd, capture_output=True, text=True, 
                                                  encoding='utf-8', errors='replace')
                            
                            if result.returncode == 0:
                                log(f"  → コミット成功: {commit_message}")
                                updated_count += 1
                            else:
                                # Check if there are changes to commit
                                cmd = ["git", "status", "--porcelain", item_path]
                                status_result = subprocess.run(cmd, capture_output=True, text=True, 
                                                             encoding='utf-8', errors='replace')
                                if not status_result.stdout.strip():
                                    # No changes, try with --allow-empty
                                    cmd = ["git", "commit", "--allow-empty", "-m", commit_message]
                                    result = subprocess.run(cmd, capture_output=True, text=True, 
                                                          encoding='utf-8', errors='replace')
                                    if result.returncode == 0:
                                        log(f"  → コミット成功（空コミット）: {commit_message}")
                                        updated_count += 1
                                    else:
                                        log(f"  → 変更なし（スキップ）")
                                else:
                                    log(f"  ✗ コミット失敗: {result.stderr}")
                            
                            log("")
                        else:
                            # Process directory - get all files in this directory
                            dir_files = [f for f in files if f.startswith(item_path + os.sep) or f == item_path]
                            
                            if dir_files:
                                log(f"処理中: {item_path}/")
                                log(f"  説明: {description}")
                                
                                for file in dir_files:
                                    try:
                                        cmd = ["git", "add", file]
                                        result = subprocess.run(cmd, capture_output=True, text=True, 
                                                              encoding='utf-8', errors='replace')
                                        if result.returncode == 0:
                                            log(f"  ✓ {os.path.basename(file)}")
                                        else:
                                            log(f"  ✗ {os.path.basename(file)}: {result.stderr}")
                                    except Exception as e:
                                        log(f"  ✗ {os.path.basename(file)}: {e}")
                                
                                # Check current commit message for this directory
                                cmd = ["git", "log", "-1", "--format=%s", "--", item_path]
                                current_msg_result = subprocess.run(cmd, capture_output=True, text=True, 
                                                                   encoding='utf-8', errors='replace')
                                current_message = current_msg_result.stdout.strip() if current_msg_result.returncode == 0 else ""
                                
                                # If message is already correct, skip
                                if current_message == description:
                                    log(f"  → コミットメッセージは既に正しい（スキップ）")
                                    log("")
                                    continue
                                
                                # Commit with description (allow empty if no changes)
                                commit_message = description
                                cmd = ["git", "commit", "--allow-empty", "-m", commit_message]
                                result = subprocess.run(cmd, capture_output=True, text=True, 
                                                      encoding='utf-8', errors='replace')
                                
                                if result.returncode == 0:
                                    log(f"  → コミット成功: {commit_message}")
                                    updated_count += 1
                                else:
                                    # Check if there are changes to commit
                                    cmd = ["git", "status", "--porcelain"]
                                    status_result = subprocess.run(cmd, capture_output=True, text=True, 
                                                                 encoding='utf-8', errors='replace')
                                    if not status_result.stdout.strip():
                                        # No changes, try with --allow-empty
                                        cmd = ["git", "commit", "--allow-empty", "-m", commit_message]
                                        result = subprocess.run(cmd, capture_output=True, text=True, 
                                                              encoding='utf-8', errors='replace')
                                        if result.returncode == 0:
                                            log(f"  → コミット成功（空コミット）: {commit_message}")
                                            updated_count += 1
                                        else:
                                            log(f"  → 変更なし（スキップ）")
                                    else:
                                        log(f"  ✗ コミット失敗: {result.stderr}")
                                
                                log("")
                    
                    log("")
                    log("=" * 60)
                    log(f"更新完了: {updated_count}個のコミットを作成")
                    log("")
                    
                    if updated_count > 0:
                        log("⚠️ 次のステップ:")
                        log("GitHubにプッシュしますか？")
                        log("")
                        
                        # Ask user if they want to push
                        progress_dialog.update()
                        response = messagebox.askyesno("確認", 
                            f"{updated_count}個のコミットを作成しました。\n\n"
                            "GitHubにプッシュしますか？")
                        
                        if response:
                            log("")
                            log("プッシュ中...")
                            cmd = ["git", "push", "origin", "master"]
                            result = subprocess.run(cmd, capture_output=True, text=True, 
                                                  encoding='utf-8', errors='replace')
                            
                            if result.returncode == 0:
                                log("✓ プッシュ成功")
                                messagebox.showinfo("成功", "コミットメッセージを更新してプッシュしました。")
                            else:
                                log(f"✗ プッシュ失敗: {result.stderr}")
                                messagebox.showerror("エラー", f"プッシュに失敗しました:\n{result.stderr}")
                    else:
                        log("更新するコミットはありませんでした。")
                        messagebox.showinfo("情報", "更新するコミットはありませんでした。")
                    
                    # Add close button
                    close_btn = ttk.Button(progress_dialog, text="閉じる", 
                                          command=progress_dialog.destroy)
                    close_btn.pack(pady=10)
                    
                except Exception as e:
                    log(f"エラー: {e}")
                    import traceback
                    log(traceback.format_exc())
                    messagebox.showerror("Error", f"エラーが発生しました: {e}")
                    progress_dialog.destroy()
            
            # Run in background thread
            threading.Thread(target=process_files, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"エラーが発生しました: {e}")

def main():
    root = tk.Tk()
    app = ChangeHistoryViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()

