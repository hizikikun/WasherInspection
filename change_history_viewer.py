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

        

        fix_messages_btn = ttk.Button(commit_frame, text="コミットメッセージ修正", command=self.fix_commit_messages)

        fix_messages_btn.pack(side=tk.LEFT, padx=(0, 5))

        

        # Auto-commit toggle

        self.auto_commit_var = tk.BooleanVar(value=self.auto_commit_enabled)

        auto_commit_check = ttk.Checkbutton(commit_frame, text="自動コミット有効", 

                                            variable=self.auto_commit_var,

                                            command=self.toggle_auto_commit)

        auto_commit_check.pack(side=tk.LEFT, padx=(5, 0))

        

        

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

    

    def fix_commit_messages(self):

        """Fix incorrect commit messages that were set by previous garbled text fix features"""

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

                

                # Main Files

                'main.py': 'メイン検査システム',

                'camera_inspection.py': 'カメラ検査システム',

                'scripts/train_4class_sparse_ensemble.py': '4クラス分類学習スクリプト',

                'auto-commit.ps1': '自動コミットスクリプト',

                

                # Documentation files

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

                

                # JSON files

                'ai_learning_analysis/dataset_comparison_report.json': 'データセット比較レポート',

                'ai_learning_analysis/基本データセット_analysis_report.json': '基本データセット分析レポート',

                'ai_learning_analysis/実際の学習データ_analysis_report.json': '実際の学習データ分析レポート',

                'ai_learning_analysis/改良データセット_analysis_report.json': '改良データセット分析レポート',

                'feedback_data.json': 'フィードバックデータ',

                'resin_washer_model/training_history.json': '学習履歴データ',

            }

            

            # Use common progress dialog

            progress_dialog, log, close_btn = self.create_progress_dialog("コミットメッセージ修正", width=800, height=700)

            

            def process_files():

                try:

                    log("=" * 60)

                    log("不適切なコミットメッセージを修正中...")

                    log("")

                    

                    # Get all tracked files

                    cmd = ["git", "ls-files"]

                    result = subprocess.run(cmd, capture_output=True, text=True, 

                                          encoding='utf-8', errors='replace')

                    

                    if result.returncode != 0:

                        log("✗ エラー: Gitリポジトリではありません")

                        return

                    

                    files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]

                    

                    # Filter out files that are ignored by .gitignore

                    # git ls-files returns tracked files, but some might be ignored after being tracked

                    # Check each file to make sure it's not ignored

                    tracked_files = []

                    for file in files:

                        cmd_check = ["git", "check-ignore", "-q", file]

                        result_check = subprocess.run(cmd_check, capture_output=True, text=True, 

                                                     encoding='utf-8', errors='replace')

                        # git check-ignore returns 0 if ignored, non-zero if not ignored

                        if result_check.returncode != 0:

                            tracked_files.append(file)

                    

                    files = tracked_files

                    

                    # Get unique directories from tracked files only

                    directories = set()

                    for file in files:

                        dir_path = os.path.dirname(file) if os.path.dirname(file) else '.'

                        if dir_path != '.':

                            parts = dir_path.split(os.sep)

                            for i in range(len(parts)):

                                dir_part = os.sep.join(parts[:i+1]) if i > 0 else parts[0]

                                # Only add if this directory contains tracked files

                                if any(f.startswith(dir_part + os.sep) or f == dir_part for f in files):

                                    directories.add(dir_part)

                    

                    # Combine files and directories, but prioritize files

                    all_items = files + [d for d in sorted(directories) if d not in files]

                    all_items.sort()

                    

                    log(f"処理対象: {len(all_items)}個のファイル/フォルダ")

                    log("")

                    

                    # Patterns to identify incorrect messages

                    incorrect_patterns = [

                        "Fix: 全ファイルをUTF-8エンコーディングに統一",

                        "Fix: 全ファイルをUTF-8",

                        "UTF-8エンコーディングに統一",

                    ]

                    

                    fixed_count = 0

                    skipped_count = 0

                    

                    for item_path in all_items:

                        try:

                            # Get description

                            description = file_descriptions.get(item_path, None)

                            

                            # If no description, generate one

                            if not description:

                                if os.path.isdir(item_path) if os.path.exists(item_path) else item_path.endswith('/'):

                                    dir_name = os.path.basename(item_path) if item_path != '.' else 'プロジェクト'

                                    description = f'{dir_name}フォルダ'

                                else:

                                    file_name = os.path.basename(item_path)

                                    name_without_ext = os.path.splitext(file_name)[0]

                                    description = f'{name_without_ext}ファイル'

                            

                            # Get current commit message for this item

                            cmd = ["git", "log", "-1", "--format=%s", "--", item_path]

                            result = subprocess.run(cmd, capture_output=True, text=True, 

                                                  encoding='utf-8', errors='replace')

                            

                            if result.returncode != 0:

                                continue

                            

                            current_message = result.stdout.strip()

                            

                            # Check if message is incorrect

                            is_incorrect = any(pattern in current_message for pattern in incorrect_patterns)

                            

                            # Also check if message doesn't match description

                            if not is_incorrect and current_message != description:

                                # Skip if message is already correct or different but not incorrect

                                skipped_count += 1

                                continue

                            

                            if is_incorrect or current_message != description:

                                log(f"修正中: {item_path}")

                                log(f"  現在: {current_message[:60]}...")

                                log(f"  新しい: {description}")

                                

                                # Check if item exists or is tracked by Git

                                is_tracked = item_path in files or any(f.startswith(item_path + os.sep) for f in files)

                                

                                if not is_tracked and not os.path.exists(item_path):

                                    log(f"  → スキップ: ファイル/フォルダが存在しないか、Gitで管理されていません")

                                    skipped_count += 1

                                    continue

                                

                                # Check if item is ignored by .gitignore

                                cmd_check = ["git", "check-ignore", "-q", item_path]

                                result_check = subprocess.run(cmd_check, capture_output=True, text=True, 

                                                           encoding='utf-8', errors='replace')

                                

                                # git check-ignore returns 0 if the path is ignored

                                if result_check.returncode == 0:

                                    log(f"  → スキップ: .gitignoreによって無視されています")

                                    skipped_count += 1

                                    continue

                                

                                # Stage the item (only if it's a file or directory with tracked files)

                                if item_path in files:

                                    # It's a tracked file - check if it's ignored

                                    cmd_check_file = ["git", "check-ignore", "-q", item_path]

                                    result_check_file = subprocess.run(cmd_check_file, capture_output=True, text=True, 

                                                                     encoding='utf-8', errors='replace')

                                    if result_check_file.returncode == 0:

                                        log(f"  → スキップ: .gitignoreによって無視されています")

                                        skipped_count += 1

                                        continue

                                    cmd = ["git", "add", item_path]

                                else:

                                    # It's a directory - stage all files in it

                                    dir_files = [f for f in files if f.startswith(item_path + os.sep) or f == item_path]

                                    if not dir_files:

                                        log(f"  → スキップ: ディレクトリ内にGitで管理されているファイルがありません")

                                        skipped_count += 1

                                        continue

                                    

                                    # Filter out ignored files from dir_files

                                    tracked_dir_files = []

                                    for f in dir_files:

                                        cmd_check_file = ["git", "check-ignore", "-q", f]

                                        result_check_file = subprocess.run(cmd_check_file, capture_output=True, text=True, 

                                                                         encoding='utf-8', errors='replace')

                                        if result_check_file.returncode != 0:  # Not ignored

                                            tracked_dir_files.append(f)

                                    

                                    if not tracked_dir_files:

                                        log(f"  → スキップ: ディレクトリ内のファイルがすべて.gitignoreで無視されています")

                                        skipped_count += 1

                                        continue

                                    

                                    cmd = ["git", "add"] + tracked_dir_files

                                

                                result = subprocess.run(cmd, capture_output=True, text=True, 

                                                      encoding='utf-8', errors='replace')

                                

                                if result.returncode != 0:

                                    # Check if it's an ignore error

                                    if "ignored by" in result.stderr.lower() or "gitignore" in result.stderr.lower():

                                        log(f"  → スキップ: .gitignoreによって無視されています")

                                    else:

                                        log(f"  ✗ ステージング失敗: {result.stderr[:100] if result.stderr else '原因不明'}")

                                    skipped_count += 1

                                    continue

                                

                                # Make a small change to the file to ensure GitHub updates the "Last commit message"
                                # Add or remove a trailing newline to make an actual file change
                                file_changed = False
                                
                                if item_path in files and os.path.isfile(item_path):
                                    try:
                                        # Read the file
                                        with open(item_path, 'rb') as f:
                                            content = f.read()
                                        
                                        # Try to decode as text
                                        try:
                                            text_content = content.decode('utf-8')
                                            
                                            # Check if file ends with newline
                                            ends_with_newline = text_content.endswith('\n')
                                            
                                            # Add or remove trailing newline
                                            # This works for all text files including JSON
                                            if ends_with_newline:
                                                # Remove trailing newline
                                                new_content = text_content.rstrip('\n')
                                            else:
                                                # Add trailing newline
                                                new_content = text_content + '\n'
                                            
                                            # Only write if content actually changed
                                            if new_content != text_content:
                                                with open(item_path, 'w', encoding='utf-8', newline='') as f:
                                                    f.write(new_content)
                                                file_changed = True
                                                log(f"  → ファイルを変更しました（末尾の改行を調整）")
                                        except UnicodeDecodeError:
                                            # Binary file - skip modification
                                            log(f"  → バイナリファイルのため変更をスキップ")
                                    except Exception as e:
                                        log(f"  → ファイル変更エラー: {e}")
                                
                                # If it's a directory, modify a file inside it
                                elif item_path not in files:
                                    # Find first text file in directory
                                    dir_files = [f for f in tracked_dir_files if os.path.isfile(f)]
                                    for dir_file in dir_files[:1]:  # Only modify first file
                                        try:
                                            with open(dir_file, 'rb') as f:
                                                content = f.read()
                                            
                                            try:
                                                text_content = content.decode('utf-8')
                                                ends_with_newline = text_content.endswith('\n')
                                                
                                                if ends_with_newline:
                                                    new_content = text_content.rstrip('\n')
                                                else:
                                                    new_content = text_content + '\n'
                                                
                                                if new_content != text_content:
                                                    with open(dir_file, 'w', encoding='utf-8', newline='') as f:
                                                        f.write(new_content)
                                                    file_changed = True
                                                    log(f"  → {dir_file}を変更しました（末尾の改行を調整）")
                                                    break
                                            except UnicodeDecodeError:
                                                continue
                                        except Exception as e:
                                            continue

                                # Stage the changes again (in case file was modified)
                                if file_changed:
                                    if item_path in files:
                                        cmd = ["git", "add", item_path]
                                    else:
                                        cmd = ["git", "add"] + tracked_dir_files
                                    
                                    subprocess.run(cmd, capture_output=True, text=True,
                                                 encoding='utf-8', errors='replace')

                                # Create commit with description
                                env = os.environ.copy()

                                env['LANG'] = 'en_US.UTF-8'

                                env['LC_ALL'] = 'en_US.UTF-8'

                                

                                cmd = ["git", "-c", "i18n.commitEncoding=utf-8", "commit", 

                                       "-m", description]

                                result = subprocess.run(cmd, capture_output=True, text=True, 

                                                      encoding='utf-8', errors='replace', env=env)

                                

                                if result.returncode == 0:

                                    log(f"  ✓ 修正成功")

                                    fixed_count += 1

                                else:

                                    log(f"  ✗ 修正失敗: {result.stderr[:50]}")

                                

                                log("")

                        except Exception as e:

                            log(f"✗ {item_path}: エラー - {e}")

                    

                    log("")

                    log("=" * 60)

                    log(f"処理完了")

                    log(f"修正成功: {fixed_count}個")

                    log(f"スキップ: {skipped_count}個")

                    log("")

                    

                    if fixed_count > 0:

                        log("")

                        log("=" * 60)

                        log("修正完了。GitHubにプッシュします...")

                        log("")

                        

                        # Automatically push to GitHub

                        env = os.environ.copy()

                        env['LANG'] = 'en_US.UTF-8'

                        env['LC_ALL'] = 'en_US.UTF-8'

                        

                        # Try to push (force push may be needed)

                        log("プッシュ中...")

                        progress_dialog.update()

                        

                        success, error = self.git_push("master", force=True, log_func=log)

                        

                        if success:

                            log("")

                            log("=" * 60)

                            log("✓ プッシュ成功！")

                            log(f"{fixed_count}個のコミットメッセージを修正してGitHubにプッシュしました。")

                            messagebox.showinfo("成功", 

                                f"{fixed_count}個のコミットメッセージを修正してGitHubにプッシュしました。")

                        else:

                            log("")

                            log("=" * 60)

                            log("✗ プッシュ失敗")

                            log(f"エラー: {error}")

                            log("")

                            log("手動でプッシュしてください:")

                            log("  git push origin master --force")

                            messagebox.showwarning("警告", 

                                f"プッシュに失敗しました:\n{error}\n\n"

                                "手動でプッシュしてください:\n"

                                "git push origin master --force")

                    else:

                        log("修正するコミットはありませんでした。")

                        messagebox.showinfo("情報", "修正するコミットはありませんでした。")

                    

                except Exception as e:

                    log(f"エラー: {e}")

                    import traceback

                    log(traceback.format_exc())

                    messagebox.showerror("Error", f"エラーが発生しました: {e}")

            

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



