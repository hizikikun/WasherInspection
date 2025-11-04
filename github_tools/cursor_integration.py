#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cursor-Integrated GitHub Auto-Sync System
- Seamless GitHub integration within Cursor IDE
- One-click send/receive functionality
- Automatic background synchronization
- No browser required - everything in Cursor
"""

import os
import json
import time
import requests
import base64
import threading
from datetime import datetime
from pathlib import Path
import webbrowser
import subprocess
import sys

class CursorGitHubIntegration:
    def __init__(self, config_file="cursor_github_config.json"):
        """
        Initialize Cursor-GitHub integration
        """
        self.config = self.load_config(config_file)
        self.github_token = self.config.get('github_token')
        
        # Auto-detect GitHub owner from token
        self.github_owner = self.config.get('github_owner')
        if self.github_token:
            actual_owner = self.get_authenticated_user()
            if actual_owner:
                if self.github_owner != actual_owner:
                    print(f"[INFO] Updating github_owner from '{self.github_owner}' to '{actual_owner}'")
                    self.github_owner = actual_owner
                    self.config['github_owner'] = actual_owner
                    self.save_config()
                else:
                    self.github_owner = actual_owner
            else:
                print(f"[WARNING] Could not detect GitHub user from token. Using configured owner: {self.github_owner}")
        else:
            print(f"[WARNING] No GitHub token found. Using configured owner: {self.github_owner}")
        
        self.github_repo = self.config.get('github_repo')
        
        # Auto-sync settings
        self.auto_sync_enabled = self.config.get('auto_sync_enabled', True)
        self.sync_interval = self.config.get('sync_interval', 60)  # seconds
        self.auto_commit_threshold = self.config.get('auto_commit_threshold', 3)
        
        # Background sync
        self.sync_thread = None
        self.running = False
        
        # File monitoring
        self.last_sync_time = time.time()
        self.pending_changes = []
        
        print(f"[CURSOR-GITHUB] Repository: {self.github_owner}/{self.github_repo}")
        print(f"[CURSOR-GITHUB] Auto-sync: {'Enabled' if self.auto_sync_enabled else 'Disabled'}")
        print(f"[CURSOR-GITHUB] Sync interval: {self.sync_interval}s")
    
    def get_authenticated_user(self):
        """Get authenticated GitHub user from token"""
        if not self.github_token:
            return None
        
        try:
            url = "https://api.github.com/user"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                user_info = response.json()
                return user_info.get('login')
            else:
                print(f"[ERROR] Failed to get user info: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Error getting authenticated user: {e}")
            return None
    
    def load_config(self, config_file):
        """Load configuration"""
        default_config = {
            "github_token": "",
            "github_owner": "your-username",
            "github_repo": "washer-inspection-code",
            "auto_sync_enabled": True,
            "sync_interval": 60,
            "auto_commit_threshold": 3,
            "one_click_enabled": True,
            "notifications_enabled": True,
            "include_files": [".py", ".json", ".md", ".yml", ".yaml", ".txt"],
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
            print(f"Created config file: {config_file}")
            return default_config
    
    def setup_github_auth(self):
        """Setup GitHub authentication automatically"""
        if self.github_token:
            print("[AUTH] GitHub token already configured")
            return True
        
        print("[AUTH] Setting up GitHub authentication...")
        
        # Try to get token from environment
        token = os.environ.get('GITHUB_TOKEN')
        if token:
            self.github_token = token
            self.config['github_token'] = token
            self.save_config()
            print("[AUTH] Token loaded from environment")
            return True
        
        # Try to get token from git config
        try:
            result = subprocess.run(['git', 'config', '--global', 'github.token'], 
                                  capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                self.github_token = result.stdout.strip()
                self.config['github_token'] = self.github_token
                self.save_config()
                print("[AUTH] Token loaded from git config")
                return True
        except:
            pass
        
        # Open browser for token creation
        print("[AUTH] Opening GitHub token creation page...")
        token_url = "https://github.com/settings/tokens/new"
        webbrowser.open(token_url)
        
        print("\n" + "="*60)
        print("GitHub Token Setup Instructions:")
        print("1. In the opened browser, create a new Personal Access Token")
        print("2. Select 'repo' scope for full repository access")
        print("3. Copy the generated token")
        print("4. Run: python cursor_github_integration.py --token YOUR_TOKEN")
        print("="*60)
        
        return False
    
    def save_config(self):
        """Save configuration"""
        with open("cursor_github_config.json", 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def verify_repository_access(self):
        """Verify repository exists and token has access"""
        if not self.github_token:
            return False, "GitHub token not configured"
        
        try:
            # Check repository exists
            repo_url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.get(repo_url, headers=headers, timeout=10)
            
            if response.status_code == 404:
                # Try to create repository automatically
                create_result = self.create_repository()
                if create_result[0]:
                    print(f"[INFO] Repository '{self.github_owner}/{self.github_repo}' created successfully")
                    # Verify again after creation
                    response = requests.get(repo_url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        return True, "Repository created and verified"
                return False, f"Repository '{self.github_owner}/{self.github_repo}' not found. Attempted auto-creation but failed: {create_result[1]}"
            elif response.status_code == 403:
                return False, "Access denied. Check token permissions (needs 'repo' scope)."
            elif response.status_code != 200:
                error_msg = response.json().get('message', 'Unknown error') if response.content else 'Unknown error'
                return False, f"Repository access failed: {response.status_code} - {error_msg}"
            
            return True, "Repository access verified"
            
        except requests.exceptions.RequestException as e:
            return False, f"Network error: {e}"
        except Exception as e:
            return False, f"Error verifying repository: {e}"
    
    def create_repository(self):
        """Create GitHub repository if it doesn't exist"""
        try:
            # Check if owner is organization or user
            user_url = f"https://api.github.com/user"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Get authenticated user info
            user_response = requests.get(user_url, headers=headers, timeout=10)
            if user_response.status_code != 200:
                return False, "Cannot authenticate. Check token."
            
            user_info = user_response.json()
            authenticated_user = user_info.get('login')
            
            # Create repository
            create_repo_url = f"https://api.github.com/user/repos"
            if self.github_owner != authenticated_user:
                # Organization repository
                create_repo_url = f"https://api.github.com/orgs/{self.github_owner}/repos"
            
            repo_data = {
                "name": self.github_repo,
                "description": "Auto-created by Cursor GitHub Integration",
                "private": False,  # Public repository
                "auto_init": True  # Initialize with README
            }
            
            create_response = requests.post(create_repo_url, headers=headers, json=repo_data, timeout=30)
            
            if create_response.status_code in [201, 200]:
                return True, "Repository created successfully"
            else:
                error_msg = create_response.json().get('message', 'Unknown error') if create_response.content else 'Unknown error'
                return False, f"Failed to create repository: {create_response.status_code} - {error_msg}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Network error: {e}"
        except Exception as e:
            return False, f"Error creating repository: {e}"
    
    def get_default_branch(self):
        """Get default branch name (main or master)"""
        try:
            repo_url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.get(repo_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json().get('default_branch', 'main')
        except:
            pass
        
        # Fallback: try to determine from git
        try:
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip() or 'main'
        except:
            pass
        
        return 'main'  # Default fallback
    
    def one_click_send(self, message="Auto-sync from Cursor"):
        """One-click send to GitHub"""
        if not self.github_token:
            print("[ERROR] GitHub token not configured")
            return False
        
        # Verify repository access first
        access_ok, error_msg = self.verify_repository_access()
        if not access_ok:
            print(f"[ERROR] {error_msg}")
            return False
        
        try:
            # Get current changes
            changes = self.get_current_changes()
            
            if not changes:
                print("[SEND] No changes to send")
                return True
            
            # Use Contents API (simpler and more reliable)
            commit_message = f"{message} - {len(changes)} files"
            success = self.create_github_commit_contents_api(changes, commit_message)
            
            # Fallback to Git Trees API if Contents API fails
            if not success:
                print("[SEND] Trying alternative method...")
                success = self.create_github_commit(changes, commit_message)
            
            if success:
                print(f"[SEND] Successfully sent {len(changes)} files to GitHub")
                self.show_notification(f"[SUCCESS] Sent {len(changes)} files to GitHub")
                return True
            else:
                print("[SEND] Failed to send to GitHub. Check console for details.")
                self.show_notification("[ERROR] Failed to send to GitHub")
                return False
                
        except Exception as e:
            error_msg = str(e)
            print(f"[SEND] Error: {error_msg}")
            print(f"[SEND] Error type: {type(e).__name__}")
            import traceback
            print(f"[SEND] Traceback:\n{traceback.format_exc()}")
            self.show_notification(f"[ERROR] Error: {error_msg}")
            return False
    
    def one_click_receive(self):
        """One-click receive from GitHub"""
        if not self.github_token:
            print("[ERROR] GitHub token not configured")
            return False
        
        try:
            # Get latest commits
            commits = self.get_latest_commits()
            
            if not commits:
                print("[RECEIVE] No new commits to receive")
                return True
            
            # Download and apply changes
            for commit in commits:
                self.apply_commit(commit)
            
            print(f"[RECEIVE] Successfully received {len(commits)} commits")
            self.show_notification(f"[RECEIVE] Received {len(commits)} commits from GitHub")
            return True
            
        except Exception as e:
            print(f"[RECEIVE] Error: {e}")
            self.show_notification(f"[ERROR] Error: {e}")
            return False
    
    def get_current_changes(self):
        """Get current changes in the workspace"""
        changes = []
        
        for root, dirs, files in os.walk('.'):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.config.get('exclude_dirs', [])]
            
            for file in files:
                if any(file.endswith(ext) for ext in self.config.get('include_files', ['.py'])):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, '.')
                    
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()
                        
                        changes.append({
                            'path': relative_path,
                            'content': content,
                            'size': len(content)
                        })
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        
        return changes
    
    def create_github_commit_contents_api(self, changes, message):
        """Create commit using Contents API (simpler and more reliable)"""
        try:
            branch = self.get_default_branch()
            success_count = 0
            
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            for change in changes:
                file_path = change['path']
                content = change['content']
                content_base64 = base64.b64encode(content).decode('utf-8')
                
                # Get current file SHA if exists
                file_url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/contents/{file_path}"
                file_response = requests.get(f"{file_url}?ref={branch}", headers=headers, timeout=10)
                
                file_data = {
                    "message": f"{message} - {file_path}",
                    "content": content_base64,
                    "branch": branch
                }
                
                # If file exists, include SHA for update
                if file_response.status_code == 200:
                    file_data["sha"] = file_response.json()['sha']
                
                # Create or update file
                put_response = requests.put(file_url, headers=headers, json=file_data, timeout=30)
                
                if put_response.status_code in [200, 201]:
                    success_count += 1
                else:
                    error_msg = put_response.json().get('message', 'Unknown error') if put_response.content else 'No error message'
                    print(f"[ERROR] Failed to upload {file_path}: {put_response.status_code} - {error_msg}")
            
            if success_count == len(changes):
                return True
            elif success_count > 0:
                print(f"[PARTIAL] Uploaded {success_count}/{len(changes)} files")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"[ERROR] Contents API commit failed: {e}")
            return False
    
    def create_github_commit(self, changes, message):
        """Create commit on GitHub"""
        try:
            # Prepare commit data
            commit_data = {
                "message": message,
                "tree": None,
                "parents": []
            }
            
            # Get current HEAD
            try:
                result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    commit_data["parents"] = [result.stdout.strip()]
            except:
                pass
            
            # Create tree with all files
            tree_items = []
            for change in changes:
                content_base64 = base64.b64encode(change['content']).decode('utf-8')
                tree_items.append({
                    "path": change['path'],
                    "mode": "100644",
                    "type": "blob",
                    "content": content_base64
                })
            
            if not tree_items:
                return False
            
            # Create tree
            tree_url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/git/trees"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            tree_data = {
                "base_tree": commit_data["parents"][0] if commit_data["parents"] else None,
                "tree": tree_items
            }
            
            tree_response = requests.post(tree_url, headers=headers, json=tree_data)
            if tree_response.status_code != 201:
                error_msg = tree_response.json().get('message', 'Unknown error') if tree_response.content else 'No error message'
                print(f"[ERROR] Failed to create tree: {tree_response.status_code} - {error_msg}")
                print(f"[ERROR] Repository: {self.github_owner}/{self.github_repo}")
                return False
            
            tree_sha = tree_response.json()['sha']
            commit_data["tree"] = tree_sha
            
            # Create commit
            commit_url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/git/commits"
            commit_response = requests.post(commit_url, headers=headers, json=commit_data)
            
            if commit_response.status_code != 201:
                print(f"Failed to create commit: {commit_response.status_code}")
                return False
            
            commit_sha = commit_response.json()['sha']
            
            # Update branch reference (use dynamic branch name)
            branch = self.get_default_branch()
            ref_url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/git/refs/heads/{branch}"
            ref_data = {"sha": commit_sha}
            
            ref_response = requests.patch(ref_url, headers=headers, json=ref_data)
            if ref_response.status_code != 200:
                # Try to create branch if it doesn't exist
                if ref_response.status_code == 404:
                    # Get base SHA from default branch or main
                    base_ref_response = requests.get(
                        f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/git/refs/heads/main",
                        headers=headers
                    )
                    if base_ref_response.status_code != 200:
                        print(f"[ERROR] Failed to create branch: {branch} - base branch not found")
                        return False
                    
                    base_sha = base_ref_response.json()['object']['sha']
                    create_ref_data = {
                        "ref": f"refs/heads/{branch}",
                        "sha": commit_sha
                    }
                    create_ref_response = requests.post(
                        f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/git/refs",
                        headers=headers,
                        json=create_ref_data
                    )
                    if create_ref_response.status_code not in [200, 201]:
                        print(f"[ERROR] Failed to create branch: {create_ref_response.status_code}")
                        return False
                else:
                    error_msg = ref_response.json().get('message', 'Unknown error') if ref_response.content else 'No error message'
                    print(f"[ERROR] Failed to update branch: {ref_response.status_code} - {error_msg}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error creating commit: {e}")
            return False
    
    def get_latest_commits(self):
        """Get latest commits from GitHub"""
        try:
            url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/commits"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                error_msg = response.json().get('message', 'Unknown error') if response.content else 'No error message'
                print(f"[ERROR] Failed to get commits: {response.status_code} - {error_msg}")
                print(f"[ERROR] Repository: {self.github_owner}/{self.github_repo}")
                return []
            
            commits = response.json()
            return commits[:5]  # Return last 5 commits
            
        except Exception as e:
            print(f"Error getting commits: {e}")
            return []
    
    def apply_commit(self, commit):
        """Apply commit to local workspace"""
        try:
            commit_sha = commit['sha']
            commit_url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/git/commits/{commit_sha}"
            
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.get(commit_url, headers=headers)
            if response.status_code != 200:
                print(f"Failed to get commit details: {response.status_code}")
                return False
            
            commit_data = response.json()
            tree_sha = commit_data['tree']['sha']
            
            # Get tree contents
            tree_url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/git/trees/{tree_sha}"
            tree_response = requests.get(tree_url, headers=headers)
            
            if tree_response.status_code != 200:
                print(f"Failed to get tree: {tree_response.status_code}")
                return False
            
            tree_data = tree_response.json()
            
            # Apply files
            for item in tree_data.get('tree', []):
                if item['type'] == 'blob':
                    file_path = item['path']
                    blob_url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/git/blobs/{item['sha']}"
                    
                    blob_response = requests.get(blob_url, headers=headers)
                    if blob_response.status_code == 200:
                        blob_data = blob_response.json()
                        content = base64.b64decode(blob_data['content']).decode('utf-8')
                        
                        # Create directory if needed
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        
                        # Write file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        print(f"Applied: {file_path}")
            
            return True
            
        except Exception as e:
            print(f"Error applying commit: {e}")
            return False
    
    def show_notification(self, message):
        """Show notification in Cursor"""
        if not self.config.get('notifications_enabled', True):
            return
        
        print(f"\n[NOTIFICATION] {message}")
        
        # Try to show system notification
        try:
            if sys.platform == "win32":
                # Windows notification
                import ctypes
                ctypes.windll.user32.MessageBoxW(0, message, "Cursor GitHub Sync", 0x40)
            elif sys.platform == "darwin":
                # macOS notification
                subprocess.run(['osascript', '-e', f'display notification "{message}" with title "Cursor GitHub Sync"'])
            else:
                # Linux notification
                subprocess.run(['notify-send', 'Cursor GitHub Sync', message])
        except:
            pass
    
    def auto_sync_worker(self):
        """Background auto-sync worker"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check if enough time has passed
                if current_time - self.last_sync_time >= self.sync_interval:
                    # Check for changes
                    changes = self.get_current_changes()
                    
                    if len(changes) >= self.auto_commit_threshold:
                        print(f"[AUTO-SYNC] Found {len(changes)} changes, auto-sending...")
                        self.one_click_send("Auto-sync from Cursor")
                        self.last_sync_time = current_time
                    
                    # Check for incoming changes
                    commits = self.get_latest_commits()
                    if commits:
                        print(f"[AUTO-SYNC] Found {len(commits)} new commits, auto-receiving...")
                        self.one_click_receive()
                
                # Wait before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"[AUTO-SYNC] Error: {e}")
                time.sleep(60)
    
    def start_auto_sync(self):
        """Start automatic synchronization"""
        if not self.auto_sync_enabled:
            print("[AUTO-SYNC] Auto-sync is disabled")
            return
        
        if not self.github_token:
            print("[AUTO-SYNC] GitHub token not configured")
            return
        
        # Verify repository access before starting
        access_ok, error_msg = self.verify_repository_access()
        if not access_ok:
            print(f"[AUTO-SYNC] Cannot start: {error_msg}")
            print("[AUTO-SYNC] Please fix the repository issue and restart auto-sync")
            return
        
        if self.sync_thread and self.sync_thread.is_alive():
            print("[AUTO-SYNC] Auto-sync already running")
            return
        
        self.running = True
        self.sync_thread = threading.Thread(target=self.auto_sync_worker)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        
        print("[AUTO-SYNC] Started automatic synchronization")
        self.show_notification("[AUTO-SYNC] Auto-sync started")
    
    def stop_auto_sync(self):
        """Stop automatic synchronization"""
        self.running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5)
        
        print("[AUTO-SYNC] Stopped automatic synchronization")
        self.show_notification("[STOP] Auto-sync stopped")
    
    def run_interactive(self):
        """Run interactive mode"""
        print("\n" + "="*60)
        print("Cursor GitHub Integration")
        print("="*60)
        
        # Setup authentication if needed
        if not self.github_token:
            if not self.setup_github_auth():
                return
        
        # Start auto-sync
        if self.auto_sync_enabled:
            self.start_auto_sync()
        
        # Interactive menu
        while True:
            print("\nOptions:")
            print("1. [SEND] Send to GitHub (One-click)")
            print("2. [RECEIVE] Receive from GitHub (One-click)")
            print("3. [AUTO-SYNC] Start Auto-sync")
            print("4. [STOP] Stop Auto-sync")
            print("5. [SETTINGS] Settings")
            print("6. [EXIT] Exit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                self.one_click_send()
            elif choice == '2':
                self.one_click_receive()
            elif choice == '3':
                self.start_auto_sync()
            elif choice == '4':
                self.stop_auto_sync()
            elif choice == '5':
                self.show_settings()
            elif choice == '6':
                self.stop_auto_sync()
                break
            else:
                print("Invalid choice")
    
    def show_settings(self):
        """Show and edit settings"""
        print("\n" + "="*40)
        print("Settings")
        print("="*40)
        print(f"Repository: {self.github_owner}/{self.github_repo}")
        print(f"Auto-sync: {'Enabled' if self.auto_sync_enabled else 'Disabled'}")
        print(f"Sync interval: {self.sync_interval}s")
        print(f"Auto-commit threshold: {self.auto_commit_threshold} files")
        print(f"Notifications: {'Enabled' if self.config.get('notifications_enabled') else 'Disabled'}")
        
        print("\nEdit settings? (y/n): ", end="")
        if input().lower() == 'y':
            self.edit_settings()
    
    def edit_settings(self):
        """Edit settings interactively"""
        print("\nEnter new values (press Enter to keep current):")
        
        new_owner = input(f"GitHub owner [{self.github_owner}]: ").strip()
        if new_owner:
            self.github_owner = new_owner
            self.config['github_owner'] = new_owner
        
        new_repo = input(f"Repository name [{self.github_repo}]: ").strip()
        if new_repo:
            self.github_repo = new_repo
            self.config['github_repo'] = new_repo
        
        new_interval = input(f"Sync interval (seconds) [{self.sync_interval}]: ").strip()
        if new_interval.isdigit():
            self.sync_interval = int(new_interval)
            self.config['sync_interval'] = self.sync_interval
        
        new_threshold = input(f"Auto-commit threshold [{self.auto_commit_threshold}]: ").strip()
        if new_threshold.isdigit():
            self.auto_commit_threshold = int(new_threshold)
            self.config['auto_commit_threshold'] = self.auto_commit_threshold
        
        self.save_config()
        print("Settings saved!")

def main():
    """Main function"""
    import sys
    
    integration = CursorGitHubIntegration()
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--token' and len(sys.argv) > 2:
            integration.github_token = sys.argv[2]
            integration.config['github_token'] = sys.argv[2]
            integration.save_config()
            print("GitHub token saved!")
            return
        elif sys.argv[1] == '--send':
            integration.one_click_send()
            return
        elif sys.argv[1] == '--receive':
            integration.one_click_receive()
            return
        elif sys.argv[1] == '--auto-sync':
            integration.start_auto_sync()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                integration.stop_auto_sync()
            return
    
    # Run interactive mode
    integration.run_interactive()

if __name__ == "__main__":
    main()
