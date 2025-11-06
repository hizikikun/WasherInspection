#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Auto-Commit System
- Monitors code changes and automatically commits to GitHub
- Sends data and comments when changes exceed threshold
- Supports batch commits and intelligent change detection
"""

import os
import sys
import json
import time
import hashlib
import requests
import base64
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import git
import shutil

# Set UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    # Set environment variables for Git
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = 'git'

class GitHubAutoCommitSystem:
    def __init__(self, config_file=None):
        """
        Initialize the GitHub auto-commit system
        """
        if config_file is None:
            # Try config directory first, then current directory
            if os.path.exists("config/github_auto_commit_config.json"):
                config_file = "config/github_auto_commit_config.json"
            elif os.path.exists("github_auto_commit_config.json"):
                config_file = "github_auto_commit_config.json"
            else:
                config_file = "config/github_auto_commit_config.json"
        self.config = self.load_config(config_file)
        self.github_token = self.config.get('github_token')
        self.github_owner = self.config.get('github_owner')
        self.github_repo = self.config.get('github_repo')
        
        # Paths
        self.code_path = self.config.get('code_path', '.')
        self.backup_path = self.config.get('backup_path', 'backup')
        
        # Change tracking
        self.file_hashes = {}
        self.change_history = []
        self.pending_changes = []
        
        # Thresholds
        self.commit_threshold = self.config.get('commit_threshold', 5)  # Number of files changed
        self.size_threshold = self.config.get('size_threshold', 1024 * 1024)  # 1MB
        self.time_threshold = self.config.get('time_threshold', 300)  # 5 minutes
        
        # Git repository
        self.git_repo = None
        self.init_git_repo()
        
        # Create backup directory
        os.makedirs(self.backup_path, exist_ok=True)
        
        print(f"[AUTO-COMMIT] Repository: {self.github_owner}/{self.github_repo}")
        print(f"[AUTO-COMMIT] Commit threshold: {self.commit_threshold} files")
        print(f"[AUTO-COMMIT] Size threshold: {self.size_threshold / 1024:.1f} KB")
        print(f"[AUTO-COMMIT] Time threshold: {self.time_threshold} seconds")
    
    def load_config(self, config_file):
        """Load configuration"""
        default_config = {
            "github_token": "",
            "github_owner": "your-username",
            "github_repo": "washer-inspection-code",
            "code_path": ".",
            "backup_path": "backup",
            "commit_threshold": 5,
            "size_threshold": 1048576,  # 1MB
            "time_threshold": 300,  # 5 minutes
            "auto_commit": True,
            "include_code_files": [".py", ".json", ".md", ".yml", ".yaml", ".txt"],
            "exclude_dirs": ["__pycache__", ".git", "node_modules", ".vscode", "backup"],
            "commit_message_template": "Auto-commit: {change_count} files changed",
            "branch": "main"
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
    
    def init_git_repo(self):
        """Initialize Git repository"""
        try:
            self.git_repo = git.Repo(self.code_path)
            print(f"[GIT] Repository initialized: {self.git_repo.working_dir}")
        except git.InvalidGitRepositoryError:
            print("[GIT] Not a Git repository, initializing...")
            try:
                self.git_repo = git.Repo.init(self.code_path)
                print(f"[GIT] New repository created: {self.code_path}")
            except Exception as e:
                print(f"[GIT] Error initializing repository: {e}")
                self.git_repo = None
        except Exception as e:
            print(f"[GIT] Error with Git repository: {e}")
            self.git_repo = None
    
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
        total_size = 0
        
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
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        
                        if relative_path not in self.file_hashes:
                            # New file
                            changes.append({
                                'type': 'new',
                                'path': relative_path,
                                'hash': current_hash,
                                'size': file_size,
                                'timestamp': time.time()
                            })
                            self.file_hashes[relative_path] = current_hash
                        elif self.file_hashes[relative_path] != current_hash:
                            # Modified file
                            changes.append({
                                'type': 'modified',
                                'path': relative_path,
                                'hash': current_hash,
                                'size': file_size,
                                'timestamp': time.time()
                            })
                            self.file_hashes[relative_path] = current_hash
        
        return changes, total_size
    
    def should_commit(self, changes, total_size):
        """Determine if changes should be committed"""
        # Check file count threshold
        if len(changes) >= self.commit_threshold:
            return True, f"File count threshold reached: {len(changes)} files"
        
        # Check size threshold
        if total_size >= self.size_threshold:
            return True, f"Size threshold reached: {total_size / 1024:.1f} KB"
        
        # Check time threshold (if there are any changes)
        if changes:
            oldest_change = min(changes, key=lambda x: x['timestamp'])
            time_since_first_change = time.time() - oldest_change['timestamp']
            if time_since_first_change >= self.time_threshold:
                return True, f"Time threshold reached: {time_since_first_change:.1f} seconds"
        
        return False, "No threshold reached"
    
    def generate_commit_message(self, changes):
        """Generate commit message based on changes (UTF-8 encoded)"""
        template = self.config.get('commit_message_template', "Auto-commit: {change_count} files changed")
        
        # Count changes by type
        change_counts = defaultdict(int)
        for change in changes:
            change_counts[change['type']] += 1
        
        # Generate message
        message = template.format(
            change_count=len(changes),
            new_count=change_counts['new'],
            modified_count=change_counts['modified']
        )
        
        # Add details
        if change_counts['new'] > 0:
            message += f" ({change_counts['new']} new)"
        if change_counts['modified'] > 0:
            message += f" ({change_counts['modified']} modified)"
        
        # Message is already a UTF-8 string, no need to encode/decode
        # Just ensure it's valid Unicode
        if not isinstance(message, str):
            message = str(message)
        
        # Validate UTF-8 encoding (Python strings are Unicode by default)
        try:
            message.encode('utf-8')
        except UnicodeEncodeError:
            # Fallback to ASCII-safe message if UTF-8 encoding fails
            message = f"Auto-commit: {len(changes)} files changed ({change_counts['new']} new, {change_counts['modified']} modified)"
        
        return message
    
    def create_github_commit_contents_api(self, changes, commit_message):
        """Create commit using Contents API (works better with fine-grained tokens)"""
        try:
            branch = self.config.get('branch', 'master')
            success_count = 0
            
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            for change in changes:
                file_path = change['path']
                full_file_path = os.path.join(self.code_path, file_path)
                
                try:
                    with open(full_file_path, 'rb') as f:
                        content = f.read()
                    content_base64 = base64.b64encode(content).decode('utf-8')
                except Exception as e:
                    print(f"[ERROR] Error reading file {full_file_path}: {e}")
                    continue
                
                # Get current file SHA if exists
                file_url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/contents/{file_path}"
                file_response = requests.get(f"{file_url}?ref={branch}", headers=headers, timeout=10)
                
                file_data = {
                    "message": commit_message if success_count == 0 else f"{commit_message} - {file_path}",
                    "content": content_base64,
                    "branch": branch
                }
                
                # If file exists, include SHA for update
                if file_response.status_code == 200:
                    file_data["sha"] = file_response.json()['sha']
                
                # Create or update file
                import json as json_module
                put_response = requests.put(file_url, headers=headers, json=file_data, timeout=30)
                
                if put_response.status_code in [200, 201]:
                    success_count += 1
                    if success_count == 1:
                        print(f"[GITHUB] First file committed: {file_path}")
                else:
                    error_msg = put_response.json().get('message', 'Unknown error') if put_response.content else 'No error message'
                    print(f"[ERROR] Failed to upload {file_path}: {put_response.status_code} - {error_msg}")
            
            if success_count == len(changes):
                print(f"[GITHUB] Successfully committed {success_count} files using Contents API")
                return True
            elif success_count > 0:
                print(f"[GITHUB] Partially committed: {success_count}/{len(changes)} files")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"[ERROR] Contents API commit failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_github_commit(self, changes, commit_message):
        """Create commit on GitHub - tries Contents API first, then Git Trees API"""
        if not self.github_token:
            print("[GITHUB] No token configured")
            return False
        
        # Try Contents API first (better for fine-grained tokens)
        print("[GITHUB] Trying Contents API (recommended for fine-grained tokens)...")
        if self.create_github_commit_contents_api(changes, commit_message):
            return True
        
        # Fallback to Git Trees API
        print("[GITHUB] Contents API failed, trying Git Trees API...")
        try:
            # Prepare commit data
            commit_data = {
                "message": commit_message,
                "tree": None,  # Will be set after creating tree
                "parents": []  # Will be set to current HEAD
            }
            
            # Get current HEAD commit
            try:
                head_commit = self.git_repo.head.commit
                commit_data["parents"] = [head_commit.hexsha]
            except:
                # First commit
                commit_data["parents"] = []
            
            # Create tree with all changed files
            tree_items = []
            for change in changes:
                file_path = os.path.join(self.code_path, change['path'])
                
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    
                    # Encode to base64
                    content_base64 = base64.b64encode(content).decode('utf-8')
                    
                    tree_items.append({
                        "path": change['path'],
                        "mode": "100644",
                        "type": "blob",
                        "content": content_base64
                    })
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue
            
            if not tree_items:
                print("[GITHUB] No files to commit")
                return False
            
            # Create tree
            tree_url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/git/trees"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json; charset=utf-8"
            }
            
            tree_data = {
                "base_tree": head_commit.hexsha if head_commit else None,
                "tree": tree_items
            }
            
            # Ensure JSON encoding is UTF-8
            import json as json_module
            tree_json = json_module.dumps(tree_data, ensure_ascii=False).encode('utf-8')
            tree_response = requests.post(tree_url, headers=headers, data=tree_json)
            if tree_response.status_code != 201:
                error_msg = tree_response.json().get('message', 'Unknown error') if tree_response.content else 'No error message'
                print(f"[GITHUB] Failed to create tree: {tree_response.status_code} - {error_msg}")
                return False
            
            tree_sha = tree_response.json()['sha']
            commit_data["tree"] = tree_sha
            
            # Create commit
            commit_url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/git/commits"
            # Commit message should be a UTF-8 string (not encoded/decoded)
            # GitHub API expects UTF-8 encoded JSON, which json.dumps handles correctly
            commit_data["message"] = commit_message  # Keep as string, UTF-8 will be preserved in JSON
            # Use JSON dumps with ensure_ascii=False to preserve UTF-8, encode to bytes for requests
            commit_json = json_module.dumps(commit_data, ensure_ascii=False).encode('utf-8')
            commit_response = requests.post(commit_url, headers=headers, data=commit_json)
            
            if commit_response.status_code != 201:
                error_msg = commit_response.json().get('message', 'Unknown error') if commit_response.content else 'No error message'
                print(f"[GITHUB] Failed to create commit: {commit_response.status_code} - {error_msg}")
                return False
            
            commit_sha = commit_response.json()['sha']
            
            # Update branch reference
            ref_url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/git/refs/heads/{self.config.get('branch', 'master')}"
            ref_data = {
                "sha": commit_sha
            }
            
            ref_json = json_module.dumps(ref_data).encode('utf-8')
            ref_response = requests.patch(ref_url, headers=headers, data=ref_json)
            
            if ref_response.status_code != 200:
                error_msg = ref_response.json().get('message', 'Unknown error') if ref_response.content else 'No error message'
                print(f"[GITHUB] Failed to update branch: {ref_response.status_code} - {error_msg}")
                return False
            
            # After successful GitHub commit, fetch and merge to local repo
            if self.git_repo:
                try:
                    # Fetch the remote changes
                    origin = self.git_repo.remote('origin')
                    if origin:
                        origin.fetch()
                        # Reset local branch to match remote
                        branch_name = self.config.get('branch', 'master')
                        try:
                            self.git_repo.heads[branch_name].set_commit(commit_sha)
                            print(f"[GIT] Local repository updated to commit {commit_sha[:8]}")
                        except Exception as e:
                            print(f"[GIT] Warning: Could not update local branch: {e}")
                except Exception as e:
                    print(f"[GIT] Warning: Could not sync local repository: {e}")
            
            print(f"[GITHUB] Commit created successfully: {commit_sha[:8]}")
            return True
            
        except Exception as e:
            print(f"[GITHUB] Error creating commit: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_github_issue(self, title, body, labels=None):
        """Create GitHub issue for significant changes"""
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
            "labels": labels or ["auto-commit", "code-changes"]
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
    
    def backup_changes(self, changes):
        """Backup changed files (skip for large batches)"""
        # Skip backup if too many files (performance optimization)
        if len(changes) > 100:
            print(f"[BACKUP] Skipping backup for {len(changes)} files (too many)")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self.backup_path, f"backup_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)
        
        for change in changes:
            source_path = os.path.join(self.code_path, change['path'])
            backup_path = os.path.join(backup_dir, change['path'])
            
            # Create directory structure
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            try:
                shutil.copy2(source_path, backup_path)
                # Only print for first few files to avoid spam
                if len([c for c in changes if changes.index(c) < 5]) > 0 and changes.index(change) < 5:
                    print(f"[BACKUP] Backed up {change['path']}")
            except Exception as e:
                print(f"[BACKUP] Error backing up {change['path']}: {e}")
    
    def process_changes(self, changes, total_size):
        """Process and commit changes"""
        if not changes:
            return
        
        print(f"[PROCESS] Processing {len(changes)} changes ({total_size / 1024:.1f} KB)")
        
        # Filter out excluded directories from changes
        exclude_dirs = self.config.get('exclude_dirs', [])
        filtered_changes = []
        for change in changes:
            path_parts = change['path'].split(os.sep)
            if not any(part in exclude_dirs for part in path_parts):
                filtered_changes.append(change)
        
        if len(filtered_changes) != len(changes):
            print(f"[PROCESS] Filtered {len(changes) - len(filtered_changes)} excluded files")
            changes = filtered_changes
        
        if not changes:
            print("[PROCESS] No changes to commit after filtering")
            return
        
        # Backup changes (only for small batches)
        self.backup_changes(changes)
        
        # Generate commit message
        commit_message = self.generate_commit_message(changes)
        
        # Create GitHub commit
        success = self.create_github_commit(changes, commit_message)
        
        if success:
            # Create issue for significant changes
            if len(changes) >= self.commit_threshold * 2:  # Double threshold
                title = f"📝 Major Code Update - {len(changes)} files changed"
                body = f"""## Major Code Update Detected

**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Files Changed**: {len(changes)}
**Total Size**: {total_size / 1024:.1f} KB

## Change Summary
"""
                
                # Group changes by type
                change_groups = defaultdict(list)
                for change in changes:
                    change_groups[change['type']].append(change['path'])
                
                for change_type, files in change_groups.items():
                    body += f"\n### {change_type.title()} Files ({len(files)})\n"
                    for file_path in files[:10]:  # Show first 10 files
                        body += f"- `{file_path}`\n"
                    if len(files) > 10:
                        body += f"- ... and {len(files) - 10} more files\n"
                
                body += f"""
## Commit Details
- **Commit Message**: {commit_message}
- **Branch**: {self.config.get('branch', 'main')}
- **Auto-generated**: Yes

This issue was automatically created due to significant code changes."""
                
                self.create_github_issue(title, body, ["major-update", "auto-commit", "code-changes"])
            
            # Clear pending changes
            self.pending_changes = []
            print("[PROCESS] Changes committed successfully")
        else:
            print("[PROCESS] Failed to commit changes")
    
    def run_monitoring(self):
        """Run continuous monitoring"""
        print("[MONITOR] Starting continuous monitoring...")
        print("Press Ctrl+C to stop")
        
        last_check_time = time.time()
        
        while True:
            try:
                current_time = time.time()
                
                # Scan for changes
                changes, total_size = self.scan_code_changes()
                
                if changes:
                    print(f"[MONITOR] Found {len(changes)} changes ({total_size / 1024:.1f} KB)")
                    
                    # Add to pending changes
                    self.pending_changes.extend(changes)
                    
                    # Check if should commit
                    should_commit, reason = self.should_commit(self.pending_changes, total_size)
                    
                    if should_commit:
                        print(f"[MONITOR] {reason}")
                        self.process_changes(self.pending_changes, total_size)
                    else:
                        print(f"[MONITOR] Waiting for threshold: {reason}")
                
                # Wait before next check
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                print("\n[MONITOR] Stopping monitoring...")
                break
            except Exception as e:
                print(f"[MONITOR] Error: {e}")
                time.sleep(30)
    
    def run_once(self):
        """Run monitoring once"""
        print("[MONITOR] Running one-time check...")
        
        changes, total_size = self.scan_code_changes()
        
        if changes:
            print(f"Found {len(changes)} changes ({total_size / 1024:.1f} KB)")
            
            should_commit, reason = self.should_commit(changes, total_size)
            
            if should_commit:
                print(f"{reason}")
                self.process_changes(changes, total_size)
            else:
                print(f"Threshold not reached: {reason}")
        else:
            print("No changes detected")

def main():
    """Main function"""
    import sys
    
    system = GitHubAutoCommitSystem()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'once':
        system.run_once()
    else:
        system.run_monitoring()

if __name__ == "__main__":
    main()
