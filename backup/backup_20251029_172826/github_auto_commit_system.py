#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Auto-Commit System
- Monitors code changes and automatically commits to GitHub
- Sends data and comments when changes exceed threshold
- Supports batch commits and intelligent change detection
"""

import os
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

class GitHubAutoCommitSystem:
    def __init__(self, config_file="github_auto_commit_config.json"):
        """
        Initialize the GitHub auto-commit system
        """
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
        """Generate commit message based on changes"""
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
        
        return message
    
    def create_github_commit(self, changes, commit_message):
        """Create commit on GitHub"""
        if not self.github_token:
            print("[GITHUB] No token configured")
            return False
        
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
                "Accept": "application/vnd.github.v3+json"
            }
            
            tree_data = {
                "base_tree": head_commit.hexsha if head_commit else None,
                "tree": tree_items
            }
            
            tree_response = requests.post(tree_url, headers=headers, json=tree_data)
            if tree_response.status_code != 201:
                print(f"[GITHUB] Failed to create tree: {tree_response.status_code}")
                return False
            
            tree_sha = tree_response.json()['sha']
            commit_data["tree"] = tree_sha
            
            # Create commit
            commit_url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/git/commits"
            commit_response = requests.post(commit_url, headers=headers, json=commit_data)
            
            if commit_response.status_code != 201:
                print(f"[GITHUB] Failed to create commit: {commit_response.status_code}")
                return False
            
            commit_sha = commit_response.json()['sha']
            
            # Update branch reference
            ref_url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/git/refs/heads/{self.config.get('branch', 'main')}"
            ref_data = {
                "sha": commit_sha
            }
            
            ref_response = requests.patch(ref_url, headers=headers, json=ref_data)
            if ref_response.status_code != 200:
                print(f"[GITHUB] Failed to update branch: {ref_response.status_code}")
                return False
            
            print(f"[GITHUB] Commit created successfully: {commit_sha[:8]}")
            return True
            
        except Exception as e:
            print(f"[GITHUB] Error creating commit: {e}")
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
        """Backup changed files"""
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
                print(f"[BACKUP] Backed up {change['path']}")
            except Exception as e:
                print(f"[BACKUP] Error backing up {change['path']}: {e}")
    
    def process_changes(self, changes, total_size):
        """Process and commit changes"""
        if not changes:
            return
        
        print(f"[PROCESS] Processing {len(changes)} changes ({total_size / 1024:.1f} KB)")
        
        # Backup changes
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
