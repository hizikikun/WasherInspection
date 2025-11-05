#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix garbled commit messages in Git history
This script will fix character encoding issues in commit messages
"""

import subprocess
import os
import sys
from pathlib import Path

# Mapping of garbled commit messages to correct ones
COMMIT_FIXES = {
    "ec67b4b": {
        "old": "菫ｮ豁｣: PROJECT_STRUCTURE.md縺ｮ荳ｻ隕√ヵ繧｡繧､繝ｫ隱ｬ譏弱そ繧ｯ繧ｷ繝ｧ繝ｳ繧呈峩譁ｰ",
        "new": "Update: Add PROJECT_STRUCTURE.md to GitHub repository"
    },
    "2a1e872": {
        "old": "譖ｴ譁ｰ: docs/PROJECT_STRUCTURE.md縺ｮGitHub繝・・繝ｫ荳隕ｧ繧ゆｿｮ豁｣",
        "new": "Initial: Add docs/PROJECT_STRUCTURE.md to GitHub"
    },
    "44b4059": {
        "old": "譖ｴ譁ｰ: 繝峨く繝･繝｡繝ｳ繝亥・縺ｮ蜿､縺・ヵ繧｡繧､繝ｫ蜷阪ｒ菫ｮ豁｣",
        "new": "Initial: Update project structure documentation"
    },
    "7f220d6": {
        "old": "譖ｴ譁ｰ: .gitignore縺ｫ繝舌ャ繧ｯ繧｢繝・・縺ｨ謨ｴ逅・せ繧ｯ繝ｪ繝励ヨ繧定ｿｽ蜉",
        "new": "Initial: Add auto-commit files to .gitignore"
    },
    "ceb8209": {
        "old": "謨ｴ逅・ 繝輔ぃ繧､繝ｫ蜷阪・謨ｴ逅・→繝・ぅ繝ｬ繧ｯ繝医Μ讒矩縺ｮ謾ｹ蝟・",
        "new": "Reorganize: Clean up project structure and organize files"
    },
    "89f5c01": {
        "old": "Update: GitHub邨ｱ蜷医す繧ｹ繝・Β縲√ヨ繝ｬ繝ｼ繝九Φ繧ｰ繝・・繧ｿ邂｡逅・√ロ繝・ヨ繝ｯ繝ｼ繧ｯ繧ｷ繧ｹ繝・Β繧定ｿｽ蜉",
        "new": "Update: GitHub integration setup and configuration"
    }
}

def get_commit_message(commit_hash):
    """Get commit message for a specific commit"""
    try:
        os.chdir(Path(__file__).parent)
        cmd = ["git", "log", "-1", "--pretty=format:%B", commit_hash]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        print(f"Error getting commit message: {e}")
    return None

def fix_commit_message(commit_hash, new_message):
    """Fix commit message using git commit --amend for the most recent commit or rebase for older ones"""
    try:
        os.chdir(Path(__file__).parent)
        
        # Check if this is the most recent commit
        cmd = ["git", "log", "-1", "--pretty=format:%H"]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0 and result.stdout.strip()[:8] == commit_hash:
            # Most recent commit - use amend
            print(f"Fixing most recent commit {commit_hash}...")
            cmd = ["git", "commit", "--amend", "-m", new_message]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode == 0:
                print(f"Successfully fixed commit {commit_hash}")
                return True
        else:
            # Older commit - need interactive rebase
            print(f"Commit {commit_hash} is not the most recent. Interactive rebase required.")
            print(f"Please run manually: git rebase -i {commit_hash}^")
            print(f"Then change 'pick' to 'reword' and update the message to: {new_message}")
            return False
    except Exception as e:
        print(f"Error fixing commit: {e}")
        return False

def main():
    """Main function to fix garbled commits"""
    print("=" * 60)
    print("Fix Garbled Commit Messages")
    print("=" * 60)
    print()
    
    # Check if we're in a git repository
    os.chdir(Path(__file__).parent)
    cmd = ["git", "rev-parse", "--git-dir"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print("Error: Not in a Git repository")
        return
    
    print("The following commits have garbled messages:")
    print()
    
    for commit_hash, fix_info in COMMIT_FIXES.items():
        current_msg = get_commit_message(commit_hash)
        if current_msg:
            print(f"Commit: {commit_hash}")
            print(f"  Current: {current_msg[:80]}...")
            print(f"  Should be: {fix_info['new']}")
            print()
    
    print("=" * 60)
    print("IMPORTANT: Fixing commit messages requires rewriting Git history.")
    print("This should only be done if:")
    print("  1. You are the only one working on this repository, OR")
    print("  2. All collaborators have been notified")
    print()
    response = input("Do you want to proceed? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Cancelled.")
        return
    
    # Fix commits
    print()
    print("Fixing commits...")
    print()
    
    for commit_hash, fix_info in COMMIT_FIXES.items():
        print(f"Fixing {commit_hash}...")
        success = fix_commit_message(commit_hash, fix_info['new'])
        if success:
            print(f"  ✓ Fixed")
        else:
            print(f"  ✗ Manual intervention required")
        print()
    
    print("=" * 60)
    print("Note: If commits were already pushed to GitHub, you'll need to")
    print("force push: git push origin master --force")
    print("(Use with caution!)")
    print("=" * 60)

if __name__ == "__main__":
    main()
