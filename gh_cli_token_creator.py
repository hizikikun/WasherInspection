#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub CLI Auto Token Creator
- Uses GitHub CLI to automatically create tokens
- No manual browser interaction required
- Fully automated token creation
"""

import os
import json
import subprocess
import time
from datetime import datetime

class GitHubCLITokenCreator:
    def __init__(self):
        """
        Initialize GitHub CLI Token Creator
        """
        self.token_name = "Cursor GitHub Integration"
        self.scopes = ["repo"]
        self.expiration_days = 30
        
        print("[GH-CLI] GitHub CLI Token Creator initialized")
    
    def check_gh_cli_installed(self):
        """Check if GitHub CLI is installed"""
        try:
            result = subprocess.run(['gh', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("[GH-CLI] GitHub CLI is installed")
                return True
            else:
                print("[GH-CLI] GitHub CLI is not installed")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("[GH-CLI] GitHub CLI is not installed")
            return False
    
    def install_gh_cli(self):
        """Install GitHub CLI automatically"""
        print("[GH-CLI] Installing GitHub CLI...")
        
        try:
            # Try to install via winget (Windows)
            result = subprocess.run(['winget', 'install', 'GitHub.cli'], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("[GH-CLI] GitHub CLI installed successfully via winget")
                return True
        except:
            pass
        
        try:
            # Try to install via chocolatey (Windows)
            result = subprocess.run(['choco', 'install', 'gh'], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("[GH-CLI] GitHub CLI installed successfully via chocolatey")
                return True
        except:
            pass
        
        print("[GH-CLI] Automatic installation failed. Please install GitHub CLI manually:")
        print("  Windows: winget install GitHub.cli")
        print("  macOS: brew install gh")
        print("  Linux: apt install gh")
        
        return False
    
    def authenticate_gh_cli(self):
        """Authenticate GitHub CLI"""
        try:
            print("[GH-CLI] Authenticating with GitHub...")
            
            # Check if already authenticated
            result = subprocess.run(['gh', 'auth', 'status'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and "Logged in" in result.stdout:
                print("[GH-CLI] Already authenticated")
                return True
            
            # Authenticate interactively
            print("[GH-CLI] Please authenticate with GitHub in the browser that will open...")
            result = subprocess.run(['gh', 'auth', 'login'], 
                                  timeout=300)
            
            if result.returncode == 0:
                print("[GH-CLI] Authentication successful")
                return True
            else:
                print("[GH-CLI] Authentication failed")
                return False
                
        except subprocess.TimeoutExpired:
            print("[GH-CLI] Authentication timeout")
            return False
        except Exception as e:
            print(f"[GH-CLI] Authentication error: {e}")
            return False
    
    def create_token_via_cli(self):
        """Create token using GitHub CLI"""
        try:
            print("[GH-CLI] Creating Personal Access Token...")
            
            # Create token with specific scopes
            cmd = [
                'gh', 'auth', 'token', '--scopes', ','.join(self.scopes),
                '--note', self.token_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                token = result.stdout.strip()
                print(f"[GH-CLI] Token created successfully!")
                print(f"[TOKEN] {token}")
                return token
            else:
                print(f"[GH-CLI] Token creation failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("[GH-CLI] Token creation timeout")
            return None
        except Exception as e:
            print(f"[GH-CLI] Token creation error: {e}")
            return None
    
    def save_token_to_config(self, token):
        """Save token to configuration file"""
        try:
            config_file = "cursor_github_config.json"
            
            # Load existing config or create new one
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Update token
            config['github_token'] = token
            
            # Save config
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"[SAVE] Token saved to {config_file}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save token: {e}")
            return False
    
    def create_token_automatically(self):
        """Create token automatically using GitHub CLI"""
        print("\n" + "="*60)
        print("GITHUB CLI AUTO TOKEN CREATOR")
        print("="*60)
        print("This will automatically create a GitHub Personal Access Token")
        print("using GitHub CLI (no manual browser interaction required).")
        print("="*60)
        
        # Check if GitHub CLI is installed
        if not self.check_gh_cli_installed():
            if not self.install_gh_cli():
                return False
        
        # Authenticate with GitHub
        if not self.authenticate_gh_cli():
            return False
        
        # Create token
        token = self.create_token_via_cli()
        if not token:
            return False
        
        # Save token
        if not self.save_token_to_config(token):
            return False
        
        print("\n" + "="*60)
        print("TOKEN CREATION COMPLETED!")
        print("="*60)
        print("Your GitHub Personal Access Token has been created and saved.")
        print("You can now use Cursor GitHub Integration without manual setup.")
        print("="*60)
        
        return True

def main():
    """Main function"""
    creator = GitHubCLITokenCreator()
    
    print("GitHub CLI Auto Token Creator")
    print("This tool will automatically create a GitHub token using GitHub CLI.")
    
    # Try to create token
    success = creator.create_token_automatically()
    
    if success:
        print("\n[SUCCESS] Token creation completed!")
        print("You can now run: python cursor_github_integration.py")
    else:
        print("\n[FAILED] Token creation failed.")
        print("Please try the manual method or install GitHub CLI.")

if __name__ == "__main__":
    main()
