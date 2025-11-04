#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Auto Token Creator
- Tries multiple methods to create GitHub token automatically
- GitHub CLI method (preferred)
- API method (fallback)
- Manual method (last resort)
"""

import os
import json
import subprocess
import time
from auto_github_token_creator import AutoGitHubTokenCreator
from gh_cli_token_creator import GitHubCLITokenCreator

class UnifiedAutoTokenCreator:
    def __init__(self):
        """
        Initialize Unified Auto Token Creator
        """
        self.methods = [
            ("GitHub CLI", self.try_gh_cli_method),
            ("GitHub API", self.try_api_method),
            ("Manual Setup", self.try_manual_method)
        ]
        
        print("[UNIFIED] Unified Auto Token Creator initialized")
    
    def try_gh_cli_method(self):
        """Try GitHub CLI method"""
        print("\n[METHOD] Trying GitHub CLI method...")
        creator = GitHubCLITokenCreator()
        return creator.create_token_automatically()
    
    def try_api_method(self):
        """Try GitHub API method"""
        print("\n[METHOD] Trying GitHub API method...")
        creator = AutoGitHubTokenCreator()
        return creator.create_token_automatically()
    
    def try_manual_method(self):
        """Try manual setup method"""
        print("\n[METHOD] Manual setup required...")
        print("\n" + "="*60)
        print("MANUAL TOKEN CREATION")
        print("="*60)
        print("Please follow these steps:")
        print("1. Go to: https://github.com/settings/tokens/new")
        print("2. Note: Cursor GitHub Integration")
        print("3. Expiration: 30 days")
        print("4. Scopes: Check 'repo'")
        print("5. Click 'Generate token'")
        print("6. Copy the token")
        print("="*60)
        
        token = input("Paste your token here: ").strip()
        
        if token and token.startswith('ghp_'):
            # Save token
            config_file = "cursor_github_config.json"
            
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {}
            
            config['github_token'] = token
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print("[SUCCESS] Token saved successfully!")
            return True
        else:
            print("[ERROR] Invalid token format")
            return False
    
    def create_token_automatically(self):
        """Try all methods to create token automatically"""
        print("\n" + "="*60)
        print("UNIFIED AUTO TOKEN CREATOR")
        print("="*60)
        print("This will try multiple methods to create a GitHub token:")
        print("1. GitHub CLI (automatic)")
        print("2. GitHub API (semi-automatic)")
        print("3. Manual setup (guided)")
        print("="*60)
        
        for method_name, method_func in self.methods:
            print(f"\n[METHOD] Trying {method_name}...")
            try:
                if method_func():
                    print(f"[SUCCESS] Token created using {method_name} method!")
                    return True
                else:
                    print(f"[FAILED] {method_name} method failed")
            except Exception as e:
                print(f"[ERROR] {method_name} method error: {e}")
        
        print("\n[FAILED] All methods failed")
        return False
    
    def check_existing_token(self):
        """Check if token already exists"""
        config_file = "cursor_github_config.json"
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                if config.get('github_token'):
                    print("[EXISTING] Token already exists in config file")
                    return True
            except:
                pass
        
        return False
    
    def test_existing_token(self):
        """Test existing token"""
        try:
            import requests
            
            config_file = "cursor_github_config.json"
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            token = config.get('github_token')
            if not token:
                return False
            
            # Test token
            url = "https://api.github.com/user"
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                user_info = response.json()
                print(f"[TEST] Existing token works! Authenticated as: {user_info['login']}")
                return True
            else:
                print(f"[TEST] Existing token is invalid: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[TEST] Token test failed: {e}")
            return False

def main():
    """Main function"""
    creator = UnifiedAutoTokenCreator()
    
    print("Unified Auto Token Creator")
    print("This tool will automatically create a GitHub token using the best available method.")
    
    # Check if token already exists
    if creator.check_existing_token():
        print("\n[EXISTING] Token found in config file")
        
        if creator.test_existing_token():
            print("[SUCCESS] Existing token is valid!")
            print("You can now run: python cursor_github_integration.py")
            return
        else:
            print("[INVALID] Existing token is invalid, creating new one...")
    
    # Try to create token
    success = creator.create_token_automatically()
    
    if success:
        print("\n[SUCCESS] Token creation completed!")
        print("You can now run: python cursor_github_integration.py")
    else:
        print("\n[FAILED] Token creation failed.")
        print("Please try again or contact support.")

if __name__ == "__main__":
    main()
