#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto GitHub Token Creator
- Automatically creates GitHub Personal Access Token
- No manual browser interaction required
- Uses GitHub API to create tokens programmatically
"""

import os
import json
import requests
import base64
import time
from datetime import datetime

class AutoGitHubTokenCreator:
    def __init__(self):
        """
        Initialize Auto GitHub Token Creator
        """
        self.github_username = ""
        self.github_password = ""
        self.token_name = "Cursor GitHub Integration"
        self.scopes = ["repo"]
        self.expiration_days = 30
        
        print("[AUTO-TOKEN] Auto GitHub Token Creator initialized")
    
    def get_github_credentials(self):
        """Get GitHub credentials from user"""
        print("\n" + "="*60)
        print("GitHub認証情報の入力")
        print("="*60)
        print("GitHubのユーザー名とパスワードを入力してください")
        print("（2要素認証が有効な場合は、アプリパスワードを使用してください）")
        print("="*60)
        
        self.github_username = input("GitHub Username: ").strip()
        self.github_password = input("GitHub Password/App Password: ").strip()
        
        if not self.github_username or not self.github_password:
            print("[ERROR] Username and password are required")
            return False
        
        return True
    
    def create_personal_access_token(self):
        """Create Personal Access Token using GitHub API"""
        try:
            # GitHub API endpoint for creating personal access tokens
            url = "https://api.github.com/authorizations"
            
            # Prepare authentication
            auth_string = f"{self.github_username}:{self.github_password}"
            auth_bytes = auth_string.encode('ascii')
            auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
            
            headers = {
                "Authorization": f"Basic {auth_b64}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Cursor-GitHub-Integration"
            }
            
            # Prepare token data
            token_data = {
                "scopes": self.scopes,
                "note": self.token_name,
                "note_url": "https://github.com/settings/tokens",
                "fingerprint": f"cursor-integration-{int(time.time())}"
            }
            
            print("[AUTO-TOKEN] Creating Personal Access Token...")
            
            # Make API request
            response = requests.post(url, headers=headers, json=token_data)
            
            if response.status_code == 201:
                token_info = response.json()
                token = token_info['token']
                
                print(f"[SUCCESS] Token created successfully!")
                print(f"[TOKEN] {token}")
                print(f"[SCOPES] {', '.join(self.scopes)}")
                print(f"[EXPIRES] {token_info.get('expires_at', 'Never')}")
                
                return token
                
            elif response.status_code == 401:
                print("[ERROR] Authentication failed. Please check your credentials.")
                print("[ERROR] If you have 2FA enabled, use an App Password instead of your regular password.")
                return None
                
            elif response.status_code == 422:
                print("[ERROR] Token creation failed. Possible reasons:")
                print("  - Token with same name already exists")
                print("  - Invalid scopes requested")
                print("  - Account limitations")
                return None
                
            else:
                print(f"[ERROR] Token creation failed: {response.status_code}")
                print(f"[ERROR] Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Exception occurred: {e}")
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
    
    def test_token(self, token):
        """Test the created token"""
        try:
            url = "https://api.github.com/user"
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                user_info = response.json()
                print(f"[TEST] Token works! Authenticated as: {user_info['login']}")
                return True
            else:
                print(f"[ERROR] Token test failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Token test failed: {e}")
            return False
    
    def create_token_automatically(self):
        """Create token automatically with user interaction"""
        print("\n" + "="*60)
        print("AUTO GITHUB TOKEN CREATOR")
        print("="*60)
        print("This will automatically create a GitHub Personal Access Token")
        print("for Cursor GitHub Integration.")
        print("="*60)
        
        # Get credentials
        if not self.get_github_credentials():
            return False
        
        # Create token
        token = self.create_personal_access_token()
        if not token:
            return False
        
        # Test token
        if not self.test_token(token):
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
    
    def create_token_with_2fa(self):
        """Create token with 2FA support"""
        print("\n" + "="*60)
        print("2FA ENABLED ACCOUNT DETECTED")
        print("="*60)
        print("Your account has 2FA enabled.")
        print("Please use an App Password instead of your regular password.")
        print("="*60)
        print("To create an App Password:")
        print("1. Go to GitHub Settings > Developer settings > Personal access tokens")
        print("2. Click 'Generate new token (classic)'")
        print("3. Select 'repo' scope")
        print("4. Copy the generated token")
        print("5. Use that token as your password in this tool")
        print("="*60)
        
        return self.create_token_automatically()

def main():
    """Main function"""
    creator = AutoGitHubTokenCreator()
    
    print("GitHub Personal Access Token Auto Creator")
    print("This tool will automatically create a GitHub token for Cursor integration.")
    
    # Try to create token
    success = creator.create_token_automatically()
    
    if success:
        print("\n[SUCCESS] Token creation completed!")
        print("You can now run: python cursor_github_integration.py")
    else:
        print("\n[FAILED] Token creation failed.")
        print("Please try again or create the token manually.")

if __name__ == "__main__":
    main()
