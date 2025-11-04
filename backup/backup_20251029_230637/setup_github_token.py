#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct Token Input for Cursor GitHub Integration
"""

import os
import json
import requests

def test_token(token):
    """Test GitHub token"""
    try:
        url = "https://api.github.com/user"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            user_info = response.json()
            print(f"[SUCCESS] Token works! Authenticated as: {user_info['login']}")
            return True
        else:
            print(f"[ERROR] Token test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Token test failed: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("CURSOR GITHUB TOKEN SETUP")
    print("="*60)
    print("To get your GitHub token:")
    print("1. Go to: https://github.com/settings/tokens/new")
    print("2. Note: Cursor GitHub Integration")
    print("3. Expiration: 30 days")
    print("4. Scopes: Check 'repo'")
    print("5. Click 'Generate token'")
    print("6. Copy the token (starts with 'ghp_')")
    print("="*60)
    print("\nOpen this file to edit your token:")
    print("cursor_github_config.json")
    print("\nReplace 'YOUR_TOKEN_HERE' with your actual token")
    print("="*60)
    
    # Check if token is already set
    config_file = "cursor_github_config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        token = config.get('github_token', '')
        if token and token != 'YOUR_TOKEN_HERE':
            print("\n[TOKEN] Found token in config file")
            print(f"[TOKEN] Token: {token[:10]}...{token[-4:]}")
            
            if test_token(token):
                print("\n[READY] Your token is configured and working!")
                print("You can now use Cursor GitHub Integration:")
                print("  python cursor_github_integration.py")
            else:
                print("\n[ERROR] Your token is invalid. Please update it in:")
                print("  cursor_github_config.json")
        else:
            print("\n[TOKEN] No token configured yet")
            print("Please edit cursor_github_config.json and add your token")
    else:
        print("\n[ERROR] Config file not found")

if __name__ == "__main__":
    main()
