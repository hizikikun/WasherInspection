#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Token Setup
- Simple token input and configuration
- No complex automation
- Just input the token from the browser
"""

import os
import json
import requests

def setup_token():
    """Simple token setup"""
    print("\n" + "="*60)
    print("SIMPLE GITHUB TOKEN SETUP")
    print("="*60)
    print("The GitHub token creation page should be open in your browser.")
    print("Please follow these steps:")
    print("1. Note: Cursor GitHub Integration")
    print("2. Expiration: 30 days (default)")
    print("3. Scopes: Check 'repo'")
    print("4. Click 'Generate token'")
    print("5. Copy the token (starts with 'ghp_')")
    print("="*60)
    
    token = input("Paste your token here: ").strip()
    
    if not token:
        print("[ERROR] No token provided")
        return False
    
    if not token.startswith('ghp_'):
        print("[ERROR] Invalid token format. Token should start with 'ghp_'")
        return False
    
    # Test token
    print("[TEST] Testing token...")
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
        else:
            print(f"[ERROR] Token test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Token test failed: {e}")
        return False
    
    # Save token
    print("[SAVE] Saving token to config...")
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
        
        print(f"[SUCCESS] Token saved to {config_file}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save token: {e}")
        return False

def main():
    """Main function"""
    print("Simple GitHub Token Setup")
    print("This tool will help you set up your GitHub token.")
    
    success = setup_token()
    
    if success:
        print("\n" + "="*60)
        print("SETUP COMPLETED!")
        print("="*60)
        print("Your GitHub token has been configured.")
        print("You can now use Cursor GitHub Integration!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run: python cursor_github_integration.py")
        print("2. Or run: python cursor_github_extension.py")
    else:
        print("\n[FAILED] Token setup failed.")
        print("Please try again.")

if __name__ == "__main__":
    main()
