#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GitHubトークンの確認スクリプト"""

import json
import requests
import sys
import codecs

# WindowsでのUTF-8出力設定
if sys.platform.startswith('win'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 設定ファイルを読み込む
with open('config/github_auto_commit_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

token = config.get('github_token', '')
owner = config.get('github_owner', '')
repo = config.get('github_repo', '')

print("=" * 60)
print("GitHub設定確認")
print("=" * 60)
print(f"オーナー: {owner}")
print(f"リポジトリ: {repo}")
print(f"トークン: {token[:10]}...{token[-4:] if len(token) > 14 else 'N/A'}")
print()

# トークンの有効性を確認
if not token:
    print("[ERROR] トークンが設定されていません")
else:
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    # ユーザー情報を取得
    print("1. ユーザー認証の確認...")
    try:
        r = requests.get('https://api.github.com/user', headers=headers, timeout=10)
        if r.status_code == 200:
            user_info = r.json()
            print(f"   [OK] 認証成功: {user_info.get('login', 'N/A')}")
            print(f"   [INFO] メール: {user_info.get('email', 'N/A')}")
        else:
            print(f"   [ERROR] 認証失敗: {r.status_code}")
            print(f"   エラー: {r.text[:200]}")
    except Exception as e:
        print(f"   [ERROR] エラー: {e}")
    
    print()
    
    # リポジトリへのアクセス権限を確認
    print(f"2. リポジトリ '{owner}/{repo}' へのアクセス確認...")
    try:
        r = requests.get(f'https://api.github.com/repos/{owner}/{repo}', headers=headers, timeout=10)
        if r.status_code == 200:
            repo_info = r.json()
            print(f"   [OK] アクセス可能")
            print(f"   [INFO] リポジトリ名: {repo_info.get('full_name', 'N/A')}")
            print(f"   [INFO] プライベート: {repo_info.get('private', False)}")
            print(f"   [INFO] デフォルトブランチ: {repo_info.get('default_branch', 'N/A')}")
        elif r.status_code == 404:
            print(f"   [ERROR] リポジトリが見つかりません")
            print(f"   リポジトリ名またはアクセス権限を確認してください")
        elif r.status_code == 401:
            print(f"   [ERROR] 認証エラー")
            print(f"   トークンが無効または権限が不足しています")
        else:
            print(f"   [ERROR] エラー: {r.status_code}")
            print(f"   エラー: {r.text[:200]}")
    except Exception as e:
        print(f"   [ERROR] エラー: {e}")
    
    print()
    
    # トークンの権限を確認
    print("3. トークンの権限確認...")
    try:
        r = requests.get('https://api.github.com/user', headers=headers, timeout=10)
        if r.status_code == 200:
            # スコープ情報はヘッダーから取得
            scopes = r.headers.get('X-OAuth-Scopes', '')
            if scopes:
                print(f"   [OK] スコープ: {scopes}")
                if 'repo' in scopes:
                    print(f"   [OK] 'repo'権限があります")
                else:
                    print(f"   [WARNING] 'repo'権限がありません（必要です）")
            else:
                print(f"   [WARNING] スコープ情報が取得できませんでした")
        else:
            print(f"   [ERROR] 認証に失敗しました")
    except Exception as e:
        print(f"   [ERROR] エラー: {e}")

print()
print("=" * 60)

