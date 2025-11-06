#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHubリポジトリの情報を更新するスクリプト
- リポジトリの説明を設定
- トピックを設定
- ウェブサイトURLを設定（オプション）
"""

import json
import requests
import sys
import os

# 設定ファイルを読み込む
try:
    with open('config/github_auto_commit_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
except FileNotFoundError:
    print("[ERROR] config/github_auto_commit_config.json が見つかりません。")
    sys.exit(1)

token = config.get('github_token', '')
owner = config.get('github_owner', 'hizikikun')
repo = config.get('github_repo', 'WasherInspection')

if not token:
    print("[ERROR] GitHubトークンが設定されていません。")
    sys.exit(1)

# リポジトリ情報
repo_info = {
    "name": repo,
    "description": "AI搭載の高精度樹脂ワッシャー不良品検出システム - EfficientNetベースのリアルタイム検査システム",
    "homepage": "",
    "private": False,
    "has_issues": True,
    "has_projects": True,
    "has_wiki": True,
    "has_downloads": True,
    "default_branch": "main"
}

# トピック
topics = [
    "ai",
    "machine-learning",
    "computer-vision",
    "defect-detection",
    "quality-inspection",
    "tensorflow",
    "efficientnet",
    "python",
    "opencv",
    "industrial-automation"
]

headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}

# リポジトリ情報を更新
print("=" * 60)
print("GitHubリポジトリ情報を更新中...")
print("=" * 60)
print(f"リポジトリ: {owner}/{repo}")
print()

# 1. リポジトリの基本情報を更新
print("[1/2] リポジトリの基本情報を更新...")
try:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.patch(url, headers=headers, json=repo_info, timeout=10)
    
    if response.status_code == 200:
        print("  [OK] リポジトリ情報を更新しました")
        print(f"  説明: {repo_info['description']}")
    else:
        print(f"  [WARNING] リポジトリ情報の更新に失敗: {response.status_code}")
        if response.content:
            print(f"  エラー: {response.json().get('message', 'Unknown error')}")
except Exception as e:
    print(f"  [ERROR] エラー: {e}")

print()

# 2. トピックを設定
print("[2/2] トピックを設定...")
try:
    url = f"https://api.github.com/repos/{owner}/{repo}/topics"
    headers_topics = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.mercy-preview+json"
    }
    response = requests.put(url, headers=headers_topics, json={"names": topics}, timeout=10)
    
    if response.status_code == 200:
        print("  [OK] トピックを設定しました")
        print(f"  トピック: {', '.join(topics)}")
    else:
        print(f"  [WARNING] トピックの設定に失敗: {response.status_code}")
        if response.content:
            print(f"  エラー: {response.json().get('message', 'Unknown error')}")
except Exception as e:
    print(f"  [ERROR] エラー: {e}")

print()
print("=" * 60)
print("更新完了！")
print("=" * 60)
print(f"\nリポジトリURL: https://github.com/{owner}/{repo}")

