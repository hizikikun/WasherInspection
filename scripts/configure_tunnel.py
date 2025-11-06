#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
トンネル設定ツール
インターネット経由でのアクセス設定を行います
"""

import json
import sys
from pathlib import Path

def main():
    """設定を対話的に入力"""
    print("=" * 60)
    print("インターネット経由アクセス設定")
    print("=" * 60)
    print()
    
    # 設定ファイルのパス
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / 'config' / 'remote_tunnel_config.json'
    config_dir = config_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # 既存の設定を読み込み
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {
            "tunnel_method": "ngrok",
            "local_port": 5000,
            "ngrok": {
                "enabled": False,
                "auth_token": "",
                "region": "us",
                "subdomain": ""
            },
            "cloudflare": {
                "enabled": False,
                "tunnel_token": ""
            },
            "custom_tunnel": {
                "enabled": False,
                "command": ""
            }
        }
    
    print("トンネル方法を選択してください:")
    print("1. ngrok (無料・簡単)")
    print("2. Cloudflare Tunnel (無料・高速)")
    print("3. カスタム (手動設定)")
    print("選択 (1-3): ", end='')
    method_choice = input().strip()
    
    method_map = {'1': 'ngrok', '2': 'cloudflare', '3': 'custom'}
    config['tunnel_method'] = method_map.get(method_choice, 'ngrok')
    
    print()
    print("ローカルサーバーのポート番号 (デフォルト: 5000): ", end='')
    port_input = input().strip()
    if port_input:
        try:
            config['local_port'] = int(port_input)
        except ValueError:
            print("無効なポート番号です。デフォルトの5000を使用します。")
    
    if config['tunnel_method'] == 'ngrok':
        print()
        print("ngrok設定:")
        print("ngrokを有効にしますか？ (y/n): ", end='')
        enabled = input().strip().lower() == 'y'
        config['ngrok']['enabled'] = enabled
        
        if enabled:
            print("ngrok認証トークン (空欄可、無料プランでも使用可能): ", end='')
            auth_token = input().strip()
            if auth_token:
                config['ngrok']['auth_token'] = auth_token
            
            print("リージョン (us/jp/ap/eu/au/sa/in, デフォルト: us): ", end='')
            region = input().strip()
            if region:
                config['ngrok']['region'] = region
    
    elif config['tunnel_method'] == 'cloudflare':
        print()
        print("Cloudflare Tunnel設定:")
        print("Cloudflare Tunnelを有効にしますか？ (y/n): ", end='')
        enabled = input().strip().lower() == 'y'
        config['cloudflare']['enabled'] = enabled
        
        if enabled:
            print("Cloudflare Tunnelトークンを取得してください:")
            print("  https://one.dash.cloudflare.com/")
            print("Tunnel Token: ", end='')
            tunnel_token = input().strip()
            if tunnel_token:
                config['cloudflare']['tunnel_token'] = tunnel_token
    
    elif config['tunnel_method'] == 'custom':
        print()
        print("カスタムトンネル設定:")
        print("カスタムトンネルを有効にしますか？ (y/n): ", end='')
        enabled = input().strip().lower() == 'y'
        config['custom_tunnel']['enabled'] = enabled
        
        if enabled:
            print("トンネル起動コマンド: ", end='')
            command = input().strip()
            if command:
                config['custom_tunnel']['command'] = command
    
    # 設定を保存
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 60)
    print("設定完了！")
    print("=" * 60)
    print(f"設定ファイル: {config_path}")
    print()
    print("設定内容:")
    print(f"  トンネル方法: {config['tunnel_method']}")
    print(f"  ローカルポート: {config['local_port']}")
    if config['tunnel_method'] == 'ngrok':
        print(f"  ngrok有効: {config['ngrok']['enabled']}")
    elif config['tunnel_method'] == 'cloudflare':
        print(f"  Cloudflare有効: {config['cloudflare']['enabled']}")
    print()
    print("トンネルを開始するには:")
    print("  python scripts/remote_server_tunnel.py --start")
    print()

if __name__ == '__main__':
    main()





