#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リモート転送設定ツール
ネットワーク共有経由で別PCに転送する設定を行います
"""

import json
import sys
from pathlib import Path

def main():
    """設定を対話的に入力"""
    print("=" * 60)
    print("リモート転送設定")
    print("=" * 60)
    print()
    
    # 設定ファイルのパス
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / 'config' / 'remote_transfer_config.json'
    config_dir = config_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # 既存の設定を読み込み
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {
            "enabled": False,
            "transfer_method": "network_share",
            "remote_pc": {},
            "transfer_items": {
                "models": True,
                "checkpoints": True,
                "logs": True,
                "config": True,
                "training_history": True
            },
            "auto_transfer_on_complete": True,
            "compress_before_transfer": True
        }
    
    print("リモート転送を有効にしますか？ (y/n): ", end='')
    enabled = input().strip().lower() == 'y'
    config['enabled'] = enabled
    
    if not enabled:
        print("リモート転送を無効にしました")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return
    
    print()
    print("転送方法を選択してください:")
    print("1. ネットワーク共有 (推奨)")
    print("2. SCP (SSH)")
    print("3. FTP")
    print("選択 (1-3): ", end='')
    method_choice = input().strip()
    
    method_map = {'1': 'network_share', '2': 'scp', '3': 'ftp'}
    config['transfer_method'] = method_map.get(method_choice, 'network_share')
    
    if config['transfer_method'] == 'network_share':
        print()
        print("ネットワーク共有の設定:")
        print("例: \\\\192.168.1.100\\WasherInspection")
        print("共有パス: ", end='')
        share_path = input().strip()
        
        if not share_path:
            print("エラー: 共有パスが入力されていません")
            return
        
        config['remote_pc'] = {
            "share_path": share_path,
            "target_folder": "remote_models"
        }
        
        print("転送先フォルダ名 (デフォルト: remote_models): ", end='')
        target_folder = input().strip()
        if target_folder:
            config['remote_pc']['target_folder'] = target_folder
    
    print()
    print("自動転送を有効にしますか？ (学習完了時に自動転送) (y/n): ", end='')
    auto_transfer = input().strip().lower() == 'y'
    config['auto_transfer_on_complete'] = auto_transfer
    
    print()
    print("転送前に圧縮しますか？ (y/n): ", end='')
    compress = input().strip().lower() == 'y'
    config['compress_before_transfer'] = compress
    
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
    print(f"  有効: {config['enabled']}")
    print(f"  転送方法: {config['transfer_method']}")
    if config['transfer_method'] == 'network_share':
        print(f"  共有パス: {config['remote_pc'].get('share_path', 'N/A')}")
        print(f"  転送先フォルダ: {config['remote_pc'].get('target_folder', 'N/A')}")
    print(f"  自動転送: {config['auto_transfer_on_complete']}")
    print(f"  圧縮: {config['compress_before_transfer']}")
    print()

if __name__ == '__main__':
    main()





