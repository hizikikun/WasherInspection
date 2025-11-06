#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リモート操作サーバー（インターネット経由対応版）
ngrokやCloudflare Tunnelを使用してインターネット経由でアクセス可能
"""

import os
import sys
import json
import subprocess
import threading
import time
import socket
import shutil
from pathlib import Path

# UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# プロジェクトルート
project_root = Path(__file__).resolve().parents[1]
config_file = project_root / 'config' / 'remote_tunnel_config.json'


def get_default_config():
    """デフォルト設定"""
    return {
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


def load_config():
    """設定を読み込み"""
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return {**get_default_config(), **config}
        except Exception as e:
            print(f"[WARN] 設定ファイル読み込みエラー: {e}")
    
    return get_default_config()


def save_config(config):
    """設定を保存"""
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


class TunnelManager:
    """トンネル管理クラス"""
    
    def __init__(self):
        self.config = load_config()
        self.tunnel_process = None
        self.tunnel_url = None
    
    def find_ngrok(self):
        """ngrokのパスを検索（プロジェクト内を優先）"""
        # プロジェクト内のbinディレクトリを優先
        local_ngrok = project_root / 'bin' / 'ngrok.exe'
        if local_ngrok.exists():
            return str(local_ngrok)
        
        # PATHから探す
        ngrok_path = shutil.which('ngrok')
        if ngrok_path:
            return ngrok_path
        
        # Windowsの場合、ngrok.exeも試す
        if sys.platform.startswith('win'):
            ngrok_exe = shutil.which('ngrok.exe')
            if ngrok_exe:
                return ngrok_exe
        
        return None
    
    def start_tunnel(self):
        """トンネルを開始"""
        method = self.config.get('tunnel_method', 'ngrok')
        
        if method == 'ngrok':
            return self.start_ngrok()
        elif method == 'cloudflare':
            return self.start_cloudflare()
        elif method == 'custom':
            return self.start_custom()
        else:
            print(f"[ERROR] 不明なトンネル方法: {method}")
            return False
    
    def start_ngrok(self):
        """ngrokトンネルを開始"""
        ngrok_config = self.config.get('ngrok', {})
        
        if not ngrok_config.get('enabled', False):
            print("[INFO] ngrokは無効になっています")
            return False
        
        # ngrokのパスを検索
        ngrok_path = self.find_ngrok()
        if not ngrok_path:
            print("[ERROR] ngrokが見つかりません")
            print("[INFO] ngrokをインストールしてください: https://ngrok.com/download")
            print(f"[INFO] または、プロジェクトのbinディレクトリにngrok.exeを配置してください")
            return False
        
        # ngrokがインストールされているか確認
        try:
            result = subprocess.run([ngrok_path, 'version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("[ERROR] ngrokの実行に失敗しました")
            print(f"[INFO] パス: {ngrok_path}")
            return False
        
        local_port = self.config.get('local_port', 5000)
        
        # 認証トークンの確認と設定
        auth_token = ngrok_config.get('auth_token', '')
        if not auth_token:
            print("[ERROR] ngrok認証トークンが設定されていません")
            print("[INFO] ngrokを使用するには認証トークンが必要です")
            print("[INFO] 1. https://dashboard.ngrok.com/signup でアカウントを作成")
            print("[INFO] 2. https://dashboard.ngrok.com/get-started/your-authtoken で認証トークンを取得")
            print("[INFO] 3. config/remote_tunnel_config.json の 'auth_token' にトークンを設定")
            return False
        
        # ngrok設定ファイルに認証トークンを保存
        ngrok_config_dir = Path.home() / '.ngrok2'
        ngrok_config_file = ngrok_config_dir / 'ngrok.yml'
        
        # 設定ファイルが存在する場合は既存の内容を確認
        existing_token = None
        has_version = False
        config_valid = False
        
        if ngrok_config_file.exists():
            try:
                with open(ngrok_config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        line = line.strip()
                        if line.startswith('authtoken:'):
                            existing_token = line.split(':', 1)[1].strip()
                        if line.startswith('version:'):
                            has_version = True
                    # versionとauthtokenの両方が存在する場合のみ有効
                    config_valid = has_version and existing_token
            except Exception as e:
                print(f"[WARN] 設定ファイルの読み込みエラー: {e}")
                config_valid = False
        
        # 設定ファイルを更新（無効な場合、トークンが異なる場合、versionがない場合）
        need_update = (not config_valid or existing_token != auth_token or not has_version)
        
        if need_update:
            ngrok_config_dir.mkdir(parents=True, exist_ok=True)
            # 既存の設定ファイルを削除（存在する場合）
            if ngrok_config_file.exists():
                try:
                    ngrok_config_file.unlink()
                    print("[INFO] 既存の設定ファイルを削除しました")
                except Exception as e:
                    print(f"[WARN] 設定ファイルの削除に失敗: {e}")
            
            # 新しい形式で設定ファイルを作成
            try:
                with open(ngrok_config_file, 'w', encoding='utf-8') as f:
                    # 新しいngrok形式ではversionフィールドが必須
                    f.write('version: "2"\n')
                    f.write(f'authtoken: {auth_token}\n')
                print("[INFO] ngrok設定ファイルを作成しました")
            except Exception as e:
                print(f"[ERROR] 設定ファイルの作成に失敗: {e}")
                return False
            
            # 設定ファイルが正しく作成されたか確認
            if not ngrok_config_file.exists():
                print("[ERROR] 設定ファイルの作成に失敗しました")
                return False
        
        # ngrokコマンドを構築
        cmd = [ngrok_path, 'http', str(local_port)]
        
        # 注意: --regionフラグは非推奨のため削除（ngrokが自動的に最適なリージョンを選択）
        # region = ngrok_config.get('region', 'us')
        # if region:
        #     cmd.extend(['--region', region])
        
        # サブドメイン指定（有料プランのみ）
        subdomain = ngrok_config.get('subdomain', '')
        if subdomain:
            cmd.extend(['--subdomain', subdomain])
        
        try:
            print(f"[INFO] ngrokトンネルを開始中... (ポート: {local_port})")
            print(f"[DEBUG] 実行コマンド: {' '.join(cmd)}")
            
            # ngrokプロセスを起動（バックグラウンドで実行）
            self.tunnel_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform.startswith('win') else 0
            )
            
            # プロセスが正常に起動したか確認
            time.sleep(1)
            if self.tunnel_process.poll() is not None:
                # プロセスが既に終了している（エラー）
                stdout, stderr = self.tunnel_process.communicate()
                print(f"[ERROR] ngrokプロセスの起動に失敗しました")
                if stderr:
                    print(f"[ERROR] エラー出力: {stderr}")
                if stdout:
                    print(f"[ERROR] 標準出力: {stdout}")
                return False
            
            # ngrok APIが利用可能になるまで待機（最大30秒、2秒間隔でリトライ）
            print("[INFO] ngrok APIの準備を待機中...")
            max_retries = 15
            retry_count = 0
            self.tunnel_url = None
            
            while retry_count < max_retries:
                self.tunnel_url = self.get_ngrok_url()
                if self.tunnel_url:
                    break
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2)
            
            # プロセスの状態を再確認
            if self.tunnel_process.poll() is not None:
                stdout, stderr = self.tunnel_process.communicate()
                print(f"[ERROR] ngrokプロセスが終了しました")
                if stderr:
                    print(f"[ERROR] エラー出力: {stderr}")
                if stdout:
                    print(f"[ERROR] 標準出力: {stdout}")
                return False
            
            if self.tunnel_url:
                print(f"[SUCCESS] ngrokトンネルが開始されました")
                print(f"[INFO] アクセスURL: {self.tunnel_url}")
                return True
            else:
                print("[WARN] ngrok URLを取得できませんでした")
                print("[INFO] ngrokプロセスは実行中ですが、APIに接続できません")
                print("[INFO] ブラウザで http://localhost:4040 にアクセスして、ngrokの状態を確認してください")
                return False
                
        except Exception as e:
            print(f"[ERROR] ngrok起動エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_ngrok_url(self):
        """ngrok APIからURLを取得"""
        try:
            import requests
            response = requests.get('http://localhost:4040/api/tunnels', timeout=3)
            if response.status_code == 200:
                data = response.json()
                tunnels = data.get('tunnels', [])
                if tunnels:
                    public_url = tunnels[0].get('public_url', '')
                    if public_url.startswith('http://'):
                        return public_url.replace('http://', 'https://')
                    return public_url
        except requests.exceptions.ConnectionError:
            # APIがまだ利用できない（静かに失敗）
            pass
        except Exception as e:
            # その他のエラーは初回のみ表示
            pass
        
        return None
    
    def start_cloudflare(self):
        """Cloudflare Tunnelを開始"""
        cloudflare_config = self.config.get('cloudflare', {})
        
        if not cloudflare_config.get('enabled', False):
            print("[INFO] Cloudflare Tunnelは無効になっています")
            return False
        
        tunnel_token = cloudflare_config.get('tunnel_token', '')
        if not tunnel_token:
            print("[ERROR] Cloudflare Tunnelトークンが設定されていません")
            return False
        
        # cloudflaredがインストールされているか確認
        cloudflared_path = shutil.which('cloudflared')
        if not cloudflared_path:
            print("[ERROR] cloudflaredが見つかりません")
            print("[INFO] cloudflaredをインストールしてください: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/")
            return False
        
        local_port = self.config.get('local_port', 5000)
        
        try:
            print(f"[INFO] Cloudflare Tunnelを開始中... (ポート: {local_port})")
            cmd = [cloudflared_path, 'tunnel', '--url', f'http://localhost:{local_port}']
            
            self.tunnel_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # URLを取得（cloudflaredの出力から）
            time.sleep(3)
            if self.tunnel_process.poll() is None:
                print("[SUCCESS] Cloudflare Tunnelが開始されました")
                print("[INFO] 出力を確認してください")
                return True
            else:
                stdout, stderr = self.tunnel_process.communicate()
                print(f"[ERROR] Cloudflare Tunnelの起動に失敗しました")
                if stderr:
                    print(f"[ERROR] エラー出力: {stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Cloudflare Tunnel起動エラー: {e}")
            return False
    
    def start_custom(self):
        """カスタムトンネルを開始"""
        custom_config = self.config.get('custom_tunnel', {})
        
        if not custom_config.get('enabled', False):
            print("[INFO] カスタムトンネルは無効になっています")
            return False
        
        command = custom_config.get('command', '')
        if not command:
            print("[ERROR] カスタムトンネルコマンドが設定されていません")
            return False
        
        try:
            print("[INFO] カスタムトンネルを開始中...")
            # コマンドをシェルで実行
            self.tunnel_process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            time.sleep(2)
            if self.tunnel_process.poll() is None:
                print("[SUCCESS] カスタムトンネルが開始されました")
                return True
            else:
                stdout, stderr = self.tunnel_process.communicate()
                print(f"[ERROR] カスタムトンネルの起動に失敗しました")
                if stderr:
                    print(f"[ERROR] エラー出力: {stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] カスタムトンネル起動エラー: {e}")
            return False
    
    def stop_tunnel(self):
        """トンネルを停止"""
        if self.tunnel_process:
            try:
                self.tunnel_process.terminate()
                self.tunnel_process.wait(timeout=5)
                print("[INFO] トンネルを停止しました")
                return True
            except subprocess.TimeoutExpired:
                self.tunnel_process.kill()
                print("[INFO] トンネルを強制終了しました")
                return True
            except Exception as e:
                print(f"[ERROR] トンネル停止エラー: {e}")
                return False
        return True


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='リモートサーバートンネル管理')
    parser.add_argument('--start', action='store_true', help='トンネルを開始')
    parser.add_argument('--stop', action='store_true', help='トンネルを停止')
    parser.add_argument('--status', action='store_true', help='トンネルの状態を表示')
    
    args = parser.parse_args()
    
    manager = TunnelManager()
    
    if args.start:
        print("=" * 60)
        print("インターネット経由アクセストンネルを開始します")
        print("=" * 60)
        print()
        
        if manager.start_tunnel():
            print()
            print("=" * 60)
            print("トンネルが正常に開始されました")
            print("=" * 60)
            print()
            print("トンネルを停止するには Ctrl+C を押してください")
            try:
                while True:
                    time.sleep(1)
                    if manager.tunnel_process and manager.tunnel_process.poll() is not None:
                        print("[WARN] トンネルプロセスが終了しました")
                        break
            except KeyboardInterrupt:
                print()
                print("[INFO] トンネルを停止中...")
                manager.stop_tunnel()
        else:
            print()
            print("=" * 60)
            print("トンネルの開始に失敗しました")
            print("=" * 60)
            sys.exit(1)
            
    elif args.stop:
        if manager.stop_tunnel():
            print("トンネルを停止しました")
        else:
            print("トンネルを停止できませんでした")
            sys.exit(1)
            
    elif args.status:
        if manager.tunnel_process and manager.tunnel_process.poll() is None:
            print("トンネルは実行中です")
            if manager.tunnel_url:
                print(f"アクセスURL: {manager.tunnel_url}")
        else:
            print("トンネルは実行されていません")
    else:
        parser.print_help()
        print()
        print("使用例:")
        print("  python remote_server_tunnel.py --start   # トンネルを開始")
        print("  python remote_server_tunnel.py --stop    # トンネルを停止")
        print("  python remote_server_tunnel.py --status  # 状態を確認")


if __name__ == '__main__':
    main()
