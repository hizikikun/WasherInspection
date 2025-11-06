#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リモートアクセス自動セットアップ
ngrokやCloudflare Tunnelの設定を自動化
"""

import os
import sys
import json
import subprocess
import shutil
import platform
import urllib.request
import zipfile
import tempfile
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
config_dir = project_root / 'config'
config_file = config_dir / 'remote_tunnel_config.json'
remote_server_config = config_dir / 'remote_server_config.json'


def check_ngrok_installed():
    """ngrokがインストールされているか確認"""
    try:
        result = subprocess.run(['ngrok', 'version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def install_ngrok():
    """ngrokを自動インストール"""
    print("[INFO] ngrokをインストール中...")
    
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # ダウンロードURLを決定
    if system == 'windows':
        if '64' in machine or 'x86_64' in machine or 'amd64' in machine:
            url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip"
            exe_name = "ngrok.exe"
        else:
            url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-386.zip"
            exe_name = "ngrok.exe"
    elif system == 'darwin':  # macOS
        if 'arm' in machine or 'aarch64' in machine:
            url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-darwin-arm64.zip"
            exe_name = "ngrok"
        else:
            url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-darwin-amd64.zip"
            exe_name = "ngrok"
    else:  # Linux
        if 'arm' in machine or 'aarch64' in machine:
            url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.tgz"
            exe_name = "ngrok"
        elif '64' in machine or 'x86_64' in machine or 'amd64' in machine:
            url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
            exe_name = "ngrok"
        else:
            url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-386.tgz"
            exe_name = "ngrok"
    
    try:
        # 一時ディレクトリにダウンロード
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"[INFO] ngrokをダウンロード中: {url}")
            zip_path = Path(tmpdir) / "ngrok.zip"
            
            urllib.request.urlretrieve(url, zip_path)
            
            # 解凍
            print("[INFO] ngrokを解凍中...")
            if system == 'windows':
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
            else:
                import tarfile
                with tarfile.open(zip_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(tmpdir)
            
            # 実行ファイルを見つける
            extracted_dir = Path(tmpdir)
            ngrok_exe = None
            for file in extracted_dir.rglob(exe_name):
                if file.is_file():
                    ngrok_exe = file
                    break
            
            if not ngrok_exe:
                print("[ERROR] ngrok実行ファイルが見つかりません")
                return False
            
            # インストール先を決定
            if system == 'windows':
                # Windows: ユーザーのbinディレクトリまたはプロジェクトディレクトリ
                install_dir = Path.home() / 'bin'
                if not install_dir.exists():
                    install_dir = project_root / 'bin'
                install_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Linux/macOS: /usr/local/bin または ~/bin
                install_dir = Path('/usr/local/bin')
                if not install_dir.exists() or not os.access(install_dir, os.W_OK):
                    install_dir = Path.home() / 'bin'
                install_dir.mkdir(parents=True, exist_ok=True)
            
            # コピー
            target_path = install_dir / exe_name
            shutil.copy2(ngrok_exe, target_path)
            
            # 実行権限を付与（Linux/macOS）
            if system != 'windows':
                os.chmod(target_path, 0o755)
            
            print(f"[SUCCESS] ngrokをインストールしました: {target_path}")
            
            # PATHに追加（Windowsの場合、環境変数を設定）
            if system == 'windows':
                # 現在のセッションのPATHに追加
                current_path = os.environ.get('PATH', '')
                if str(install_dir) not in current_path:
                    os.environ['PATH'] = f"{install_dir};{current_path}"
                    print(f"[INFO] このセッションのPATHに追加しました: {install_dir}")
                    print("[INFO] 永続的に有効にするには、システム環境変数PATHに手動で追加してください")
            
            return True
            
    except Exception as e:
        print(f"[ERROR] ngrokインストールエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def setup_ngrok_config():
    """ngrok設定を自動セットアップ"""
    print("[INFO] ngrok設定を自動セットアップ中...")
    
    # 設定ディレクトリを作成
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # デフォルト設定
    config = {
        "tunnel_method": "ngrok",
        "local_port": 5000,
        "ngrok": {
            "enabled": True,
            "auth_token": "",
            "region": "jp",  # 日本リージョン
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
    
    # 既存の設定がある場合は読み込み
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
                # 既存の設定を保持しつつ、新しい設定をマージ
                if 'ngrok' in existing_config:
                    config['ngrok'].update(existing_config.get('ngrok', {}))
                config['tunnel_method'] = existing_config.get('tunnel_method', 'ngrok')
                config['local_port'] = existing_config.get('local_port', 5000)
        except Exception as e:
            print(f"[WARN] 既存設定の読み込みエラー: {e}")
    
    # 設定を保存
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS] ngrok設定を保存しました: {config_file}")
    return True


def setup_remote_server_config():
    """リモートサーバー設定を自動セットアップ"""
    print("[INFO] リモートサーバー設定を自動セットアップ中...")
    
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 5000,
        "password_protected": False,
        "password": "",
        "allowed_ips": [],
        "auto_start": False
    }
    
    # 既存の設定がある場合は読み込み
    if remote_server_config.exists():
        try:
            with open(remote_server_config, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
                config.update(existing_config)
        except Exception as e:
            print(f"[WARN] 既存設定の読み込みエラー: {e}")
    
    # 設定を保存
    with open(remote_server_config, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS] リモートサーバー設定を保存しました: {remote_server_config}")
    return True


def check_flask_installed():
    """Flaskがインストールされているか確認"""
    try:
        import flask
        import flask_cors
        return True
    except ImportError:
        return False


def install_flask():
    """Flaskをインストール"""
    print("[INFO] Flaskとflask-corsをインストール中...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'flask', 'flask-cors'],
                      check=True,
                      capture_output=True,
                      text=True)
        print("[SUCCESS] Flaskとflask-corsをインストールしました")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Flaskインストールエラー: {e}")
        return False


def main():
    """メイン関数"""
    print("=" * 60)
    print("リモートアクセス自動セットアップ")
    print("=" * 60)
    print()
    
    success = True
    
    # 1. Flaskの確認とインストール
    print("[1/4] Flaskの確認...")
    if not check_flask_installed():
        print("[INFO] Flaskがインストールされていません。インストールします。")
        if not install_flask():
            print("[ERROR] Flaskのインストールに失敗しました")
            success = False
    else:
        print("[OK] Flaskは既にインストールされています")
    print()
    
    # 2. ngrokの確認とインストール
    print("[2/4] ngrokの確認...")
    if not check_ngrok_installed():
        print("[INFO] ngrokがインストールされていません。インストールします。")
        if not install_ngrok():
            print("[WARN] ngrokの自動インストールに失敗しました")
            print("[INFO] 手動でインストールしてください: https://ngrok.com/download")
            print("[INFO] または、後で 'python scripts/configure_tunnel.py' を実行して設定してください")
        else:
            # 再確認
            if not check_ngrok_installed():
                print("[WARN] ngrokがインストールされましたが、PATHに追加されていない可能性があります")
                print("[INFO] システムを再起動するか、PATHを手動で設定してください")
    else:
        print("[OK] ngrokは既にインストールされています")
    print()
    
    # 3. ngrok設定
    print("[3/4] ngrok設定のセットアップ...")
    if setup_ngrok_config():
        print("[OK] ngrok設定が完了しました")
    else:
        print("[ERROR] ngrok設定に失敗しました")
        success = False
    print()
    
    # 4. リモートサーバー設定
    print("[4/4] リモートサーバー設定のセットアップ...")
    if setup_remote_server_config():
        print("[OK] リモートサーバー設定が完了しました")
    else:
        print("[ERROR] リモートサーバー設定に失敗しました")
        success = False
    print()
    
    print("=" * 60)
    if success:
        print("自動セットアップ完了！")
        print("=" * 60)
        print()
        print("次のステップ:")
        print("1. 統合アプリの「🌍 インターネット経由アクセス」ボタンをクリック")
        print("2. または、以下のコマンドでトンネルを起動:")
        print("   python scripts/remote_server_tunnel.py --start")
        print()
        print("注意:")
        print("- ngrok無料プランでは認証トークンは不要ですが、")
        print("  より安定した接続のために認証トークンを設定することを推奨します")
        print("- 認証トークンは https://dashboard.ngrok.com/get-started/your-authtoken から取得できます")
    else:
        print("セットアップ中にエラーが発生しました")
        print("上記のエラーメッセージを確認してください")
    print("=" * 60)


if __name__ == '__main__':
    main()





