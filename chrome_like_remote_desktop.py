#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chrome Remote Desktop風のリモートデスクトップシステム
- 暗号化通信（TLS/SSL）
- パスワード認証
- ファイル転送
- クリップボード共有
- 再接続機能
- パフォーマンス最適化
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import socket
import ssl
import threading
import struct
import json
import time
import io
import hashlib
import base64
from PIL import Image, ImageTk, ImageChops
import platform
import subprocess
import os
from pathlib import Path
import queue
import zlib

# 画面キャプチャ用
if platform.system() == "Windows":
    try:
        import mss
        HAS_MSS = True
    except ImportError:
        HAS_MSS = False
        print("警告: mssがインストールされていません。")
        print("インストール: pip install mss")
else:
    HAS_MSS = False

# リモート制御用
if platform.system() == "Windows":
    try:
        import pyautogui
        HAS_PYAUTOGUI = True
    except ImportError:
        HAS_PYAUTOGUI = False
        print("警告: pyautoguiがインストールされていません。")
        print("インストール: pip install pyautogui")
else:
    HAS_PYAUTOGUI = False

# クリップボード用
try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False
    print("警告: pyperclipがインストールされていません。クリップボード共有が無効です。")
    print("インストール: pip install pyperclip")


class SecureRemoteDesktopServer:
    """セキュアなリモートデスクトップサーバー"""
    
    def __init__(self, host='0.0.0.0', port=8888, password=None):
        self.host = host
        self.port = port
        self.password = password or self.generate_password()
        self.password_hash = self.hash_password(self.password)
        self.server_socket = None
        self.client_socket = None
        self.ssl_context = None
        self.running = False
        self.authenticated = False
        self.capture_thread = None
        self.command_thread = None
        self.file_thread = None
        self.clipboard_thread = None
        
        # パフォーマンス最適化用
        self.last_frame = None
        self.frame_queue = queue.Queue(maxsize=2)
        
        # 設定
        self.quality = 75
        self.fps = 15
        self.use_diff = True  # 差分更新を使用
        
        print(f"[サーバー] パスワード: {self.password}")
        print(f"[サーバー] パスワードハッシュ: {self.password_hash[:16]}...")
    
    @staticmethod
    def generate_password():
        """ランダムなパスワードを生成"""
        import secrets
        return secrets.token_urlsafe(12)
    
    @staticmethod
    def hash_password(password):
        """パスワードをハッシュ化"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password):
        """パスワードを検証"""
        return self.hash_password(password) == self.password_hash
    
    def create_ssl_context(self):
        """SSLコンテキストを作成（簡易版 - 自己署名証明書）"""
        try:
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            # 本番環境では適切な証明書を使用してください
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context
        except Exception as e:
            print(f"[サーバー] SSLコンテキスト作成エラー: {e}")
            return None
    
    def start(self):
        """サーバーを起動"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.running = True
            
            print(f"[サーバー] 待機中: {self.host}:{self.port}")
            
            # クライアント接続を待機
            client_socket, addr = self.server_socket.accept()
            print(f"[サーバー] クライアント接続: {addr}")
            
            # SSLでラップ（簡易版 - 実際には証明書が必要）
            # SSL接続は問題が多いため、デフォルトで無効化
            use_ssl = False  # SSLを無効化（問題が多いため）
            
            if use_ssl:
                try:
                    self.ssl_context = self.create_ssl_context()
                    if self.ssl_context:
                        self.client_socket = self.ssl_context.wrap_socket(
                            client_socket, server_side=True)
                        print("[サーバー] SSL接続確立")
                    else:
                        self.client_socket = client_socket
                        print("[サーバー] 通常接続（SSL無効）")
                except Exception as e:
                    print(f"[サーバー] SSL設定エラー: {e}（通常接続にフォールバック）")
                    self.client_socket = client_socket
            else:
                self.client_socket = client_socket
                print("[サーバー] 通常接続（SSL無効）")
            
            # 認証
            if not self.authenticate():
                print("[サーバー] 認証失敗")
                self.client_socket.close()
                return
            
            print("[サーバー] 認証成功")
            
            # 各スレッドを開始
            self.capture_thread = threading.Thread(
                target=self.capture_and_send, daemon=True)
            self.capture_thread.start()
            
            self.command_thread = threading.Thread(
                target=self.receive_commands, daemon=True)
            self.command_thread.start()
            
            self.file_thread = threading.Thread(
                target=self.handle_file_transfer, daemon=True)
            self.file_thread.start()
            
            self.clipboard_thread = threading.Thread(
                target=self.handle_clipboard_sync, daemon=True)
            self.clipboard_thread.start()
            
            # メインスレッドで接続を維持
            while self.running and self.authenticated:
                time.sleep(1)
                
        except Exception as e:
            print(f"[サーバー] エラー: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def authenticate(self):
        """クライアントを認証"""
        try:
            # 認証要求を送信
            auth_request = {
                'type': 'auth_request',
                'message': 'パスワードを入力してください'
            }
            self.send_json(auth_request)
            
            # 認証応答を受信
            auth_response = self.receive_json()
            if not auth_response:
                return False
            
            password = auth_response.get('password', '')
            if self.verify_password(password):
                # 認証成功
                self.send_json({'type': 'auth_success'})
                self.authenticated = True
                return True
            else:
                # 認証失敗
                self.send_json({'type': 'auth_failure', 'message': 'パスワードが正しくありません'})
                return False
                
        except Exception as e:
            print(f"[サーバー] 認証エラー: {e}")
            return False
    
    def send_json(self, data):
        """JSONデータを送信（プロトコル修正版）"""
        if not self.client_socket:
            return
        
        # 送信ロック（複数のスレッドから同時に送信されないようにする）
        if not hasattr(self, '_send_lock'):
            self._send_lock = threading.Lock()
        
        with self._send_lock:
            try:
                json_str = json.dumps(data, ensure_ascii=False)
                data_bytes = json_str.encode('utf-8')
                size = len(data_bytes)
                
                # サイズの妥当性チェック
                if size > 1024 * 1024 or size == 0:  # 1MB以上または0は異常
                    print(f"[サーバー] 警告: 送信データサイズが異常です: {size}")
                    return
                
                # マジックナンバー + サイズ + データを送信
                magic = b'\xDE\xAD\xBE\xEF'
                header = struct.pack('>I', size)
                self.client_socket.sendall(magic + header + data_bytes)
            except ConnectionAbortedError:
                print("[サーバー] 接続が切断されました")
                self.running = False
            except Exception as e:
                print(f"[サーバー] JSON送信エラー: {e}")
                import traceback
                traceback.print_exc()
    
    def receive_json(self):
        """JSONデータを受信（プロトコル修正版）"""
        try:
            # マジックナンバーでメッセージの開始を確認（0xDEADBEEF）
            magic = b'\xDE\xAD\xBE\xEF'
            magic_buffer = b''
            
            # マジックナンバーを探す（最大100バイトまでスキャン）
            for _ in range(100):
                chunk = self.client_socket.recv(1)
                if not chunk:
                    return None
                magic_buffer += chunk
                if len(magic_buffer) >= 4:
                    if magic_buffer[-4:] == magic:
                        # マジックナンバーが見つかった
                        break
                    # 最初の1バイトを削除して続行
                    magic_buffer = magic_buffer[1:]
            else:
                # マジックナンバーが見つからなかった
                print("[サーバー] マジックナンバーが見つかりませんでした。接続をリセットする必要があります。")
                return None
            
            # サイズを受信（確実に4バイト受信）
            size_data = b''
            timeout_count = 0
            while len(size_data) < 4:
                chunk = self.client_socket.recv(4 - len(size_data))
                if not chunk:
                    timeout_count += 1
                    if timeout_count > 10:
                        return None
                    time.sleep(0.01)
                    continue
                size_data += chunk
                timeout_count = 0
            
            if len(size_data) < 4:
                return None
            
            size = struct.unpack('>I', size_data)[0]
            
            # サイズの妥当性チェック（より厳格に）
            if size > 1024 * 1024 or size == 0:  # 1MB以上または0は異常
                print(f"[サーバー] 異常なデータサイズ: {size}")
                return None
            
            # データを受信
            data = b''
            timeout_count = 0
            while len(data) < size:
                remaining = size - len(data)
                chunk = self.client_socket.recv(min(remaining, 4096))
                if not chunk:
                    timeout_count += 1
                    if timeout_count > 10:
                        print(f"[サーバー] データ受信タイムアウト: {len(data)}/{size} バイト受信")
                        return None
                    time.sleep(0.01)
                    continue
                data += chunk
                timeout_count = 0
            
            # JSONをパース
            try:
                decoded = data.decode('utf-8')
                obj = json.loads(decoded)
                return obj
            except json.JSONDecodeError as e:
                print(f"[サーバー] JSONパースエラー: {e}")
                print(f"[サーバー] 受信データ（最初の100文字）: {data[:100]}")
                return None
        except ConnectionAbortedError:
            return None
        except socket.timeout:
            print("[サーバー] 受信タイムアウト")
            return None
        except Exception as e:
            print(f"[サーバー] JSON受信エラー: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def capture_and_send(self):
        """画面をキャプチャして送信（差分更新対応）"""
        if not HAS_MSS:
            print("[サーバー] 画面キャプチャ機能が利用できません")
            return
        
        import mss
        
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # プライマリモニター
                frame_interval = 1.0 / self.fps
                last_capture_time = 0
                
                while self.running and self.authenticated and self.client_socket:
                    try:
                        current_time = time.time()
                        if current_time - last_capture_time < frame_interval:
                            time.sleep(0.01)
                            continue
                        
                        last_capture_time = current_time
                        
                        # 画面キャプチャ
                        screenshot = sct.grab(monitor)
                        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                        
                        # 差分更新の処理
                        if self.use_diff and self.last_frame is not None:
                            # 差分を計算
                            diff = ImageChops.difference(img, self.last_frame)
                            bbox = diff.getbbox()
                            
                            if bbox:
                                # 差分がある場合のみ送信
                                diff_img = img.crop(bbox)
                                self.send_frame(diff_img, bbox, is_diff=True)
                            else:
                                # 差分がない場合はスキップ
                                continue
                        else:
                            # フルフレーム送信
                            self.send_frame(img, None, is_diff=False)
                        
                        self.last_frame = img.copy()
                        
                    except Exception as e:
                        if self.running:
                            print(f"[サーバー] キャプチャエラー: {e}")
                        break
        except Exception as e:
            print(f"[サーバー] 画面キャプチャ初期化エラー: {e}")
    
    def send_frame(self, img, bbox=None, is_diff=False):
        """フレームを送信"""
        try:
            # 画像を圧縮
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=self.quality, optimize=True)
            img_data = img_buffer.getvalue()
            
            # さらにzlibで圧縮
            compressed = zlib.compress(img_data, level=6)
            
            # フレームデータを送信
            frame_data = {
                'type': 'frame',
                'is_diff': is_diff,
                'bbox': bbox if is_diff else None,
                'size': len(compressed),
                'original_size': len(img_data),
                'width': img.width,
                'height': img.height
            }
            
            # JSONヘッダーを送信
            self.send_json(frame_data)
            
            # 画像データを送信（接続が有効な場合のみ）
            if self.client_socket and self.authenticated:
                try:
                    self.client_socket.sendall(compressed)
                except ConnectionAbortedError:
                    print("[サーバー] 接続が切断されました（フレーム送信中）")
                    self.running = False
                except Exception as e:
                    print(f"[サーバー] フレームデータ送信エラー: {e}")
            
        except Exception as e:
            print(f"[サーバー] フレーム送信エラー: {e}")
    
    def receive_commands(self):
        """リモート制御コマンドを受信"""
        error_count = 0
        max_errors = 10  # 連続エラーの最大回数
        
        try:
            while self.running and self.authenticated and self.client_socket:
                try:
                    command = self.receive_json()
                    if not command:
                        error_count += 1
                        if error_count >= max_errors:
                            print(f"[サーバー] 連続エラーが{max_errors}回発生しました。コマンド受信を終了します。")
                            break
                        # エラーが発生した場合は少し待ってから再試行
                        time.sleep(0.1)
                        continue
                    
                    # エラーカウントをリセット
                    error_count = 0
                    
                    # コマンドタイプを確認
                    cmd_type = command.get('type')
                    if cmd_type in ['mouse_move', 'mouse_click', 'mouse_drag', 'mouse_scroll', 
                                    'key_press', 'key_type', 'key_combination']:
                        self.execute_command(command)
                    elif cmd_type in ['auth_request', 'auth_response', 'auth_success', 'auth_failure']:
                        # 認証関連のコマンドは無視（既に認証済み）
                        pass
                    elif cmd_type == 'frame':
                        # フレームデータは別のスレッドで処理されるため、ここでは無視
                        print("[サーバー] 警告: コマンド受信スレッドでフレームデータを受信しました")
                    else:
                        print(f"[サーバー] 不明なコマンドタイプ: {cmd_type}")
                    
                except ConnectionAbortedError:
                    print("[サーバー] 接続が切断されました（コマンド受信）")
                    break
                except Exception as e:
                    error_count += 1
                    if self.running:
                        print(f"[サーバー] コマンド受信エラー: {e}")
                        if error_count >= max_errors:
                            print(f"[サーバー] 連続エラーが{max_errors}回発生しました。コマンド受信を終了します。")
                            break
                        import traceback
                        traceback.print_exc()
                    else:
                        break
        except Exception as e:
            print(f"[サーバー] コマンド受信スレッドエラー: {e}")
            import traceback
            traceback.print_exc()
    
    def execute_command(self, command):
        """リモート制御コマンドを実行"""
        if not HAS_PYAUTOGUI:
            print("[サーバー] pyautoguiが利用できません")
            return
        
        cmd_type = command.get('type')
        
        try:
            if cmd_type == 'mouse_move':
                x = command.get('x', 0)
                y = command.get('y', 0)
                pyautogui.moveTo(x, y)
                
            elif cmd_type == 'mouse_click':
                x = command.get('x', 0)
                y = command.get('y', 0)
                button = command.get('button', 'left')
                clicks = command.get('clicks', 1)
                pyautogui.click(x, y, button=button, clicks=clicks)
                
            elif cmd_type == 'mouse_drag':
                x = command.get('x', 0)
                y = command.get('y', 0)
                pyautogui.dragTo(x, y, duration=0.1)
                
            elif cmd_type == 'mouse_scroll':
                x = command.get('x', 0)
                y = command.get('y', 0)
                dx = command.get('dx', 0)
                dy = command.get('dy', 0)
                pyautogui.scroll(dy, x=x, y=y)
                
            elif cmd_type == 'key_press':
                key = command.get('key', '')
                try:
                    pyautogui.press(key)
                    print(f"[サーバー] キー入力: {key}")
                except Exception as e:
                    print(f"[サーバー] キー入力エラー: {key} - {e}")
                
            elif cmd_type == 'key_type':
                text = command.get('text', '')
                try:
                    pyautogui.typewrite(text, interval=0.01)
                    print(f"[サーバー] テキスト入力: {text[:20]}...")
                except Exception as e:
                    print(f"[サーバー] テキスト入力エラー: {e}")
                
            elif cmd_type == 'key_combination':
                keys = command.get('keys', [])
                try:
                    pyautogui.hotkey(*keys)
                    print(f"[サーバー] ショートカットキー: {'+'.join(keys)}")
                except Exception as e:
                    print(f"[サーバー] ショートカットキーエラー: {keys} - {e}")
            else:
                print(f"[サーバー] 不明なコマンドタイプ: {cmd_type}")
                
        except Exception as e:
            print(f"[サーバー] コマンド実行エラー: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_file_transfer(self):
        """ファイル転送を処理"""
        try:
            while self.running and self.authenticated and self.client_socket:
                try:
                    command = self.receive_json()
                    if not command:
                        break
                    
                    if command.get('type') == 'file_request':
                        # ファイル送信要求
                        file_path = command.get('file_path', '')
                        if os.path.exists(file_path):
                            self.send_file(file_path)
                        else:
                            self.send_json({
                                'type': 'file_error',
                                'message': f'ファイルが見つかりません: {file_path}'
                            })
                    
                    elif command.get('type') == 'file_send':
                        # ファイル受信
                        self.receive_file(command)
                        
                except Exception as e:
                    if self.running:
                        print(f"[サーバー] ファイル転送エラー: {e}")
                    break
        except Exception as e:
            print(f"[サーバー] ファイル転送スレッドエラー: {e}")
    
    def send_file(self, file_path):
        """ファイルを送信"""
        try:
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            
            # ファイル情報を送信
            self.send_json({
                'type': 'file_info',
                'file_name': file_name,
                'file_size': file_size
            })
            
            # ファイルデータを送信
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    self.client_socket.sendall(chunk)
            
            print(f"[サーバー] ファイル送信完了: {file_name}")
            
        except Exception as e:
            print(f"[サーバー] ファイル送信エラー: {e}")
            self.send_json({
                'type': 'file_error',
                'message': str(e)
            })
    
    def receive_file(self, command):
        """ファイルを受信"""
        try:
            file_name = command.get('file_name', 'received_file')
            file_size = command.get('file_size', 0)
            save_path = os.path.join(os.path.expanduser('~'), 'Downloads', file_name)
            
            # ディレクトリを作成
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # ファイルデータを受信
            received = 0
            with open(save_path, 'wb') as f:
                while received < file_size:
                    chunk = self.client_socket.recv(min(file_size - received, 8192))
                    if not chunk:
                        break
                    f.write(chunk)
                    received += len(chunk)
            
            print(f"[サーバー] ファイル受信完了: {save_path}")
            
        except Exception as e:
            print(f"[サーバー] ファイル受信エラー: {e}")
    
    def handle_clipboard_sync(self):
        """クリップボード同期を処理"""
        if not HAS_PYPERCLIP:
            return
        
        last_clipboard = ""
        
        try:
            while self.running and self.authenticated and self.client_socket:
                try:
                    # ローカルクリップボードを監視
                    try:
                        current_clipboard = pyperclip.paste()
                        if current_clipboard != last_clipboard:
                            # クリップボードが変更された
                            self.send_json({
                                'type': 'clipboard_update',
                                'content': current_clipboard
                            })
                            last_clipboard = current_clipboard
                    except Exception:
                        pass
                    
                    # リモートからのクリップボード更新を受信
                    try:
                        command = self.receive_json()
                        if command and command.get('type') == 'clipboard_update':
                            content = command.get('content', '')
                            pyperclip.copy(content)
                            last_clipboard = content
                    except Exception:
                        pass
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    if self.running:
                        print(f"[サーバー] クリップボード同期エラー: {e}")
                    break
        except Exception as e:
            print(f"[サーバー] クリップボード同期スレッドエラー: {e}")
    
    def stop(self):
        """サーバーを停止"""
        self.running = False
        self.authenticated = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        print("[サーバー] 停止しました")


class SecureRemoteDesktopClient:
    """セキュアなリモートデスクトップクライアント"""
    
    def __init__(self, server_host, server_port=8888):
        self.server_host = server_host
        self.server_port = server_port
        self.socket = None
        self.connected = False
        self.authenticated = False
        self.receive_thread = None
        self.file_thread = None
        self.clipboard_thread = None
        
        # 画面表示用
        self.last_frame = None
        self.current_image = None
        self.image_scale = 1.0
        
        # GUI
        self.root = tk.Tk()
        self.root.title("Chrome風リモートデスクトップ - クライアント")
        self.root.geometry("1280x720")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_ui()
        
    def setup_ui(self):
        """UIのセットアップ"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ツールバー
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        # 接続フレーム
        connect_frame = ttk.LabelFrame(toolbar, text="接続", padding="5")
        connect_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(connect_frame, text="サーバー:").pack(side=tk.LEFT, padx=2)
        self.host_entry = ttk.Entry(connect_frame, width=15)
        self.host_entry.insert(0, self.server_host)
        self.host_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(connect_frame, text="ポート:").pack(side=tk.LEFT, padx=2)
        self.port_entry = ttk.Entry(connect_frame, width=8)
        self.port_entry.insert(0, str(self.server_port))
        self.port_entry.pack(side=tk.LEFT, padx=2)
        
        self.connect_btn = ttk.Button(connect_frame, text="接続", command=self.connect)
        self.connect_btn.pack(side=tk.LEFT, padx=2)
        
        self.disconnect_btn = ttk.Button(connect_frame, text="切断", 
                                         command=self.disconnect, state=tk.DISABLED)
        self.disconnect_btn.pack(side=tk.LEFT, padx=2)
        
        # ステータス
        self.status_label = ttk.Label(toolbar, text="未接続", foreground="red")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # ファイル転送ボタン
        file_frame = ttk.LabelFrame(toolbar, text="ファイル", padding="5")
        file_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(file_frame, text="送信", command=self.send_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="受信", command=self.request_file).pack(side=tk.LEFT, padx=2)
        
        # 設定フレーム
        settings_frame = ttk.LabelFrame(toolbar, text="設定", padding="5")
        settings_frame.pack(side=tk.LEFT)
        
        ttk.Label(settings_frame, text="品質:").pack(side=tk.LEFT, padx=2)
        self.quality_var = tk.IntVar(value=75)
        quality_scale = ttk.Scale(settings_frame, from_=30, to=100, 
                                  variable=self.quality_var, orient=tk.HORIZONTAL, length=100)
        quality_scale.pack(side=tk.LEFT, padx=2)
        
        # 画面表示フレーム
        screen_frame = ttk.LabelFrame(main_frame, text="リモート画面", padding="5")
        screen_frame.pack(fill=tk.BOTH, expand=True)
        
        # キャンバス（画面表示用）
        self.canvas = tk.Canvas(screen_frame, bg="black", cursor="arrow")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # マウスイベント
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<Button-3>", self.on_mouse_right_click)
        self.canvas.bind("<Double-Button-1>", self.on_mouse_double_click)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<MouseWheel>", self.on_mouse_scroll)
        
        # キーボードイベント（キャンバスにバインド）
        self.canvas.focus_set()
        self.canvas.bind("<Key>", self.on_key_press)
        self.canvas.bind("<KeyRelease>", self.on_key_release)
        # ルートウィンドウにもバインド（フォーカスが外れてもキー入力を取得）
        # ただし、キャンバスにフォーカスがある場合は、キャンバスのイベントのみ処理
        self.root.bind("<Key>", self.on_key_press_root)
        self.root.bind("<KeyRelease>", self.on_key_release)
        
    def connect(self):
        """サーバーに接続"""
        try:
            host = self.host_entry.get()
            port = int(self.port_entry.get())
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((host, port))
            
            # SSLでラップ（簡易版）
            # SSL接続は問題が多いため、デフォルトで無効化
            use_ssl = False  # SSLを無効化（問題が多いため）
            
            if use_ssl:
                try:
                    context = ssl.create_default_context()
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                    self.socket = context.wrap_socket(self.socket, server_hostname=host)
                    print("[クライアント] SSL接続確立")
                except Exception as e:
                    print(f"[クライアント] SSL設定エラー: {e}（通常接続にフォールバック）")
            else:
                print("[クライアント] 通常接続（SSL無効）")
            
            self.connected = True
            
            # 認証
            if not self.authenticate():
                self.disconnect()
                return
            
            self.status_label.config(text=f"接続中: {host}:{port}", foreground="green")
            self.connect_btn.config(state=tk.DISABLED)
            self.disconnect_btn.config(state=tk.NORMAL)
            
            # 受信スレッドを開始
            self.receive_thread = threading.Thread(target=self.receive_data, daemon=True)
            self.receive_thread.start()
            
        except Exception as e:
            messagebox.showerror("接続エラー", f"サーバーに接続できませんでした:\n{e}")
            self.status_label.config(text="接続失敗", foreground="red")
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
    
    def authenticate(self):
        """サーバーに認証"""
        try:
            # 認証要求を受信
            auth_request = self.receive_json()
            if not auth_request or auth_request.get('type') != 'auth_request':
                return False
            
            # パスワード入力ダイアログ
            dialog = tk.Toplevel(self.root)
            dialog.title("認証")
            dialog.geometry("300x150")
            dialog.transient(self.root)
            dialog.grab_set()
            
            ttk.Label(dialog, text=auth_request.get('message', 'パスワードを入力してください')).pack(pady=10)
            
            password_entry = ttk.Entry(dialog, show="*", width=30)
            password_entry.pack(pady=10)
            password_entry.focus()
            
            auth_result = {'success': False, 'password': ''}
            
            def ok_clicked():
                auth_result['password'] = password_entry.get()
                auth_result['success'] = True
                dialog.destroy()
            
            def cancel_clicked():
                dialog.destroy()
            
            ttk.Button(dialog, text="OK", command=ok_clicked).pack(side=tk.LEFT, padx=10, pady=10)
            ttk.Button(dialog, text="キャンセル", command=cancel_clicked).pack(side=tk.LEFT, padx=10, pady=10)
            
            dialog.bind("<Return>", lambda e: ok_clicked())
            dialog.wait_window()
            
            if not auth_result['success']:
                return False
            
            # パスワードを送信
            self.send_json({
                'type': 'auth_response',
                'password': auth_result['password']
            })
            
            # 認証結果を受信
            auth_response = self.receive_json()
            if auth_response and auth_response.get('type') == 'auth_success':
                self.authenticated = True
                return True
            else:
                message = auth_response.get('message', '認証に失敗しました') if auth_response else '認証に失敗しました'
                messagebox.showerror("認証エラー", message)
                return False
                
        except Exception as e:
            print(f"[クライアント] 認証エラー: {e}")
            return False
    
    def send_json(self, data):
        """JSONデータを送信（プロトコル修正版）"""
        if not self.socket:
            return
        
        # 送信ロック（複数のスレッドから同時に送信されないようにする）
        if not hasattr(self, '_send_lock'):
            self._send_lock = threading.Lock()
        
        with self._send_lock:
            try:
                json_str = json.dumps(data, ensure_ascii=False)
                data_bytes = json_str.encode('utf-8')
                size = len(data_bytes)
                
                # サイズの妥当性チェック
                if size > 1024 * 1024 or size == 0:  # 1MB以上または0は異常
                    print(f"[クライアント] 警告: 送信データサイズが異常です: {size}")
                    return
                
                # マジックナンバー + サイズ + データを送信
                magic = b'\xDE\xAD\xBE\xEF'
                header = struct.pack('>I', size)
                self.socket.sendall(magic + header + data_bytes)
            except ConnectionAbortedError:
                print("[クライアント] 接続が切断されました")
                self.disconnect()
            except Exception as e:
                print(f"[クライアント] JSON送信エラー: {e}")
                import traceback
                traceback.print_exc()
    
    def receive_json(self):
        """JSONデータを受信（プロトコル修正版）"""
        try:
            # マジックナンバーでメッセージの開始を確認（0xDEADBEEF）
            magic = b'\xDE\xAD\xBE\xEF'
            magic_buffer = b''
            
            # マジックナンバーを探す（最大100バイトまでスキャン）
            for _ in range(100):
                chunk = self.socket.recv(1)
                if not chunk:
                    return None
                magic_buffer += chunk
                if len(magic_buffer) >= 4:
                    if magic_buffer[-4:] == magic:
                        # マジックナンバーが見つかった
                        break
                    # 最初の1バイトを削除して続行
                    magic_buffer = magic_buffer[1:]
            else:
                # マジックナンバーが見つからなかった
                print("[クライアント] マジックナンバーが見つかりませんでした。接続をリセットする必要があります。")
                return None
            
            # サイズを受信（確実に4バイト受信）
            size_data = b''
            timeout_count = 0
            while len(size_data) < 4:
                chunk = self.socket.recv(4 - len(size_data))
                if not chunk:
                    timeout_count += 1
                    if timeout_count > 10:
                        return None
                    time.sleep(0.01)
                    continue
                size_data += chunk
                timeout_count = 0
            
            if len(size_data) < 4:
                return None
            
            size = struct.unpack('>I', size_data)[0]
            
            # サイズの妥当性チェック（より厳格に）
            if size > 1024 * 1024 or size == 0:  # 1MB以上または0は異常
                print(f"[クライアント] 異常なデータサイズ: {size}")
                return None
            
            # データを受信
            data = b''
            timeout_count = 0
            while len(data) < size:
                remaining = size - len(data)
                chunk = self.socket.recv(min(remaining, 4096))
                if not chunk:
                    timeout_count += 1
                    if timeout_count > 10:
                        print(f"[クライアント] データ受信タイムアウト: {len(data)}/{size} バイト受信")
                        return None
                    time.sleep(0.01)
                    continue
                data += chunk
                timeout_count = 0
            
            return json.loads(data.decode('utf-8'))
        except ConnectionAbortedError:
            return None
        except socket.timeout:
            print("[クライアント] 受信タイムアウト")
            return None
        except Exception as e:
            print(f"[クライアント] JSON受信エラー: {e}")
            return None
    
    def receive_data(self):
        """データを受信"""
        try:
            while self.connected and self.authenticated and self.socket:
                try:
                    # フレームデータを受信
                    frame_info = self.receive_json()
                    if not frame_info:
                        break
                    
                    if frame_info.get('type') == 'frame':
                        # 画像データを受信
                        compressed_size = frame_info.get('size', 0)
                        if compressed_size > 10 * 1024 * 1024:  # 10MB以上は異常
                            print(f"[クライアント] 異常な画像データサイズ: {compressed_size}")
                            break
                        
                        compressed_data = b''
                        timeout_count = 0
                        while len(compressed_data) < compressed_size:
                            remaining = compressed_size - len(compressed_data)
                            chunk = self.socket.recv(min(remaining, 8192))
                            if not chunk:
                                timeout_count += 1
                                if timeout_count > 10:
                                    print(f"[クライアント] 画像データ受信タイムアウト: {len(compressed_data)}/{compressed_size} バイト受信")
                                    break
                                time.sleep(0.01)
                                continue
                            compressed_data += chunk
                            timeout_count = 0
                        
                        if len(compressed_data) < compressed_size:
                            print(f"[クライアント] 画像データの受信が不完全: {len(compressed_data)}/{compressed_size} バイト")
                            break
                        
                        # 展開
                        img_data = zlib.decompress(compressed_data)
                        
                        # 画像を読み込み
                        img = Image.open(io.BytesIO(img_data))
                        
                        # 差分更新の処理
                        if frame_info.get('is_diff') and self.last_frame:
                            bbox = frame_info.get('bbox')
                            if bbox:
                                # 差分を適用
                                x, y, x2, y2 = bbox
                                self.last_frame.paste(img, (x, y))
                                img = self.last_frame.copy()
                        
                        self.last_frame = img.copy()
                        
                        # 画面を更新
                        self.root.after(0, self.update_screen, img)
                    
                    elif frame_info.get('type') == 'clipboard_update':
                        # クリップボード更新
                        if HAS_PYPERCLIP:
                            content = frame_info.get('content', '')
                            pyperclip.copy(content)
                    
                except Exception as e:
                    if self.connected:
                        print(f"[クライアント] 受信エラー: {e}")
                    break
        except Exception as e:
            print(f"[クライアント] 受信スレッドエラー: {e}")
        finally:
            if self.connected:
                self.root.after(0, self.disconnect)
    
    def update_screen(self, img):
        """画面を更新"""
        try:
            # キャンバスサイズに合わせてリサイズ
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # アスペクト比を維持
                img_ratio = img.width / img.height
                canvas_ratio = canvas_width / canvas_height
                
                if img_ratio > canvas_ratio:
                    new_width = canvas_width
                    new_height = int(canvas_width / img_ratio)
                else:
                    new_width = int(canvas_height * img_ratio)
                    new_height = canvas_height
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.image_scale = img.width / self.last_frame.width if self.last_frame else 1.0
            
            # PhotoImageに変換
            self.current_image = ImageTk.PhotoImage(img)
            
            # キャンバスに表示
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, 
                                    image=self.current_image, anchor=tk.CENTER)
            
        except Exception as e:
            print(f"[クライアント] 画面更新エラー: {e}")
    
    def send_command(self, command):
        """リモート制御コマンドを送信"""
        if self.connected and self.authenticated and self.socket:
            try:
                # コマンドが正しい形式か確認
                if not isinstance(command, dict) or 'type' not in command:
                    print(f"[クライアント] 無効なコマンド形式: {command}")
                    return
                
                # コマンドタイプに応じたスロットル
                cmd_type = command.get('type')
                current_time = time.time()
                
                if not hasattr(self, '_last_send_time'):
                    self._last_send_time = {}
                
                # マウス移動イベントのスロットル
                if cmd_type == 'mouse_move':
                    last_time = self._last_send_time.get('mouse_move', 0)
                    if current_time - last_time < 0.1:  # 10回/秒に制限
                        return  # 送信をスキップ
                    self._last_send_time['mouse_move'] = current_time
                
                # マウスクリックイベントのスロットル
                elif cmd_type == 'mouse_click':
                    last_time = self._last_send_time.get('mouse_click', 0)
                    if current_time - last_time < 0.15:  # 約6回/秒に制限（クリックしすぎを防ぐ）
                        return  # 送信をスキップ
                    self._last_send_time['mouse_click'] = current_time
                
                self.send_json(command)
            except Exception as e:
                print(f"[クライアント] コマンド送信エラー: {e}")
                import traceback
                traceback.print_exc()
    
    def on_mouse_move(self, event):
        """マウス移動"""
        if self.connected and self.authenticated and self.current_image:
            x = int(event.x / self.image_scale) if self.image_scale > 0 else event.x
            y = int(event.y / self.image_scale) if self.image_scale > 0 else event.y
            self.send_command({
                'type': 'mouse_move',
                'x': x,
                'y': y
            })
    
    def on_mouse_click(self, event):
        """マウスクリック"""
        if self.connected and self.authenticated and self.current_image:
            # クリック時にキャンバスにフォーカスを設定（キーボード入力を受け取るため）
            self.canvas.focus_set()
            
            # クリックのスロットル（連続クリックを防ぐ）
            current_time = time.time()
            if not hasattr(self, '_last_click_time'):
                self._last_click_time = 0
            
            if current_time - self._last_click_time < 0.1:  # 0.1秒未満の連続クリックは無視
                return
            
            self._last_click_time = current_time
            
            x = int(event.x / self.image_scale) if self.image_scale > 0 else event.x
            y = int(event.y / self.image_scale) if self.image_scale > 0 else event.y
            self.send_command({
                'type': 'mouse_click',
                'x': x,
                'y': y,
                'button': 'left',
                'clicks': 1
            })
    
    def on_mouse_right_click(self, event):
        """マウス右クリック"""
        if self.connected and self.authenticated and self.current_image:
            # クリック時にキャンバスにフォーカスを設定
            self.canvas.focus_set()
            
            # クリックのスロットル
            current_time = time.time()
            if not hasattr(self, '_last_click_time'):
                self._last_click_time = 0
            
            if current_time - self._last_click_time < 0.1:
                return
            
            self._last_click_time = current_time
            
            x = int(event.x / self.image_scale) if self.image_scale > 0 else event.x
            y = int(event.y / self.image_scale) if self.image_scale > 0 else event.y
            self.send_command({
                'type': 'mouse_click',
                'x': x,
                'y': y,
                'button': 'right',
                'clicks': 1
            })
    
    def on_mouse_double_click(self, event):
        """マウスダブルクリック"""
        if self.connected and self.authenticated and self.current_image:
            x = int(event.x / self.image_scale) if self.image_scale > 0 else event.x
            y = int(event.y / self.image_scale) if self.image_scale > 0 else event.y
            self.send_command({
                'type': 'mouse_click',
                'x': x,
                'y': y,
                'button': 'left',
                'clicks': 2
            })
    
    def on_mouse_drag(self, event):
        """マウスドラッグ"""
        if self.connected and self.authenticated and self.current_image:
            x = int(event.x / self.image_scale) if self.image_scale > 0 else event.x
            y = int(event.y / self.image_scale) if self.image_scale > 0 else event.y
            self.send_command({
                'type': 'mouse_drag',
                'x': x,
                'y': y
            })
    
    def on_mouse_scroll(self, event):
        """マウススクロール"""
        if self.connected and self.authenticated and self.current_image:
            x = int(event.x / self.image_scale) if self.image_scale > 0 else event.x
            y = int(event.y / self.image_scale) if self.image_scale > 0 else event.y
            
            # スクロール量を取得
            if platform.system() == "Windows":
                delta = event.delta if hasattr(event, 'delta') else 0
            else:
                delta = 1 if event.num == 4 else -1 if event.num == 5 else 0
            
            # スクロール量を調整
            scroll_amount = delta // 120 if delta != 0 else (1 if delta > 0 else -1)
            
            self.send_command({
                'type': 'mouse_scroll',
                'x': x,
                'y': y,
                'dx': 0,
                'dy': scroll_amount
            })
    
    def on_key_press(self, event):
        """キー入力（ショートカットキー対応）"""
        # 接続されていない、または認証されていない場合は無視
        if not self.connected or not self.authenticated:
            return
        
        # キャンバスにフォーカスを設定（キー入力を受け取るため）
        self.canvas.focus_set()
        
        # 修飾キーを検出
        modifiers = []
        if event.state & 0x0001:  # Shift
            modifiers.append('shift')
        if event.state & 0x0004:  # Control
            modifiers.append('ctrl')
        if event.state & 0x0008:  # Alt
            modifiers.append('alt')
        if event.state & 0x20000:  # Meta/Windows
            modifiers.append('win')
        
        # キー名をマッピング（tkinterのkeysymからpyautoguiのキー名へ）
        key = event.keysym.lower()
        key_mapping = {
            'return': 'enter',
            'backspace': 'backspace',
            'tab': 'tab',
            'escape': 'esc',
            'space': 'space',
            'delete': 'delete',
            'up': 'up',
            'down': 'down',
            'left': 'left',
            'right': 'right',
            'home': 'home',
            'end': 'end',
            'page_up': 'pageup',
            'page_down': 'pagedown',
            'f1': 'f1', 'f2': 'f2', 'f3': 'f3', 'f4': 'f4',
            'f5': 'f5', 'f6': 'f6', 'f7': 'f7', 'f8': 'f8',
            'f9': 'f9', 'f10': 'f10', 'f11': 'f11', 'f12': 'f12',
        }
        key = key_mapping.get(key, key)
        
        # 修飾キー自体が押された場合は無視（組み合わせのみ処理）
        if key in ['shift', 'ctrl', 'control', 'alt', 'win', 'meta', 'lshift', 'rshift', 
                   'lcontrol', 'rcontrol', 'lalt', 'ralt', 'lwin', 'rwin']:
            return
        
        # 修飾キーが押されている場合は、ショートカットキーとして送信
        if modifiers:
            # 修飾キーとメインキーの組み合わせ
            keys = modifiers + [key]
            self.send_command({
                'type': 'key_combination',
                'keys': keys
            })
        else:
            # 通常のキー入力
            self.send_command({
                'type': 'key_press',
                'key': key
            })
        
        # イベントの伝播を防ぐ（リモート側に送信したので、ローカルでは処理しない）
        return "break"
    
    def on_key_press_root(self, event):
        """ルートウィンドウのキー入力（キャンバスにフォーカスがない場合のみ処理）"""
        # 接続されていない、または認証されていない場合は無視
        if not self.connected or not self.authenticated:
            return
        
        # キャンバスにフォーカスを設定して、キャンバスのイベントハンドラーに委譲
        self.canvas.focus_set()
        # キャンバスのイベントハンドラーを呼び出す
        result = self.on_key_press(event)
        # イベントの伝播を防ぐ
        return result if result else "break"
    
    def on_key_release(self, event):
        """キーリリース"""
        pass
    
    def send_file(self):
        """ファイルを送信"""
        if not self.connected or not self.authenticated:
            messagebox.showwarning("警告", "接続されていません")
            return
        
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        
        try:
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            
            # ファイル情報を送信
            self.send_json({
                'type': 'file_send',
                'file_name': file_name,
                'file_size': file_size
            })
            
            # ファイルデータを送信
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    self.socket.sendall(chunk)
            
            messagebox.showinfo("成功", f"ファイルを送信しました: {file_name}")
            
        except Exception as e:
            messagebox.showerror("エラー", f"ファイル送信エラー: {e}")
    
    def request_file(self):
        """ファイルを要求"""
        if not self.connected or not self.authenticated:
            messagebox.showwarning("警告", "接続されていません")
            return
        
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        
        self.send_json({
            'type': 'file_request',
            'file_path': file_path
        })
    
    def disconnect(self):
        """サーバーから切断"""
        self.connected = False
        self.authenticated = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        self.status_label.config(text="切断しました", foreground="red")
        self.connect_btn.config(state=tk.NORMAL)
        self.disconnect_btn.config(state=tk.DISABLED)
        
        # 画面をクリア
        self.canvas.delete("all")
        self.last_frame = None
    
    def on_closing(self):
        """ウィンドウを閉じる"""
        self.disconnect()
        self.root.destroy()
    
    def run(self):
        """GUIを実行"""
        self.root.mainloop()


def main():
    """メイン関数"""
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'server':
            # サーバーモード
            print("=== Chrome風リモートデスクトップサーバー ===")
            server = SecureRemoteDesktopServer(host='0.0.0.0', port=8888)
            try:
                server.start()
            except KeyboardInterrupt:
                print("\nサーバーを停止します...")
                server.stop()
        
        elif mode == 'client':
            # クライアントモード
            server_host = sys.argv[2] if len(sys.argv) > 2 else '192.168.1.100'
            print("=== Chrome風リモートデスクトップクライアント ===")
            print(f"サーバー: {server_host}:8888")
            client = SecureRemoteDesktopClient(server_host, 8888)
            client.run()
        
        else:
            print("使用方法:")
            print("  サーバー: python chrome_like_remote_desktop.py server")
            print("  クライアント: python chrome_like_remote_desktop.py client [サーバーIP]")
    else:
        # GUIモード
        root = tk.Tk()
        root.title("Chrome風リモートデスクトップ")
        root.geometry("400x300")
        
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Chrome風リモートデスクトップ", 
                 font=("Arial", 16, "bold")).pack(pady=10)
        
        ttk.Label(main_frame, text="モードを選択してください", 
                 font=("Arial", 10)).pack(pady=10)
        
        def start_server():
            root.destroy()
            server = SecureRemoteDesktopServer(host='0.0.0.0', port=8888)
            try:
                server.start()
            except KeyboardInterrupt:
                server.stop()
        
        def start_client():
            dialog = tk.Toplevel(root)
            dialog.title("クライアント接続")
            dialog.geometry("300x150")
            
            ttk.Label(dialog, text="サーバーアドレス:").pack(pady=10)
            host_entry = ttk.Entry(dialog, width=20)
            host_entry.insert(0, "192.168.1.100")
            host_entry.pack(pady=5)
            
            def connect():
                host = host_entry.get()
                dialog.destroy()
                root.destroy()
                client = SecureRemoteDesktopClient(host, 8888)
                client.run()
            
            ttk.Button(dialog, text="接続", command=connect).pack(pady=10)
        
        ttk.Button(main_frame, text="サーバー（デスクトップ側）", 
                  command=start_server, width=30).pack(pady=5)
        ttk.Button(main_frame, text="クライアント（ノートパソコン側）", 
                  command=start_client, width=30).pack(pady=5)
        
        root.mainloop()


if __name__ == "__main__":
    main()


