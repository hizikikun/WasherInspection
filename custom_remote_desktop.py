#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
カスタムリモートデスクトップシステム
- デスクトップ側: サーバー（画面共有 + リモート制御受信）
- ノートパソコン側: クライアント（画面表示 + リモート制御送信）
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import socket
import threading
import struct
import json
import time
import io
from PIL import Image, ImageTk
import platform
import subprocess
import os

# 画面キャプチャ用（Windows）
if platform.system() == "Windows":
    try:
        import mss
        HAS_MSS = True
    except ImportError:
        HAS_MSS = False
        print("警告: mssがインストールされていません。画面キャプチャ機能が制限されます。")
        print("インストール: pip install mss pillow")
else:
    HAS_MSS = False

# リモート制御用（Windows）
if platform.system() == "Windows":
    try:
        import pyautogui
        HAS_PYAUTOGUI = True
    except ImportError:
        HAS_PYAUTOGUI = False
        print("警告: pyautoguiがインストールされていません。リモート制御機能が制限されます。")
        print("インストール: pip install pyautogui")
else:
    HAS_PYAUTOGUI = False


class RemoteDesktopServer:
    """リモートデスクトップサーバー（デスクトップ側）"""
    
    def __init__(self, host='0.0.0.0', port=8888):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.capture_thread = None
        
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
            self.client_socket, addr = self.server_socket.accept()
            print(f"[サーバー] クライアント接続: {addr}")
            
            # 画面キャプチャスレッドを開始
            self.capture_thread = threading.Thread(target=self.capture_and_send, daemon=True)
            self.capture_thread.start()
            
            # リモート制御コマンドを受信
            self.receive_commands()
            
        except Exception as e:
            print(f"[サーバー] エラー: {e}")
            self.stop()
    
    def capture_and_send(self):
        """画面をキャプチャして送信"""
        if not HAS_MSS:
            print("[サーバー] 画面キャプチャ機能が利用できません")
            return
        
        import mss
        
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # プライマリモニター
                
                while self.running and self.client_socket:
                    try:
                        # 画面キャプチャ
                        screenshot = sct.grab(monitor)
                        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                        
                        # 画像を圧縮
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format='JPEG', quality=70)
                        img_data = img_buffer.getvalue()
                        
                        # データサイズを送信
                        size = len(img_data)
                        self.client_socket.sendall(struct.pack('>I', size))
                        
                        # 画像データを送信
                        self.client_socket.sendall(img_data)
                        
                        time.sleep(0.1)  # 10 FPS
                        
                    except Exception as e:
                        if self.running:
                            print(f"[サーバー] キャプチャエラー: {e}")
                        break
        except Exception as e:
            print(f"[サーバー] 画面キャプチャ初期化エラー: {e}")
    
    def receive_commands(self):
        """リモート制御コマンドを受信"""
        try:
            while self.running and self.client_socket:
                try:
                    # コマンドを受信
                    data = self.client_socket.recv(1024)
                    if not data:
                        break
                    
                    command = json.loads(data.decode('utf-8'))
                    self.execute_command(command)
                    
                except Exception as e:
                    if self.running:
                        print(f"[サーバー] コマンド受信エラー: {e}")
                    break
        except Exception as e:
            print(f"[サーバー] コマンド受信スレッドエラー: {e}")
    
    def execute_command(self, command):
        """リモート制御コマンドを実行"""
        cmd_type = command.get('type')
        
        if not HAS_PYAUTOGUI:
            print(f"[サーバー] リモート制御機能が利用できません")
            return
        
        try:
            if cmd_type == 'mouse_move':
                x = command.get('x', 0)
                y = command.get('y', 0)
                pyautogui.moveTo(x, y)
                
            elif cmd_type == 'mouse_click':
                x = command.get('x', 0)
                y = command.get('y', 0)
                button = command.get('button', 'left')
                pyautogui.click(x, y, button=button)
                
            elif cmd_type == 'key_press':
                key = command.get('key', '')
                pyautogui.press(key)
                
            elif cmd_type == 'key_type':
                text = command.get('text', '')
                pyautogui.typewrite(text)
                
        except Exception as e:
            print(f"[サーバー] コマンド実行エラー: {e}")
    
    def stop(self):
        """サーバーを停止"""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        print("[サーバー] 停止しました")


class RemoteDesktopClient:
    """リモートデスクトップクライアント（ノートパソコン側）"""
    
    def __init__(self, server_host, server_port=8888):
        self.server_host = server_host
        self.server_port = server_port
        self.socket = None
        self.connected = False
        self.receive_thread = None
        
        # GUI
        self.root = tk.Tk()
        self.root.title("リモートデスクトップ - クライアント")
        self.root.geometry("1024x768")
        
        self.setup_ui()
        
    def setup_ui(self):
        """UIのセットアップ"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 接続フレーム
        connect_frame = ttk.LabelFrame(main_frame, text="接続設定", padding="10")
        connect_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(connect_frame, text="サーバーアドレス:").grid(row=0, column=0, padx=5)
        self.host_entry = ttk.Entry(connect_frame, width=20)
        self.host_entry.insert(0, self.server_host)
        self.host_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(connect_frame, text="ポート:").grid(row=0, column=2, padx=5)
        self.port_entry = ttk.Entry(connect_frame, width=10)
        self.port_entry.insert(0, str(self.server_port))
        self.port_entry.grid(row=0, column=3, padx=5)
        
        self.connect_btn = ttk.Button(connect_frame, text="接続", command=self.connect)
        self.connect_btn.grid(row=0, column=4, padx=5)
        
        self.disconnect_btn = ttk.Button(connect_frame, text="切断", command=self.disconnect, state=tk.DISABLED)
        self.disconnect_btn.grid(row=0, column=5, padx=5)
        
        # ステータス
        self.status_label = ttk.Label(connect_frame, text="未接続", foreground="red")
        self.status_label.grid(row=1, column=0, columnspan=6, pady=(5, 0))
        
        # 画面表示フレーム
        screen_frame = ttk.LabelFrame(main_frame, text="リモート画面", padding="10")
        screen_frame.pack(fill=tk.BOTH, expand=True)
        
        # キャンバス（画面表示用）
        self.canvas = tk.Canvas(screen_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # マウスイベント
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<Button-3>", self.on_mouse_right_click)
        self.canvas.bind("<Double-Button-1>", self.on_mouse_double_click)
        
        # キーボードイベント
        self.root.bind("<Key>", self.on_key_press)
        
        # 現在の画像
        self.current_image = None
        self.image_scale = 1.0
        
    def connect(self):
        """サーバーに接続"""
        try:
            host = self.host_entry.get()
            port = int(self.port_entry.get())
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((host, port))
            self.connected = True
            
            self.status_label.config(text=f"接続中: {host}:{port}", foreground="green")
            self.connect_btn.config(state=tk.DISABLED)
            self.disconnect_btn.config(state=tk.NORMAL)
            
            # 受信スレッドを開始
            self.receive_thread = threading.Thread(target=self.receive_screen, daemon=True)
            self.receive_thread.start()
            
        except Exception as e:
            messagebox.showerror("接続エラー", f"サーバーに接続できませんでした:\n{e}")
            self.status_label.config(text="接続失敗", foreground="red")
    
    def disconnect(self):
        """サーバーから切断"""
        self.connected = False
        if self.socket:
            self.socket.close()
            self.socket = None
        
        self.status_label.config(text="切断しました", foreground="red")
        self.connect_btn.config(state=tk.NORMAL)
        self.disconnect_btn.config(state=tk.DISABLED)
        
        # 画面をクリア
        self.canvas.delete("all")
    
    def receive_screen(self):
        """画面データを受信"""
        try:
            while self.connected and self.socket:
                try:
                    # データサイズを受信
                    size_data = self.socket.recv(4)
                    if len(size_data) < 4:
                        break
                    
                    size = struct.unpack('>I', size_data)[0]
                    
                    # 画像データを受信
                    img_data = b''
                    while len(img_data) < size:
                        chunk = self.socket.recv(min(size - len(img_data), 4096))
                        if not chunk:
                            break
                        img_data += chunk
                    
                    if len(img_data) == size:
                        # 画像を表示
                        self.root.after(0, self.update_screen, img_data)
                    
                except Exception as e:
                    if self.connected:
                        print(f"[クライアント] 受信エラー: {e}")
                    break
        except Exception as e:
            print(f"[クライアント] 受信スレッドエラー: {e}")
        finally:
            if self.connected:
                self.root.after(0, self.disconnect)
    
    def update_screen(self, img_data):
        """画面を更新"""
        try:
            # 画像を読み込み
            img = Image.open(io.BytesIO(img_data))
            
            # キャンバスサイズに合わせてリサイズ
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
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
        if self.connected and self.socket:
            try:
                data = json.dumps(command).encode('utf-8')
                self.socket.sendall(data)
            except Exception as e:
                print(f"[クライアント] コマンド送信エラー: {e}")
    
    def on_mouse_move(self, event):
        """マウス移動"""
        if self.connected and self.current_image:
            # 画面座標を計算（スケールを考慮）
            x = int(event.x * self.image_scale)
            y = int(event.y * self.image_scale)
            
            self.send_command({
                'type': 'mouse_move',
                'x': x,
                'y': y
            })
    
    def on_mouse_click(self, event):
        """マウスクリック"""
        if self.connected and self.current_image:
            x = int(event.x * self.image_scale)
            y = int(event.y * self.image_scale)
            
            self.send_command({
                'type': 'mouse_click',
                'x': x,
                'y': y,
                'button': 'left'
            })
    
    def on_mouse_right_click(self, event):
        """マウス右クリック"""
        if self.connected and self.current_image:
            x = int(event.x * self.image_scale)
            y = int(event.y * self.image_scale)
            
            self.send_command({
                'type': 'mouse_click',
                'x': x,
                'y': y,
                'button': 'right'
            })
    
    def on_mouse_double_click(self, event):
        """マウスダブルクリック"""
        if self.connected and self.current_image:
            x = int(event.x * self.image_scale)
            y = int(event.y * self.image_scale)
            
            self.send_command({
                'type': 'mouse_click',
                'x': x,
                'y': y,
                'button': 'left',
                'double': True
            })
    
    def on_key_press(self, event):
        """キー入力"""
        if self.connected:
            key = event.keysym
            self.send_command({
                'type': 'key_press',
                'key': key
            })
    
    def run(self):
        """GUIを実行"""
        self.root.mainloop()


class RemoteDesktopAutoLauncher:
    """既存ソフトウェアの自動起動ラッパー"""
    
    @staticmethod
    def launch_teamviewer(teamviewer_id, password=None):
        """TeamViewerを自動起動して接続"""
        try:
            # TeamViewerのパスを検索
            teamviewer_paths = [
                r"C:\Program Files\TeamViewer\TeamViewer.exe",
                r"C:\Program Files (x86)\TeamViewer\TeamViewer.exe",
                "teamviewer"
            ]
            
            teamviewer_exe = None
            for path in teamviewer_paths:
                if os.path.exists(path) or path == "teamviewer":
                    teamviewer_exe = path
                    break
            
            if not teamviewer_exe:
                return False, "TeamViewerが見つかりません"
            
            # TeamViewerを起動
            if password:
                subprocess.Popen([teamviewer_exe, "-i", teamviewer_id, "-p", password])
            else:
                subprocess.Popen([teamviewer_exe, "-i", teamviewer_id])
            
            return True, "TeamViewerを起動しました"
        except Exception as e:
            return False, f"TeamViewer起動エラー: {e}"
    
    @staticmethod
    def launch_anydesk(anydesk_id, password=None):
        """AnyDeskを自動起動して接続"""
        try:
            # AnyDeskのパスを検索
            anydesk_paths = [
                r"C:\Program Files\AnyDesk\AnyDesk.exe",
                r"C:\Program Files (x86)\AnyDesk\AnyDesk.exe",
                "anydesk"
            ]
            
            anydesk_exe = None
            for path in anydesk_paths:
                if os.path.exists(path) or path == "anydesk":
                    anydesk_exe = path
                    break
            
            if not anydesk_exe:
                return False, "AnyDeskが見つかりません"
            
            # AnyDeskを起動
            subprocess.Popen([anydesk_exe, anydesk_id])
            
            return True, "AnyDeskを起動しました"
        except Exception as e:
            return False, f"AnyDesk起動エラー: {e}"


def main():
    """メイン関数"""
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'server':
            # サーバーモード（デスクトップ側）
            print("=== リモートデスクトップサーバー ===")
            print("デスクトップ側で実行してください")
            print("クライアントの接続を待機しています...")
            
            server = RemoteDesktopServer(host='0.0.0.0', port=8888)
            try:
                server.start()
            except KeyboardInterrupt:
                print("\nサーバーを停止します...")
                server.stop()
        
        elif mode == 'client':
            # クライアントモード（ノートパソコン側）
            server_host = sys.argv[2] if len(sys.argv) > 2 else '192.168.1.100'
            
            print("=== リモートデスクトップクライアント ===")
            print(f"サーバー: {server_host}:8888")
            
            client = RemoteDesktopClient(server_host, 8888)
            client.run()
        
        else:
            print("使用方法:")
            print("  サーバー: python custom_remote_desktop.py server")
            print("  クライアント: python custom_remote_desktop.py client [サーバーIP]")
    else:
        # GUIモード
        root = tk.Tk()
        root.title("カスタムリモートデスクトップ")
        root.geometry("400x300")
        
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="カスタムリモートデスクトップ", 
                 font=("Arial", 16, "bold")).pack(pady=10)
        
        ttk.Label(main_frame, text="モードを選択してください", 
                 font=("Arial", 10)).pack(pady=10)
        
        def start_server():
            root.destroy()
            server = RemoteDesktopServer(host='0.0.0.0', port=8888)
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
                client = RemoteDesktopClient(host, 8888)
                client.run()
            
            ttk.Button(dialog, text="接続", command=connect).pack(pady=10)
        
        ttk.Button(main_frame, text="サーバー（デスクトップ側）", 
                  command=start_server, width=30).pack(pady=5)
        ttk.Button(main_frame, text="クライアント（ノートパソコン側）", 
                  command=start_client, width=30).pack(pady=5)
        
        root.mainloop()


if __name__ == "__main__":
    main()






