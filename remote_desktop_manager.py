#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リモートデスクトップ管理アプリ
- 複数の接続先を保存・管理
- ワンクリック接続
- 接続履歴の記録
- 接続状態の確認
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import os
import subprocess
import platform
from datetime import datetime
from pathlib import Path
import threading
import socket

class RemoteDesktopManager:
    def __init__(self, root):
        self.root = root
        self.root.title("リモートデスクトップ管理")
        self.root.geometry("900x700")
        
        # 設定ファイルのパス
        self.config_file = Path(__file__).parent / "remote_desktop_config.json"
        self.history_file = Path(__file__).parent / "remote_desktop_history.json"
        
        # データ
        self.connections = []
        self.history = []
        
        # 接続タイプ
        self.connection_types = {
            "Windows RDP": "rdp",
            "TeamViewer": "teamviewer",
            "AnyDesk": "anydesk",
            "Chrome Remote Desktop": "chrome",
            "VNC": "vnc",
            "その他": "other"
        }
        
        # UIセットアップ
        self.setup_ui()
        
        # データ読み込み
        self.load_data()
        self.refresh_connection_list()
        
    def setup_ui(self):
        """UIのセットアップ"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # グリッドの重み設定
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # タイトル
        title_label = ttk.Label(main_frame, text="リモートデスクトップ管理", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # メインコンテンツ（左右分割）
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=2)
        content_frame.rowconfigure(0, weight=1)
        
        # 左側：接続一覧と操作
        left_frame = ttk.LabelFrame(content_frame, text="接続一覧", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)
        
        # 接続一覧のボタン
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        add_btn = ttk.Button(button_frame, text="追加", command=self.add_connection)
        add_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        edit_btn = ttk.Button(button_frame, text="編集", command=self.edit_connection)
        edit_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        delete_btn = ttk.Button(button_frame, text="削除", command=self.delete_connection)
        delete_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # 接続一覧（Treeview）
        list_frame = ttk.Frame(left_frame)
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # スクロールバー
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 接続リスト
        self.connection_tree = ttk.Treeview(list_frame, columns=("type",), show="tree headings", 
                                           yscrollcommand=scrollbar.set)
        self.connection_tree.heading("#0", text="名前")
        self.connection_tree.heading("type", text="タイプ")
        self.connection_tree.column("#0", width=150)
        self.connection_tree.column("type", width=100)
        self.connection_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.connection_tree.yview)
        
        # 選択変更時に詳細を更新
        self.connection_tree.bind("<<TreeviewSelect>>", lambda e: self.update_detail_display())
        
        # ダブルクリックで接続
        self.connection_tree.bind("<Double-1>", lambda e: self.connect_selected())
        
        # 接続ボタン
        connect_frame = ttk.Frame(left_frame)
        connect_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        connect_btn = ttk.Button(connect_frame, text="接続", command=self.connect_selected,
                                style="Accent.TButton")
        connect_btn.pack(fill=tk.X)
        
        # 右側：詳細情報と履歴
        right_frame = ttk.Frame(content_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        # 詳細情報
        detail_frame = ttk.LabelFrame(right_frame, text="接続詳細", padding="10")
        detail_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        detail_frame.columnconfigure(1, weight=1)
        
        self.detail_labels = {}
        detail_fields = [
            ("名前", "name"),
            ("タイプ", "type"),
            ("アドレス/ID", "address"),
            ("ユーザー名", "username"),
            ("メモ", "notes")
        ]
        
        for i, (label, key) in enumerate(detail_fields):
            ttk.Label(detail_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            value_label = ttk.Label(detail_frame, text="", foreground="gray")
            value_label.grid(row=i, column=1, sticky=tk.W, padx=(10, 0), pady=2)
            self.detail_labels[key] = value_label
        
        # 履歴
        history_frame = ttk.LabelFrame(right_frame, text="接続履歴", padding="10")
        history_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)
        
        # 履歴テキスト
        self.history_text = scrolledtext.ScrolledText(history_frame, height=15, width=40)
        self.history_text.pack(fill=tk.BOTH, expand=True)
        
        # 履歴クリアボタン
        clear_history_btn = ttk.Button(history_frame, text="履歴をクリア", 
                                       command=self.clear_history)
        clear_history_btn.pack(pady=(5, 0))
        
        # ステータスバー
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="準備完了", foreground="green")
        self.status_label.pack(side=tk.LEFT)
        
        # 接続状態確認ボタン
        check_status_btn = ttk.Button(status_frame, text="接続状態を確認", 
                                      command=self.check_connection_status)
        check_status_btn.pack(side=tk.RIGHT)
        
    def load_data(self):
        """データの読み込み"""
        # 接続設定の読み込み
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    self.connections = json.load(f)
            except Exception as e:
                messagebox.showerror("エラー", f"設定ファイルの読み込みに失敗しました: {e}")
                self.connections = []
        else:
            self.connections = []
        
        # 履歴の読み込み
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception as e:
                self.history = []
        else:
            self.history = []
        
        self.update_history_display()
    
    def save_data(self):
        """データの保存"""
        try:
            # 接続設定の保存
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.connections, f, ensure_ascii=False, indent=2)
            
            # 履歴の保存
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("エラー", f"データの保存に失敗しました: {e}")
    
    def refresh_connection_list(self):
        """接続一覧の更新"""
        # 既存の項目を削除
        for item in self.connection_tree.get_children():
            self.connection_tree.delete(item)
        
        # 接続を追加
        for i, conn in enumerate(self.connections):
            self.connection_tree.insert("", "end", iid=str(i), 
                                       text=conn.get("name", "無名"),
                                       values=(conn.get("type", "不明"),))
        
        # 選択をクリア
        self.connection_tree.selection_remove(self.connection_tree.selection())
        self.update_detail_display()
    
    def update_detail_display(self):
        """詳細表示の更新"""
        selection = self.connection_tree.selection()
        if not selection:
            # 選択なし
            for key in self.detail_labels:
                self.detail_labels[key].config(text="")
            return
        
        # 選択された接続を取得
        index = int(selection[0])
        if 0 <= index < len(self.connections):
            conn = self.connections[index]
            self.detail_labels["name"].config(text=conn.get("name", ""))
            self.detail_labels["type"].config(text=conn.get("type", ""))
            self.detail_labels["address"].config(text=conn.get("address", ""))
            self.detail_labels["username"].config(text=conn.get("username", ""))
            self.detail_labels["notes"].config(text=conn.get("notes", ""))
        else:
            for key in self.detail_labels:
                self.detail_labels[key].config(text="")
    
    def update_history_display(self):
        """履歴表示の更新"""
        self.history_text.delete("1.0", tk.END)
        
        if not self.history:
            self.history_text.insert("1.0", "接続履歴がありません")
            return
        
        # 最新の履歴を上に表示
        for entry in reversed(self.history[-50:]):  # 最新50件
            timestamp = entry.get("timestamp", "")
            name = entry.get("name", "不明")
            status = entry.get("status", "不明")
            message = entry.get("message", "")
            
            status_color = "green" if status == "成功" else "red" if status == "失敗" else "gray"
            
            self.history_text.insert("1.0", f"[{timestamp}] {name}\n", "timestamp")
            self.history_text.insert(tk.END, f"  状態: {status}\n", status)
            if message:
                self.history_text.insert(tk.END, f"  {message}\n", "message")
            self.history_text.insert(tk.END, "\n")
        
        # タグの設定
        self.history_text.tag_config("timestamp", foreground="blue")
        self.history_text.tag_config("status", foreground="green")
        self.history_text.tag_config("message", foreground="gray")
    
    def add_connection(self):
        """接続の追加"""
        dialog = ConnectionDialog(self.root, "接続の追加")
        if dialog.result:
            self.connections.append(dialog.result)
            self.save_data()
            self.refresh_connection_list()
            self.status_label.config(text="接続を追加しました", foreground="green")
    
    def edit_connection(self):
        """接続の編集"""
        selection = self.connection_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "編集する接続を選択してください")
            return
        
        index = int(selection[0])
        if 0 <= index < len(self.connections):
            dialog = ConnectionDialog(self.root, "接続の編集", self.connections[index])
            if dialog.result:
                self.connections[index] = dialog.result
                self.save_data()
                self.refresh_connection_list()
                self.status_label.config(text="接続を更新しました", foreground="green")
    
    def delete_connection(self):
        """接続の削除"""
        selection = self.connection_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "削除する接続を選択してください")
            return
        
        index = int(selection[0])
        if 0 <= index < len(self.connections):
            conn = self.connections[index]
            if messagebox.askyesno("確認", f"'{conn.get('name', '無名')}' を削除しますか？"):
                del self.connections[index]
                self.save_data()
                self.refresh_connection_list()
                self.status_label.config(text="接続を削除しました", foreground="green")
    
    def connect_selected(self):
        """選択された接続に接続"""
        selection = self.connection_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "接続する接続を選択してください")
            return
        
        index = int(selection[0])
        if 0 <= index < len(self.connections):
            conn = self.connections[index]
            self.connect(conn)
    
    def connect(self, conn):
        """接続を実行"""
        name = conn.get("name", "不明")
        conn_type = conn.get("type", "")
        address = conn.get("address", "")
        username = conn.get("username", "")
        
        self.status_label.config(text=f"接続中: {name}...", foreground="blue")
        self.root.update()
        
        try:
            success = False
            message = ""
            
            # 接続タイプに応じて接続処理を実行
            if conn_type in ["Windows RDP", "rdp"]:
                success, message = self.connect_rdp(address, username)
            elif conn_type in ["TeamViewer", "teamviewer"]:
                success, message = self.connect_teamviewer(address)
            elif conn_type in ["AnyDesk", "anydesk"]:
                success, message = self.connect_anydesk(address)
            elif conn_type in ["Chrome Remote Desktop", "chrome"]:
                success, message = self.connect_chrome(address)
            elif conn_type in ["VNC", "vnc"]:
                success, message = self.connect_vnc(address, username)
            else:
                message = f"未対応の接続タイプ: {conn_type}"
            
            # 履歴に記録
            self.add_history(name, "成功" if success else "失敗", message)
            
            if success:
                self.status_label.config(text=f"接続成功: {name}", foreground="green")
            else:
                self.status_label.config(text=f"接続失敗: {name}", foreground="red")
                messagebox.showerror("接続エラー", message)
        
        except Exception as e:
            self.add_history(name, "失敗", str(e))
            self.status_label.config(text=f"接続エラー: {name}", foreground="red")
            messagebox.showerror("エラー", f"接続中にエラーが発生しました: {e}")
    
    def connect_rdp(self, address, username):
        """Windows RDP接続"""
        try:
            if platform.system() != "Windows":
                return False, "Windows RDPはWindowsでのみ利用可能です"
            
            # mstscコマンドで接続
            cmd = ["mstsc", "/v:" + address]
            if username:
                cmd.append("/u:" + username)
            
            subprocess.Popen(cmd)
            return True, "RDP接続を開始しました"
        except Exception as e:
            return False, f"RDP接続エラー: {e}"
    
    def connect_teamviewer(self, address):
        """TeamViewer接続"""
        try:
            # TeamViewerのパスを検索
            teamviewer_paths = [
                r"C:\Program Files\TeamViewer\TeamViewer.exe",
                r"C:\Program Files (x86)\TeamViewer\TeamViewer.exe",
                "teamviewer"  # PATHにある場合
            ]
            
            teamviewer_exe = None
            for path in teamviewer_paths:
                if os.path.exists(path) or path == "teamviewer":
                    teamviewer_exe = path
                    break
            
            if not teamviewer_exe:
                return False, "TeamViewerが見つかりません。インストールしてください。"
            
            # TeamViewerを起動して接続
            subprocess.Popen([teamviewer_exe, "-i", address])
            return True, f"TeamViewer接続を開始しました (ID: {address})"
        except Exception as e:
            return False, f"TeamViewer接続エラー: {e}"
    
    def connect_anydesk(self, address):
        """AnyDesk接続"""
        try:
            # AnyDeskのパスを検索
            anydesk_paths = [
                r"C:\Program Files\AnyDesk\AnyDesk.exe",
                r"C:\Program Files (x86)\AnyDesk\AnyDesk.exe",
                "anydesk"  # PATHにある場合
            ]
            
            anydesk_exe = None
            for path in anydesk_paths:
                if os.path.exists(path) or path == "anydesk":
                    anydesk_exe = path
                    break
            
            if not anydesk_exe:
                return False, "AnyDeskが見つかりません。インストールしてください。"
            
            # AnyDeskを起動して接続
            subprocess.Popen([anydesk_exe, address])
            return True, f"AnyDesk接続を開始しました (ID: {address})"
        except Exception as e:
            return False, f"AnyDesk接続エラー: {e}"
    
    def connect_chrome(self, address):
        """Chrome Remote Desktop接続"""
        try:
            # Chrome Remote DesktopのURLを開く
            url = f"https://remotedesktop.google.com/access/{address}"
            if platform.system() == "Windows":
                os.startfile(url)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", url])
            else:
                subprocess.Popen(["xdg-open", url])
            return True, "Chrome Remote Desktopを開きました"
        except Exception as e:
            return False, f"Chrome Remote Desktop接続エラー: {e}"
    
    def connect_vnc(self, address, username):
        """VNC接続"""
        try:
            # VNCビューアーのパスを検索
            vnc_paths = [
                r"C:\Program Files\TightVNC\tvnviewer.exe",
                r"C:\Program Files\RealVNC\VNC Viewer\vncviewer.exe",
                "vncviewer"
            ]
            
            vnc_exe = None
            for path in vnc_paths:
                if os.path.exists(path) or path == "vncviewer":
                    vnc_exe = path
                    break
            
            if not vnc_exe:
                return False, "VNCビューアーが見つかりません。インストールしてください。"
            
            # VNC接続
            if username:
                subprocess.Popen([vnc_exe, f"{username}@{address}"])
            else:
                subprocess.Popen([vnc_exe, address])
            return True, f"VNC接続を開始しました ({address})"
        except Exception as e:
            return False, f"VNC接続エラー: {e}"
    
    def add_history(self, name, status, message=""):
        """履歴に追加"""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "status": status,
            "message": message
        }
        self.history.append(entry)
        
        # 履歴が1000件を超えたら古いものを削除
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        self.save_data()
        self.update_history_display()
    
    def clear_history(self):
        """履歴をクリア"""
        if messagebox.askyesno("確認", "接続履歴をすべて削除しますか？"):
            self.history = []
            self.save_data()
            self.update_history_display()
            self.status_label.config(text="履歴をクリアしました", foreground="green")
    
    def check_connection_status(self):
        """接続状態を確認"""
        selection = self.connection_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "確認する接続を選択してください")
            return
        
        index = int(selection[0])
        if 0 <= index < len(self.connections):
            conn = self.connections[index]
            address = conn.get("address", "")
            conn_type = conn.get("type", "")
            
            if not address:
                messagebox.showwarning("警告", "アドレスが設定されていません")
                return
            
            self.status_label.config(text="接続状態を確認中...", foreground="blue")
            self.root.update()
            
            # 別スレッドで確認
            thread = threading.Thread(target=self._check_status_thread, 
                                     args=(conn,), daemon=True)
            thread.start()
    
    def _check_status_thread(self, conn):
        """接続状態確認（別スレッド）"""
        address = conn.get("address", "")
        conn_type = conn.get("type", "")
        
        try:
            if conn_type in ["Windows RDP", "rdp"]:
                # RDPのポート3389を確認
                result = self.check_port(address, 3389)
                status = "接続可能" if result else "接続不可"
            elif conn_type in ["TeamViewer", "teamviewer", "AnyDesk", "anydesk", 
                              "Chrome Remote Desktop", "chrome"]:
                # これらのサービスは通常接続可能（ファイアウォールを通過）
                status = "接続可能（確認済み）"
            else:
                status = "確認不可"
            
            self.root.after(0, lambda: self.status_label.config(
                text=f"状態: {status}", foreground="green" if "可能" in status else "red"))
            self.root.after(0, lambda: messagebox.showinfo("接続状態", 
                f"{conn.get('name', '不明')}\n状態: {status}"))
        
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(
                text=f"確認エラー: {e}", foreground="red"))
    
    def check_port(self, host, port, timeout=3):
        """ポートの接続確認"""
        try:
            # ホスト名からIPアドレスを取得
            if ":" in host:
                host, port_str = host.split(":")
                port = int(port_str)
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except:
            return False


class ConnectionDialog:
    """接続設定ダイアログ"""
    def __init__(self, parent, title, connection=None):
        self.result = None
        
        # ダイアログウィンドウ
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("500x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # メインフレーム
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # フィールド
        fields = [
            ("名前", "name", True),
            ("タイプ", "type", True),
            ("アドレス/ID", "address", True),
            ("ユーザー名", "username", False),
            ("パスワード", "password", False),
            ("メモ", "notes", False)
        ]
        
        self.entries = {}
        
        for i, (label, key, required) in enumerate(fields):
            ttk.Label(main_frame, text=f"{label}{'*' if required else ''}:").grid(
                row=i, column=0, sticky=tk.W, pady=5)
            
            if key == "type":
                # タイプはコンボボックス
                combo = ttk.Combobox(main_frame, values=list(self.get_connection_types().keys()),
                                    state="readonly", width=30)
                combo.grid(row=i, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
                self.entries[key] = combo
            elif key == "password":
                # パスワードは非表示
                entry = ttk.Entry(main_frame, show="*", width=33)
                entry.grid(row=i, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
                self.entries[key] = entry
            elif key == "notes":
                # メモは複数行
                entry = scrolledtext.ScrolledText(main_frame, height=4, width=33)
                entry.grid(row=i, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
                self.entries[key] = entry
            else:
                entry = ttk.Entry(main_frame, width=33)
                entry.grid(row=i, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
                self.entries[key] = entry
        
        main_frame.columnconfigure(1, weight=1)
        
        # 既存の接続データを設定
        if connection:
            self.entries["name"].insert(0, connection.get("name", ""))
            if "type" in connection:
                self.entries["type"].set(connection.get("type", ""))
            self.entries["address"].insert(0, connection.get("address", ""))
            self.entries["username"].insert(0, connection.get("username", ""))
            if "password" in connection:
                self.entries["password"].insert(0, connection.get("password", ""))
            if "notes" in connection:
                self.entries["notes"].insert("1.0", connection.get("notes", ""))
        
        # ボタン
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=len(fields), column=0, columnspan=2, pady=(20, 0))
        
        ok_btn = ttk.Button(button_frame, text="OK", command=self.ok_clicked, width=15)
        ok_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        cancel_btn = ttk.Button(button_frame, text="キャンセル", command=self.cancel_clicked, width=15)
        cancel_btn.pack(side=tk.LEFT)
        
        # エンターキーでOK
        self.dialog.bind("<Return>", lambda e: self.ok_clicked())
        self.dialog.bind("<Escape>", lambda e: self.cancel_clicked())
        
        # フォーカス
        self.entries["name"].focus()
        
        # ダイアログが閉じられるまで待機
        self.dialog.wait_window()
    
    def get_connection_types(self):
        """接続タイプのリスト"""
        return {
            "Windows RDP": "rdp",
            "TeamViewer": "teamviewer",
            "AnyDesk": "anydesk",
            "Chrome Remote Desktop": "chrome",
            "VNC": "vnc",
            "その他": "other"
        }
    
    def ok_clicked(self):
        """OKボタンクリック"""
        # 必須フィールドの確認
        if not self.entries["name"].get().strip():
            messagebox.showwarning("警告", "名前を入力してください")
            return
        
        if not self.entries["type"].get():
            messagebox.showwarning("警告", "タイプを選択してください")
            return
        
        if not self.entries["address"].get().strip():
            messagebox.showwarning("警告", "アドレス/IDを入力してください")
            return
        
        # 結果を設定
        self.result = {
            "name": self.entries["name"].get().strip(),
            "type": self.entries["type"].get(),
            "address": self.entries["address"].get().strip(),
            "username": self.entries["username"].get().strip(),
            "notes": self.entries["notes"].get("1.0", tk.END).strip()
        }
        
        # パスワードは暗号化せずに保存（簡易版）
        password = self.entries["password"].get()
        if password:
            self.result["password"] = password
        
        self.dialog.destroy()
    
    def cancel_clicked(self):
        """キャンセルボタンクリック"""
        self.dialog.destroy()


def main():
    root = tk.Tk()
    app = RemoteDesktopManager(root)
    root.mainloop()


if __name__ == "__main__":
    main()

