#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
アプリケーションショートカットを作成（タスクバーアイコン設定用）
"""

import os
import sys
from pathlib import Path

# UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    import win32com.client
    HAS_WIN32COM = True
except ImportError:
    HAS_WIN32COM = False
    print("警告: pywin32がインストールされていません。")
    print("インストール方法: pip install pywin32")

def create_shortcut():
    """アプリケーションショートカットを作成"""
    if not HAS_WIN32COM:
        print("エラー: pywin32が必要です。'pip install pywin32'を実行してください。")
        return False
    
    try:
        # パスを取得
        script_dir = Path(__file__).resolve().parents[2]
        batch_path = script_dir / 'start_washer_app.bat'
        icon_path = script_dir / 'assets' / 'logo_icon.ico'
        desktop_path = Path.home() / 'Desktop'
        
        # デスクトップにショートカットを作成
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut_path = desktop_path / 'WasherInspection.lnk'
        shortcut = shell.CreateShortCut(str(shortcut_path))
        
        # バッチファイルを起動
        shortcut.TargetPath = str(batch_path)
        shortcut.WorkingDirectory = str(script_dir)
        shortcut.IconLocation = str(icon_path)
        shortcut.Description = "統合ワッシャー検査・学習システム"
        
        shortcut.save()
        
        print(f"✓ ショートカット作成完了: {shortcut_path}")
        print(f"  アイコン: {icon_path}")
        print(f"  このショートカットをタスクバーにピン留めしてください。")
        return True
        
    except Exception as e:
        print(f"エラー: ショートカット作成に失敗しました: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン関数"""
    print("アプリケーションショートカットを作成中...")
    if create_shortcut():
        print("\n使用方法:")
        print("1. デスクトップに作成されたショートカットを確認")
        print("2. ショートカットを右クリック → 「タスクバーにピン留め」")
        print("3. または、ショートカットをタスクバーにドラッグ&ドロップ")
    else:
        print("\n手動でショートカットを作成する場合:")
        print("1. デスクトップを右クリック → 新規作成 → ショートカット")
        print(f"2. 項目の場所: {sys.executable} \"{Path(__file__).resolve().parents[2] / 'dashboard' / 'integrated_washer_app.py'}\"")
        print("3. ショートカットを右クリック → プロパティ → アイコンの変更")
        print(f"4. アイコンを選択: {Path(__file__).resolve().parents[2] / 'assets' / 'logo_icon.ico'}")

if __name__ == '__main__':
    main()

