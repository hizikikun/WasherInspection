#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chrome風リモートデスクトップのセットアップ確認スクリプト
必要なライブラリがインストールされているか確認します
"""

import sys

def check_python_version():
    """Pythonのバージョンを確認"""
    print("=" * 60)
    print("Pythonバージョンの確認")
    print("=" * 60)
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 7:
        print("✅ Pythonのバージョンは問題ありません")
        return True
    else:
        print("❌ Python 3.7以上が必要です")
        return False

def check_library(library_name, import_name=None):
    """ライブラリがインストールされているか確認"""
    if import_name is None:
        import_name = library_name
    
    try:
        __import__(import_name)
        print(f"✅ {library_name} がインストールされています")
        return True
    except ImportError:
        print(f"❌ {library_name} がインストールされていません")
        print(f"   インストール: pip install {library_name}")
        return False

def check_all_libraries():
    """すべてのライブラリを確認"""
    print("\n" + "=" * 60)
    print("必要なライブラリの確認")
    print("=" * 60)
    
    libraries = [
        ("mss", "mss"),
        ("pillow", "PIL"),
        ("pyautogui", "pyautogui"),
        ("pyperclip", "pyperclip"),
        ("cryptography", "cryptography"),
    ]
    
    results = []
    for lib_name, import_name in libraries:
        results.append(check_library(lib_name, import_name))
    
    return all(results)

def check_tkinter():
    """tkinterが利用可能か確認"""
    print("\n" + "=" * 60)
    print("GUIライブラリ（tkinter）の確認")
    print("=" * 60)
    
    try:
        import tkinter
        print("✅ tkinterが利用可能です")
        return True
    except ImportError:
        print("❌ tkinterが利用できません")
        print("   Pythonを再インストールしてください")
        return False

def check_file():
    """メインファイルが存在するか確認"""
    print("\n" + "=" * 60)
    print("ファイルの確認")
    print("=" * 60)
    
    import os
    main_file = "chrome_like_remote_desktop.py"
    
    if os.path.exists(main_file):
        print(f"✅ {main_file} が存在します")
        return True
    else:
        print(f"❌ {main_file} が見つかりません")
        print(f"   現在のディレクトリ: {os.getcwd()}")
        return False

def main():
    """メイン関数"""
    print("\n" + "=" * 60)
    print("Chrome風リモートデスクトップ - セットアップ確認")
    print("=" * 60)
    print()
    
    results = []
    
    # Pythonバージョン確認
    results.append(check_python_version())
    
    # ライブラリ確認
    results.append(check_all_libraries())
    
    # tkinter確認
    results.append(check_tkinter())
    
    # ファイル確認
    results.append(check_file())
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("確認結果")
    print("=" * 60)
    
    if all(results):
        print("✅ すべてのチェックが完了しました！")
        print("   準備完了です。以下のコマンドで起動できます:")
        print()
        print("   サーバー: python chrome_like_remote_desktop.py server")
        print("   クライアント: python chrome_like_remote_desktop.py client [IPアドレス]")
    else:
        print("❌ いくつかの問題が見つかりました")
        print("   上記のメッセージに従って、不足しているものをインストールしてください")
        print()
        print("   すべてのライブラリをインストール:")
        print("   python -m pip install mss pillow pyautogui pyperclip cryptography")
    
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()




