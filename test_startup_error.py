#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""アプリ起動エラー診断スクリプト"""

import sys
import os
import traceback

# Windowsでのエンコーディング設定
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# パスを追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("アプリ起動エラー診断")
print("=" * 60)

try:
    # PyQt5のチェック
    print("\n[1] PyQt5のインポート...")
    from PyQt5 import QtWidgets, QtCore, QtGui
    print("  [OK] PyQt5 OK")
    
    # QApplicationの作成
    print("\n[2] QApplicationの作成...")
    app = QtWidgets.QApplication(sys.argv)
    print("  [OK] QApplication作成 OK")
    
    # アプリケーションのインポート
    print("\n[3] アプリケーションモジュールのインポート...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dashboard'))
    from integrated_washer_app import IntegratedWasherApp
    print("  [OK] アプリケーションモジュール OK")
    
    # ウィンドウの作成
    print("\n[4] ウィンドウの作成...")
    try:
        window = IntegratedWasherApp()
        print("  [OK] ウィンドウ作成 OK")
    except Exception as e:
        print(f"  [ERROR] ウィンドウ作成エラー: {e}")
        traceback.print_exc()
        input("\n何かキーを押して終了...")
        sys.exit(1)
    
    # ウィンドウの表示
    print("\n[5] ウィンドウの表示...")
    try:
        window.show()
        print("  [OK] ウィンドウ表示 OK")
    except Exception as e:
        print(f"  [ERROR] ウィンドウ表示エラー: {e}")
        traceback.print_exc()
        input("\n何かキーを押して終了...")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("アプリケーションを起動しました")
    print("ウィンドウを閉じるか、Ctrl+Cで終了してください")
    print("=" * 60)
    
    # イベントループの実行
    sys.exit(app.exec_())
    
except Exception as e:
    print(f"\n[ERROR] 重大なエラーが発生しました:")
    print(f"   {type(e).__name__}: {e}")
    print("\n詳細なエラー情報:")
    traceback.print_exc()
    input("\n何かキーを押して終了...")
    sys.exit(1)

