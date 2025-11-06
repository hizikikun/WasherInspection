#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""アプリ起動テストスクリプト"""

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
print("アプリ起動テスト")
print("=" * 60)

# PyQt5のチェック
print("\n[1] PyQt5のインポートチェック...")
try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    print("  [OK] PyQt5 OK")
except Exception as e:
    print(f"  [ERROR] PyQt5エラー: {e}")
    traceback.print_exc()
    sys.exit(1)

# アプリケーションのインポート
print("\n[2] アプリケーションファイルのインポート...")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dashboard'))
    from integrated_washer_app import IntegratedWasherApp
    print("  [OK] アプリケーションファイル OK")
except Exception as e:
    print(f"  [ERROR] インポートエラー: {e}")
    traceback.print_exc()
    sys.exit(1)

# アプリケーションの起動
print("\n[3] アプリケーションの起動...")
try:
    app = QtWidgets.QApplication(sys.argv)
    print("  [OK] QApplication作成 OK")
    
    window = IntegratedWasherApp()
    print("  [OK] ウィンドウ作成 OK")
    
    window.show()
    print("  [OK] ウィンドウ表示 OK")
    print("\nアプリケーションを起動しました。ウィンドウが表示されているはずです。")
    print("閉じるにはウィンドウを閉じるか、Ctrl+Cを押してください。\n")
    
    sys.exit(app.exec_())
except Exception as e:
    print(f"  [ERROR] 起動エラー: {e}")
    traceback.print_exc()
    input("\n何かキーを押して終了...")
    sys.exit(1)


