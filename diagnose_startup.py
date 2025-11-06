#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""アプリ起動診断スクリプト - 詳細なエラー情報を取得"""

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

print("=" * 60)
print("アプリ起動診断")
print("=" * 60)

# ステップ1: 基本インポート
print("\n[1] 基本モジュールのインポート...")
try:
    import json
    import time
    import cv2
    import numpy as np
    from pathlib import Path
    print("  [OK] 基本モジュール OK")
except Exception as e:
    print(f"  [ERROR] 基本モジュールエラー: {e}")
    traceback.print_exc()
    sys.exit(1)

# ステップ2: PyQt5のインポート
print("\n[2] PyQt5のインポート...")
try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    print("  [OK] PyQt5 OK")
except Exception as e:
    print(f"  [ERROR] PyQt5エラー: {e}")
    traceback.print_exc()
    sys.exit(1)

# ステップ3: アプリケーションファイルのインポート
print("\n[3] アプリケーションファイルのインポート...")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dashboard'))
    from integrated_washer_app import IntegratedWasherApp
    print("  [OK] アプリケーションファイル OK")
except Exception as e:
    print(f"  [ERROR] インポートエラー: {e}")
    traceback.print_exc()
    print("\n詳細なエラー情報:")
    print(traceback.format_exc())
    sys.exit(1)

# ステップ4: QApplicationの作成
print("\n[4] QApplicationの作成...")
try:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    print("  [OK] QApplication作成 OK")
except Exception as e:
    print(f"  [ERROR] QApplication作成エラー: {e}")
    traceback.print_exc()
    sys.exit(1)

# ステップ5: ウィンドウの作成
print("\n[5] ウィンドウの作成...")
try:
    window = IntegratedWasherApp()
    print("  [OK] ウィンドウ作成 OK")
except Exception as e:
    print(f"  [ERROR] ウィンドウ作成エラー: {e}")
    traceback.print_exc()
    print("\n詳細なエラー情報:")
    print(traceback.format_exc())
    
    # エラーログに記録
    try:
        error_log_path = Path('app_error_log.txt')
        with open(error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"診断スクリプト実行時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n")
            f.write(f"ウィンドウ作成エラー: {str(e)}\n")
            f.write(f"スタックトレース:\n{traceback.format_exc()}\n")
            f.write(f"{'='*60}\n\n")
    except:
        pass
    
    input("\n何かキーを押して終了...")
    sys.exit(1)

# ステップ6: ウィンドウの表示
print("\n[6] ウィンドウの表示...")
try:
    window.show()
    print("  [OK] ウィンドウ表示 OK")
except Exception as e:
    print(f"  [ERROR] ウィンドウ表示エラー: {e}")
    traceback.print_exc()
    input("\n何かキーを押して終了...")
    sys.exit(1)

print("\n" + "=" * 60)
print("診断完了: アプリケーションを起動しました")
print("=" * 60)
print("\nウィンドウが表示されているはずです。")
print("閉じるにはウィンドウを閉じるか、Ctrl+Cを押してください。\n")

try:
    sys.exit(app.exec_())
except KeyboardInterrupt:
    print("\n[情報] ユーザーによって中断されました。")
    sys.exit(0)
except Exception as e:
    print(f"\n[ERROR] 実行中にエラーが発生しました: {e}")
    traceback.print_exc()
    input("\n何かキーを押して終了...")
    sys.exit(1)






