#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習開始機能のテストスクリプト
アプリを起動して学習開始ボタンを押した時のエラーをキャッチ
"""

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

print("="*60)
print("学習開始機能テスト")
print("="*60)

try:
    # PyQt5のインポート
    print("\n[1] PyQt5のインポート...")
    from PyQt5 import QtWidgets, QtCore, QtGui
    print("  [OK] PyQt5 OK")
    
    # アプリケーションのインポート
    print("\n[2] アプリケーションモジュールのインポート...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dashboard'))
    from integrated_washer_app import IntegratedWasherApp, TrainingWorker
    print("  [OK] アプリケーションモジュール OK")
    
    # QApplicationの作成
    print("\n[3] QApplicationの作成...")
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    print("  [OK] QApplication作成 OK")
    
    # ウィンドウの作成
    print("\n[4] ウィンドウの作成...")
    window = IntegratedWasherApp()
    print("  [OK] ウィンドウ作成 OK")
    
    # ウィンドウの表示
    print("\n[5] ウィンドウの表示...")
    window.show()
    print("  [OK] ウィンドウ表示 OK")
    
    # 学習開始ボタンの存在確認
    print("\n[6] 学習開始ボタンの確認...")
    if hasattr(window, 'start_training_btn'):
        print("  [OK] start_training_btn が存在します")
        print(f"  [OK] ボタンテキスト: {window.start_training_btn.text()}")
    else:
        print("  [ERROR] start_training_btn が見つかりません")
        print("  利用可能な属性:")
        attrs = [attr for attr in dir(window) if 'training' in attr.lower() or 'start' in attr.lower()]
        for attr in attrs[:20]:
            print(f"    - {attr}")
    
    # 学習開始メソッドのテスト
    print("\n[7] 学習開始メソッドのテスト...")
    print("  注意: 実際に学習を開始するわけではありません。")
    print("  エラーが発生するかどうかを確認します。")
    
    # リソース設定を初期化
    window.resource_config = {
        'batch_size': 32,
        'workers': 8,
        'max_epochs': 200,
        'patience': 30,
        'use_mixed_precision': True,
        'gpu_utilization': 'high',
        'cpu_utilization': 'high',
    }
    print("  [OK] リソース設定を初期化しました")
    
    # 学習開始メソッドを直接呼び出さず、まずは準備だけ確認
    print("\n[8] 学習開始前の準備確認...")
    
    # 必要な属性が存在するか確認
    required_attrs = [
        'resource_config',
        'start_training_btn',
        'stop_training_btn',
    ]
    
    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(window, attr):
            missing_attrs.append(attr)
        else:
            print(f"  [OK] {attr} が存在します")
    
    if missing_attrs:
        print(f"  [WARN] 以下の属性が見つかりません: {', '.join(missing_attrs)}")
    
    print("\n" + "="*60)
    print("準備完了")
    print("="*60)
    print("\nアプリケーションウィンドウが表示されています。")
    print("「学習開始」ボタンを押して、エラーが発生するか確認してください。")
    print("エラーが発生した場合は、エラーメッセージを記録します。")
    print("\n閉じるにはウィンドウを閉じるか、Ctrl+Cを押してください。\n")
    
    # エラーハンドリング付きでイベントループを実行
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("\n[情報] ユーザーによって中断されました。")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] 実行中にエラーが発生しました: {e}")
        traceback.print_exc()
        
        # エラーログに記録
        try:
            error_log_path = os.path.join(os.path.dirname(__file__), 'app_error_log.txt')
            with open(error_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"学習開始テストエラー: {os.path.basename(__file__)}\n")
                f.write(f"エラー発生時刻: {os.popen('date /t && time /t').read().strip()}\n")
                f.write(f"{'='*60}\n")
                f.write(f"エラー: {str(e)}\n")
                f.write(f"スタックトレース:\n{traceback.format_exc()}\n")
                f.write(f"{'='*60}\n\n")
        except:
            pass
        
        input("\n何かキーを押して終了...")
        sys.exit(1)
        
except Exception as e:
    print(f"\n[ERROR] 重大なエラーが発生しました:")
    print(f"   {type(e).__name__}: {e}")
    print("\n詳細なエラー情報:")
    traceback.print_exc()
    
    # エラーログに記録
    try:
        error_log_path = os.path.join(os.path.dirname(__file__), 'app_error_log.txt')
        with open(error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"学習開始テスト - 重大なエラー\n")
            f.write(f"{'='*60}\n")
            f.write(f"エラー: {str(e)}\n")
            f.write(f"スタックトレース:\n{traceback.format_exc()}\n")
            f.write(f"{'='*60}\n\n")
    except:
        pass
    
    input("\n何かキーを押して終了...")
    sys.exit(1)
