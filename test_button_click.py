#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習開始ボタンのテストスクリプト
"""

import sys
import os
from pathlib import Path

# ダッシュボードディレクトリをパスに追加
dashboard_dir = Path(__file__).parent / 'dashboard'
sys.path.insert(0, str(dashboard_dir))

print("[TEST] テスト開始")
print(f"[TEST] 作業ディレクトリ: {os.getcwd()}")
print(f"[TEST] Pythonパス: {sys.path[:3]}")

try:
    from PyQt5 import QtWidgets, QtCore
    print("[TEST] PyQt5インポート成功")
except Exception as e:
    print(f"[TEST ERROR] PyQt5インポート失敗: {e}")
    sys.exit(1)

try:
    import integrated_washer_app
    print("[TEST] integrated_washer_appインポート成功")
except Exception as e:
    print(f"[TEST ERROR] integrated_washer_appインポート失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def test_button_click():
    """ボタンクリックをテスト"""
    print("[TEST] アプリケーションを作成中...")
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    
    print("[TEST] IntegratedWasherAppインスタンスを作成中...")
    try:
        window = integrated_washer_app.IntegratedWasherApp()
        print("[TEST] IntegratedWasherAppインスタンス作成成功")
        
        # 学習開始ボタンが存在するか確認
        if hasattr(window, 'start_training_btn'):
            print(f"[TEST] start_training_btn存在確認: {window.start_training_btn}")
            print(f"[TEST] start_training_btn有効状態: {window.start_training_btn.isEnabled()}")
            print(f"[TEST] start_training_btn可視状態: {window.start_training_btn.isVisible()}")
            
            # ボタンクリックをシミュレート
            print("[TEST] ボタンクリックをシミュレート中...")
            try:
                window.start_training_btn.click()
                print("[TEST] ボタンクリックシミュレート成功")
            except Exception as click_error:
                print(f"[TEST ERROR] ボタンクリックシミュレート失敗: {click_error}")
                import traceback
                traceback.print_exc()
        else:
            print("[TEST ERROR] start_training_btnが見つかりません")
        
        window.show()
        print("[TEST] ウィンドウを表示しました")
        print("[TEST] 5秒後に自動終了します...")
        
        # 5秒後に自動終了
        QtCore.QTimer.singleShot(5000, app.quit)
        
        sys.exit(app.exec_())
    except Exception as e:
        print(f"[TEST ERROR] アプリケーション作成エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    test_button_click()

