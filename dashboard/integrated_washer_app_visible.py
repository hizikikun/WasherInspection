#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合ワッシャー検査・学習アプリケーション（コンソール表示版）
- デバッグ用：コンソールウィンドウを表示するバージョン
"""

import sys
import os

# integrated_washer_app.pyをインポートするためにパスを追加
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# main関数をインポートして実行（ただしコンソール非表示コードを無効化）
if __name__ == '__main__':
    # コンソール非表示コードをスキップするために、モジュールを直接実行
    import integrated_washer_app
    
    # main関数を直接呼び出し（ただしコンソール非表示処理を無効化）
    try:
        # 既存のインスタンスを終了
        integrated_washer_app.kill_existing_instances()
        
        from PyQt5 import QtWidgets, QtCore
        
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # 警告音を無効化
        try:
            app.setAttribute(QtCore.Qt.AA_DisableWindowContextHelpButton)
        except:
            pass
        
        window = integrated_washer_app.IntegratedWasherApp()
        window.show()
        
        print("アプリケーションウィンドウを表示しました。")
        print("閉じるにはウィンドウを閉じるか、Ctrl+Cを押してください。")
        
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        error_msg = f"アプリケーション起動エラー:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        print("\n何かキーを押して終了してください...")
        input()
        sys.exit(1)












