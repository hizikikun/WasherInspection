#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合ワッシャー検査・学習アプリケーション（コンソール非表示版）
- このファイルはWindowsでコンソールウィンドウを表示せずに実行するためのものです
- .pyw拡張子により、Pythonウィンドウなしで実行されます
"""

import os
import sys

# integrated_washer_app.pyと同じディレクトリにパスを追加
if __name__ == '__main__':
    # このスクリプトと同じディレクトリにあるintegrated_washer_app.pyをインポート
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # integrated_washer_app.pyのmain関数をインポートして実行
    from integrated_washer_app import main
    main()













