#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リソース選択 + 進捗ビューアー（HWiNFO統合）付き学習スクリプト
"""

import os
import sys

# UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# train_4class_sparse_ensemble.pyをインポートして実行
if __name__ == '__main__':
    # 学習スクリプトを直接実行
    import importlib.util
    train_script_path = os.path.join(os.path.dirname(__file__), 'train_4class_sparse_ensemble.py')
    
    if os.path.exists(train_script_path):
        spec = importlib.util.spec_from_file_location("train_module", train_script_path)
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        
        # main()を実行
        if hasattr(train_module, 'main'):
            train_module.main()
        else:
            print("エラー: train_4class_sparse_ensemble.pyにmain()関数が見つかりません")
    else:
        print(f"エラー: {train_script_path} が見つかりません")



















