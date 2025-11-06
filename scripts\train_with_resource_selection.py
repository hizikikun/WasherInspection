#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リソース選択機能付き学習スクリプト
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

# リソース選択をインポート
try:
    from training_resource_selector import TrainingResourceSelector
except ImportError:
    # パスを調整
    sys.path.insert(0, os.path.dirname(__file__))
    from training_resource_selector import TrainingResourceSelector

def main():
    """リソース選択付きで学習を実行"""
    print("=" * 80)
    print("リソース選択機能付き学習")
    print("=" * 80)
    
    # リソース選択
    selector = TrainingResourceSelector()
    config = selector.select_profile()
    selector.apply_config_to_environment()
    
    # 確認
    print("\n学習を開始しますか？")
    try:
        response = input("Enterで開始、Ctrl+Cでキャンセル: ").strip()
    except KeyboardInterrupt:
        print("\nキャンセルされました。")
        return
    
    # 学習スクリプトを実行
    print("\n学習を開始します...")
    print("=" * 80)
    
    # train_4class_sparse_ensemble.pyをインポートして実行
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

if __name__ == '__main__':
    main()



















