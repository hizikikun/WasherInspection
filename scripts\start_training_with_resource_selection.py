#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リソース選択機能付き学習起動スクリプト
ユーザーがGPU/CPU負荷レベルを選択してから学習を開始
"""

import os
import sys
import subprocess

# UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def main():
    print("=" * 80)
    print("リソース選択機能付き学習")
    print("=" * 80)
    print()
    
    # リソース選択をインポート
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from training_resource_selector import TrainingResourceSelector
    except ImportError:
        print("エラー: training_resource_selector.pyが見つかりません")
        return
    
    # リソース選択
    selector = TrainingResourceSelector()
    config = selector.select_profile()
    selector.apply_config_to_environment()
    
    # 確認
    print("\n" + "=" * 80)
    print("学習を開始します")
    print("=" * 80)
    print()
    
    # 学習スクリプトを実行
    train_script = os.path.join(os.path.dirname(__file__), 'train_4class_sparse_ensemble.py')
    
    if not os.path.exists(train_script):
        print(f"エラー: {train_script} が見つかりません")
        return
    
    # Pythonで実行
    print("学習スクリプトを起動しています...")
    print()
    
    try:
        # subprocessで実行（環境変数を引き継ぐ）
        result = subprocess.run(
            [sys.executable, train_script],
            env=os.environ.copy(),
            cwd=os.path.dirname(train_script) if os.path.dirname(train_script) else None
        )
        
        if result.returncode == 0:
            print("\n" + "=" * 80)
            print("学習が正常に完了しました")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print(f"学習がエラーで終了しました (終了コード: {result.returncode})")
            print("=" * 80)
    except KeyboardInterrupt:
        print("\n学習が中断されました")
    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()



















