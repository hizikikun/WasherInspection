#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Training Runner
学習性能と精度を向上させるための統合実行スクリプト
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_script(script_path, description):
    """スクリプトを実行"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    
    if not Path(script_path).exists():
        print(f"エラー: {script_path} が見つかりません")
        return False
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              text=True, 
                              encoding='utf-8')
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"\n✓ {description} が正常に完了しました")
            print(f"所要時間: {(end_time - start_time)/60:.1f}分")
            return True
        else:
            print(f"\n✗ {description} が失敗しました (終了コード: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n✗ {description} の実行中にエラーが発生しました: {e}")
        return False

def check_dependencies():
    """依存関係をチェック"""
    print("=" * 80)
    print("依存関係チェック")
    print("=" * 80)
    
    required_packages = [
        'tensorflow',
        'albumentations', 
        'optuna',
        'opencv-python',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'numpy',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (未インストール)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠ {len(missing_packages)} 個のパッケージが不足していまム")
        print("以下のコマンドでインストールしてください:")
        print(f"python scripts/install_advanced_dependencies.py")
        return False
    else:
        print("\n✓ すべての依存関係が満たされていまム")
        return True

def main():
    """メイン実行関数"""
    print("=" * 80)
    print("Enhanced Training Runner")
    print("学習性能と精度を向上させるための統合実行システム")
    print("=" * 80)
    
    # 依存関係チェック
    if not check_dependencies():
        print("\n依存関係をインストールしてから再実行してください。")
        return
    
    # 実行するスクリプトのリスト
    training_scripts = [
        {
            'script': 'scripts/ultra_high_performance_trainer.py',
            'description': 'Ultra-High Performance 学習システム',
            'required': True
        },
        {
            'script': 'scripts/advanced_optimization_trainer.py', 
            'description': 'Advanced Optimization 学習システム',
            'required': False
        }
    ]
    
    print(f"\n実行予定の学習システム: {len(training_scripts)}個")
    for i, script_info in enumerate(training_scripts, 1):
        status = "必須" if script_info['required'] else "オプション"
        print(f"  {i}. {script_info['description']} ({status})")
    
    # ユーザー確認
    print(f"\n学習を開始しますか？ [y/N] ", end='', flush=True)
    try:
        answer = input().strip().lower()
    except KeyboardInterrupt:
        print("\n\n学習をキャンセルしました。")
        return
    
    if answer not in ('y', 'yes', '1'):
        print("学習をキャンセルしました。")
        return
    
    # 学習実行
    total_start_time = time.time()
    success_count = 0
    total_count = len(training_scripts)
    
    for i, script_info in enumerate(training_scripts, 1):
        print(f"\n[{i}/{total_count}] {script_info['description']} を実行中...")
        
        if run_script(script_info['script'], script_info['description']):
            success_count += 1
        else:
            if script_info['required']:
                print(f"\n⚠ 必須の学習システムが失敗しました。")
                print("学習を中断します。")
                break
            else:
                print(f"\n⚠ オプションの学習システムが失敗しました。")
                print("次の学習システムに進みまム。")
    
    # 結果表示
    total_time = time.time() - total_start_time
    print(f"\n{'='*80}")
    print("学習実行結果")
    print(f"{'='*80}")
    print(f"成功: {success_count}/{total_count} システム")
    print(f"失敗: {total_count - success_count}/{total_count} システム")
    print(f"合計所要時間: {total_time/60:.1f}分")
    
    if success_count > 0:
        print(f"\n✓ {success_count} 個の学習システムが正常に完了しました!")
        print("改善されたモデルが保存されました。")
        
        # 保存されたモデルファイルの確認
        model_files = [
            'ultra_high_performance_ensemble_model_*.h5',
            'advanced_optimization_final_model.h5',
            'ultra_high_performance_ensemble_info.json',
            'advanced_optimization_info.json'
        ]
        
        print(f"\n保存されたファイル:")
        for pattern in model_files:
            if '*' in pattern:
                import glob
                files = glob.glob(pattern)
                for file in files:
                    if Path(file).exists():
                        print(f"  ✓ {file}")
            else:
                if Path(pattern).exists():
                    print(f"  ✓ {pattern}")
        
        print(f"\n次のステップ:")
        print("1. 改善されたモデルで検査システムをテスト")
        print("2. 必要に応じて追加の学習データを収集")
        print("3. ハイパーパラメータをさらに調整")
    else:
        print(f"\n✗ すべての学習システムが失敗しました。")
        print("エラーログを確認して問題を解決してください。")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
