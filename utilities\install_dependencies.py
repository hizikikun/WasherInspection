#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Dependencies Installer
高度な学習システムに必要な依存関係をインストール
"""

import subprocess
import sys
import os

def install_package(package):
    """パッケージをインストール"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} のインストールが完了しました")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {package} のインストールに失敗しました: {e}")
        return False

def main():
    """メイン実行関数"""
    print("=" * 80)
    print("Advanced Dependencies Installer")
    print("高度な学習システムに必要な依存関係をインストールします")
    print("=" * 80)
    
    # 必要なパッケージリスト
    packages = [
        "albumentations>=1.3.0",
        "optuna>=3.0.0",
        "tensorflow>=2.10.0",
        "tensorflow-addons>=0.20.0",
        "opencv-python>=4.8.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pillow>=10.0.0",
        "psutil>=5.9.0",
        "pynvml>=11.5.0"
    ]
    
    print(f"\nインストール対象パッケージ: {len(packages)}個")
    for i, package in enumerate(packages, 1):
        print(f"  {i}. {package}")
    
    print("\nインストールを開始します...")
    
    success_count = 0
    total_count = len(packages)
    
    for i, package in enumerate(packages, 1):
        print(f"\n[{i}/{total_count}] {package} をインストール中...")
        if install_package(package):
            success_count += 1
    
    print("\n" + "=" * 80)
    print("インストール結果")
    print("=" * 80)
    print(f"成功: {success_count}/{total_count} パッケージ")
    print(f"失敗: {total_count - success_count}/{total_count} パッケージ")
    
    if success_count == total_count:
        print("\n✓ すべてのパッケージのインストールが完了しました!")
        print("高度な学習システムを使用できまム。")
    else:
        print(f"\n⚠ {total_count - success_count} 個のパッケージのインストールに失敗しました。")
        print("失敗したパッケージを手動でインストールしてください。")
    
    print("\n次のステップ:")
    print("1. ultra_high_performance_trainer.py を実行")
    print("2. advanced_optimization_trainer.py を実行")
    print("=" * 80)

if __name__ == "__main__":
    main()
