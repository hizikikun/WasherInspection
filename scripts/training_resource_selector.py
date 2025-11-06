#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習時のGPU/CPU負荷レベルを選択するシステム
"""

import os
import sys

# UTF-8 encoding for Windows (環境変数を使用、より安全)
if sys.platform.startswith('win'):
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    os.environ.setdefault('PYTHONUTF8', '1')

class TrainingResourceSelector:
    """学習時のリソース負荷レベルを選択"""
    
    # GPU/CPU個別選択用の設定
    GPU_LEVELS = {
        'low': {
            'name': '軽量',
            'batch_size': 8,
            'use_mixed_precision': True,
        },
        'medium': {
            'name': '標準',
            'batch_size': 16,
            'use_mixed_precision': True,
        },
        'high': {
            'name': '高性能',
            'batch_size': 32,
            'use_mixed_precision': True,
        },
        'maximum': {
            'name': '最大性能',
            'batch_size': 64,
            'use_mixed_precision': False,  # フル精度
        },
    }
    
    CPU_LEVELS = {
        'low': {
            'name': '軽量',
            'workers': 1,
            'max_queue_size': 5,
            'use_multiprocessing': False,
        },
        'medium': {
            'name': '標準',
            'workers': 4,
            'max_queue_size': 10,
            'use_multiprocessing': True,
        },
        'high': {
            'name': '高性能',
            'workers': 8,
            'max_queue_size': 20,
            'use_multiprocessing': True,
        },
        'maximum': {
            'name': '最大性能',
            'workers': 16,
            'max_queue_size': 50,
            'use_multiprocessing': True,
        },
    }
    
    RESOURCE_PROFILES = {
        '1': {
            'name': '軽量（省エネモード）',
            'description': 'CPU/GPUを軽く使用。他の作業も可能。',
            'batch_size': 8,
            'workers': 1,
            'max_queue_size': 5,
            'use_multiprocessing': False,
            'max_epochs': 100,
            'patience': 20,
            'use_mixed_precision': True,
            'gpu_utilization': 'low',
            'cpu_utilization': 'low',
        },
        '2': {
            'name': '標準（バランス）',
            'description': 'CPU/GPUを適度に使用。バランスが良い。',
            'batch_size': 16,
            'workers': 4,
            'max_queue_size': 10,
            'use_multiprocessing': True,
            'max_epochs': 150,
            'patience': 25,
            'use_mixed_precision': True,
            'gpu_utilization': 'medium',
            'cpu_utilization': 'medium',
        },
        '3': {
            'name': '高性能（推奨）',
            'description': 'CPU/GPUを積極的に使用。速度と精度のバランス。',
            'batch_size': 32,
            'workers': 8,
            'max_queue_size': 20,
            'use_multiprocessing': True,
            'max_epochs': 200,
            'patience': 30,
            'use_mixed_precision': True,
            'gpu_utilization': 'high',
            'cpu_utilization': 'high',
        },
        '4': {
            'name': '最大性能（フル活用）',
            'description': 'CPU/GPUを最大限に使用。最高の精度と速度を追求。',
            'batch_size': 64,
            'workers': 16,
            'max_queue_size': 50,
            'use_multiprocessing': True,
            'max_epochs': 300,
            'patience': 50,
            'use_mixed_precision': False,  # フル精度
            'gpu_utilization': 'maximum',
            'cpu_utilization': 'maximum',
        },
        '5': {
            'name': '超高性能（実験的）',
            'description': 'CPU/GPUを超限界まで使用。実験的設定で最高の性能を追求（メモリが十分な場合のみ推奨）。',
            'batch_size': 96,
            'workers': 24,
            'max_queue_size': 100,
            'use_multiprocessing': True,
            'max_epochs': 400,
            'patience': 60,
            'use_mixed_precision': False,  # フル精度
            'gpu_utilization': 'maximum',
            'cpu_utilization': 'maximum',
        },
    }
    
    def __init__(self):
        self.selected_profile = None
        self.config = {}
    
    def display_menu(self):
        """メニューを表示"""
        print("\n" + "=" * 80)
        print("学習時のGPU/CPU負荷レベルを選択してください")
        print("=" * 80)
        print()
        print("【クイック選択】")
        for key, profile in self.RESOURCE_PROFILES.items():
            print(f"[{key}] {profile['name']}")
            print(f"    説明: {profile['description']}")
            print(f"    Batch Size: {profile['batch_size']}, "
                  f"Workers: {profile['workers']}, "
                  f"Epochs: {profile['max_epochs']}")
            print(f"    GPU: {profile['gpu_utilization']}, "
                  f"CPU: {profile['cpu_utilization']}")
            print()
        print("[5] GPU/CPUを個別に選択")
        print()
    
    def select_custom_profile(self):
        """GPU/CPUを個別に選択"""
        print("\n" + "=" * 80)
        print("GPU/CPUを個別に選択")
        print("=" * 80)
        print()
        
        # GPU選択
        print("【GPU負荷レベル】")
        for level_key, level_info in self.GPU_LEVELS.items():
            print(f"  [{level_key[0]}] {level_info['name']} - Batch Size: {level_info['batch_size']}")
        print()
        
        gpu_choice = None
        while gpu_choice not in self.GPU_LEVELS:
            try:
                choice = input("GPUレベルを選択 (l/m/h/x): ").strip().lower()
                if choice == 'l':
                    gpu_choice = 'low'
                elif choice == 'm':
                    gpu_choice = 'medium'
                elif choice == 'h':
                    gpu_choice = 'high'
                elif choice == 'x':
                    gpu_choice = 'maximum'
                else:
                    print("無効な選択です。l/m/h/xを入力してください。")
            except KeyboardInterrupt:
                return None
        
        # CPU選択
        print("\n【CPU負荷レベル】")
        for level_key, level_info in self.CPU_LEVELS.items():
            print(f"  [{level_key[0]}] {level_info['name']} - Workers: {level_info['workers']}")
        print()
        
        cpu_choice = None
        while cpu_choice not in self.CPU_LEVELS:
            try:
                choice = input("CPUレベルを選択 (l/m/h/x): ").strip().lower()
                if choice == 'l':
                    cpu_choice = 'low'
                elif choice == 'm':
                    cpu_choice = 'medium'
                elif choice == 'h':
                    cpu_choice = 'high'
                elif choice == 'x':
                    cpu_choice = 'maximum'
                else:
                    print("無効な選択です。l/m/h/xを入力してください。")
            except KeyboardInterrupt:
                return None
        
        # カスタム設定を作成
        gpu_config = self.GPU_LEVELS[gpu_choice]
        cpu_config = self.CPU_LEVELS[cpu_choice]
        
        # 最大学習回数とパテンスはGPU/CPUの平均レベルから決定
        avg_level = ['low', 'medium', 'high', 'maximum'].index(gpu_choice) + ['low', 'medium', 'high', 'maximum'].index(cpu_choice)
        avg_level = avg_level / 2
        
        if avg_level < 1:
            max_epochs = 100
            patience = 20
        elif avg_level < 2:
            max_epochs = 150
            patience = 25
        elif avg_level < 3:
            max_epochs = 200
            patience = 30
        else:
            max_epochs = 300
            patience = 50
        
        self.config = {
            'name': f'カスタム（GPU: {gpu_config["name"]}, CPU: {cpu_config["name"]}）',
            'description': f'GPUとCPUを個別に設定',
            'batch_size': gpu_config['batch_size'],
            'workers': cpu_config['workers'],
            'max_queue_size': cpu_config['max_queue_size'],
            'use_multiprocessing': cpu_config['use_multiprocessing'],
            'max_epochs': max_epochs,
            'patience': patience,
            'use_mixed_precision': gpu_config['use_mixed_precision'],
            'gpu_utilization': gpu_choice,
            'cpu_utilization': cpu_choice,
        }
        
        self.selected_profile = 'custom'
        return self.config
    
    def select_profile(self):
        """プロファイルを選択"""
        self.display_menu()
        
        while True:
            try:
                choice = input("選択してください (1-5): ").strip()
                
                if choice == '5':
                    # GPU/CPUを個別に選択
                    custom_config = self.select_custom_profile()
                    if custom_config:
                        return custom_config
                    else:
                        continue
                elif choice in self.RESOURCE_PROFILES:
                    self.selected_profile = choice
                    self.config = self.RESOURCE_PROFILES[choice].copy()
                    return self.config
                else:
                    print("無効な選択です。1-5の数字を入力してください。")
            except KeyboardInterrupt:
                print("\nキャンセルされました。")
                sys.exit(0)
            except Exception as e:
                print(f"エラー: {e}")
                print("1-5の数字を入力してください。")
    
    def apply_config_to_environment(self):
        """選択された設定を環境変数に適用"""
        if not self.config:
            return
        
        # 環境変数に設定
        os.environ['TRAINING_BATCH_SIZE'] = str(self.config['batch_size'])
        os.environ['TRAINING_WORKERS'] = str(self.config['workers'])
        os.environ['TRAINING_MAX_QUEUE_SIZE'] = str(self.config['max_queue_size'])
        os.environ['TRAINING_USE_MULTIPROCESSING'] = str(self.config['use_multiprocessing'])
        os.environ['TRAINING_MAX_EPOCHS'] = str(self.config['max_epochs'])
        os.environ['TRAINING_PATIENCE'] = str(self.config['patience'])
        os.environ['TRAINING_USE_MIXED_PRECISION'] = str(self.config['use_mixed_precision'])
        os.environ['TRAINING_GPU_UTILIZATION'] = self.config['gpu_utilization']
        os.environ['TRAINING_CPU_UTILIZATION'] = self.config['cpu_utilization']
        
        # FORCE_ACCURACYフラグ（最大性能モードの場合）
        if self.selected_profile == '4':
            os.environ['FORCE_ACCURACY'] = '1'
        else:
            os.environ.pop('FORCE_ACCURACY', None)
        
        print("\n" + "=" * 80)
        print("設定が適用されました")
        print("=" * 80)
        print(f"選択されたプロファイル: {self.config['name']}")
        print(f"Batch Size: {self.config['batch_size']}")
        print(f"Workers: {self.config['workers']}")
        print(f"Max Queue Size: {self.config['max_queue_size']}")
        print(f"Max Epochs: {self.config['max_epochs']}")
        print(f"GPU Utilization: {self.config['gpu_utilization']}")
        print(f"CPU Utilization: {self.config['cpu_utilization']}")
        print("=" * 80)
    
    def get_config(self):
        """設定を取得"""
        return self.config.copy()

def main():
    """メイン関数（対話型選択）"""
    selector = TrainingResourceSelector()
    config = selector.select_profile()
    selector.apply_config_to_environment()
    
    print("\n設定完了！学習を開始できます。")
    print("\n実行コマンド:")
    print("  python scripts/train_4class_sparse_ensemble.py")
    print()
    
    return config

def quick_select(profile_id=None):
    """クイック選択（対話なしでプロファイルを指定）"""
    selector = TrainingResourceSelector()
    if profile_id and profile_id in selector.RESOURCE_PROFILES:
        selector.selected_profile = profile_id
        selector.config = selector.RESOURCE_PROFILES[profile_id].copy()
        selector.apply_config_to_environment()
        return selector.config
    else:
        return selector.select_profile()

if __name__ == '__main__':
    main()

