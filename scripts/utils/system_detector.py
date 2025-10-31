#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
システムスペック検出ユーティリティ
PCのスペックを検出して、それに応じた最適な設定を提供
"""

import sys
# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
import os
import platform
import psutil

if sys.platform.startswith('win'):
    # Safer UTF-8 configuration without using detach (which is unavailable in some consoles)
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# psutilのインポートが失敗する場合のフォールバック
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("警告: psutilがインストールされていません。'pip install psutil'を実行してください。")

class SystemSpecDetector:
    """システムスペックを検出して最適化設定を提供"""
    
    def __init__(self):
        self.specs = self.detect_specs()
        self.config = self.optimize_config()
    
    def detect_specs(self):
        """システムスペックを検出"""
        if not HAS_PSUTIL:
            # psutilがない場合はデフォルト値を返ム（安全のためのートPC設定）
            print("警告: psutilがないため、デフォルトの軽量設定を使用します。")
            return {
                'cpu_info': platform.processor() if hasattr(platform, 'processor') else 'Unknown',
                'cpu_cores_physical': 4,
                'cpu_cores_logical': 8,
                'memory_gb': 16.0,
                'is_notebook': True,  # 安全のためのートPCとして扱う
                'is_high_end': False,
                'gpu_info': self._detect_gpu(),
                'device_type': 'notebook',
            }
        
        specs = {
            'cpu_info': platform.processor(),
            'cpu_cores_physical': psutil.cpu_count(logical=False),
            'cpu_cores_logical': psutil.cpu_count(logical=True),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'is_notebook': False,
            'is_high_end': False,
            'gpu_info': self._detect_gpu(),
        }
        
        # のートPC判定（CPUコア数とメモリから推定）
        if specs['cpu_cores_physical'] <= 4 and specs['memory_gb'] <= 32:
            specs['is_notebook'] = True
            specs['device_type'] = 'notebook'
        else:
            specs['device_type'] = 'desktop'
        
        # ハイエンド判定
        if (specs['cpu_cores_physical'] >= 8 and 
            specs['memory_gb'] >= 32 and 
            'RTX' in specs['gpu_info']):
            specs['is_high_end'] = True
        
        return specs
    
    def _detect_gpu(self):
        """GPU情報を検出"""
        gpu_info = "Unknown"
        
        # NVIDIA GPU検出
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                return gpu_info
        except:
            pass
        
        # GPUtilを使用（利用可能な場合）
        if HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_info = gpus[0].name
                    return gpu_info
            except:
                pass
        
        # Windows DirectX検出
        if sys.platform.startswith('win'):
            try:
                import subprocess
                result = subprocess.run(
                    ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and 'Name' in result.stdout:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip() and 'Name' not in line:
                            gpu_info = line.strip()
                            break
            except:
                pass
        
        return gpu_info if gpu_info != "Unknown" else "No GPU detected"
    
    def optimize_config(self):
        """スペックに応じた最適化設定を生成"""
        specs = self.specs
        
        if specs['is_notebook']:
            # のートPC向け軽量設定
            config = {
                'batch_size': 8,  # 軽量
                'workers': 2,  # CPUワーカー数
                'max_queue_size': 5,  # キューサイズ
                'use_multiprocessing': False,  # マルチプロセス無効
                'epochs': 100,  # エポック数（減らム）
                'patience': 20,  # Early stopping
                'image_size': 224,  # 画像サイズ
                'augmentation_intensity': 'light',  # 軽量データ拡張
                'model_complexity': 'low',  # 軽量モデル
                'use_mixed_precision': True,  # 混合精度で高速化
            }
        elif specs['is_high_end']:
            # ハイエンドデスクトップ向け高性能設定
            config = {
                'batch_size': 32,  # 大容量
                'workers': specs['cpu_cores_logical'],  # 全コア使用
                'max_queue_size': 20,  # 大きいキュー
                'use_multiprocessing': True,  # マルチプロセス有効
                'epochs': 200,  # 最大エポック数
                'patience': 30,  # Early stopping
                'image_size': 260,  # 高解像度
                'augmentation_intensity': 'heavy',  # 強力なデータ拡張
                'model_complexity': 'high',  # 高性能モデル
                'use_mixed_precision': False,  # フル精度
            }
        else:
            # 標準デスクトップ向け中程度設定
            config = {
                'batch_size': 16,  # 中程度
                'workers': max(4, specs['cpu_cores_logical'] // 2),  # コア数の半分
                'max_queue_size': 10,  # 中程度キュー
                'use_multiprocessing': True,
                'epochs': 150,  # 中程度エポック数
                'patience': 25,
                'image_size': 240,  # 中程度解像度
                'augmentation_intensity': 'medium',  # 中程度データ拡張
                'model_complexity': 'medium',
                'use_mixed_precision': True,
            }
        
        return config
    
    def print_specs(self):
        """検出したスペックを表示"""
        print("=" * 60)
        print("システムスペック検出結果")
        print("=" * 60)
        print(f"CPU: {self.specs['cpu_info']}")
        print(f"物理コア数: {self.specs['cpu_cores_physical']}")
        print(f"論理コア数: {self.specs['cpu_cores_logical']}")
        print(f"メモリ: {self.specs['memory_gb']:.1f} GB")
        print(f"GPU: {self.specs['gpu_info']}")
        print(f"デバイスタイプ: {self.specs['device_type']}")
        print(f"ハイエンド: {'はい' if self.specs['is_high_end'] else 'いいえ'}")
        print("=" * 60)
        
    def print_config(self):
        """最適化設定を表示"""
        print("\n" + "=" * 60)
        print("最適化された設定")
        print("=" * 60)
        for key, value in self.config.items():
            print(f"{key}: {value}")
        print("=" * 60)

def get_optimized_config():
    """最適化された設定を取得するヘルパー関数"""
    detector = SystemSpecDetector()
    return detector.config, detector.specs

if __name__ == "__main__":
    detector = SystemSpecDetector()
    detector.print_specs()
    detector.print_config()

