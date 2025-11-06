#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
システムスペック検出ユーティリティ
PCのスペックを検出して、それに応じた最適な設定を提供
"""

import sys
import os

# UTF-8 encoding for Windows (環境変数を使用、より安全)
if sys.platform.startswith('win'):
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    os.environ.setdefault('PYTHONUTF8', '1')

import platform
import sys

# psutilのインポートが失敗する場合のフォールバック
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None
    print("警告: psutilがインストールされていません。Windows APIでシステム情報を取得します。")

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

class SystemSpecDetector:
    """システムスペックを検出して最適化設定を提供"""
    
    def __init__(self):
        self.specs = self.detect_specs()
        self.config = self.optimize_config()
    
    def detect_specs(self):
        """システムスペックを検出"""
        # psutilがない場合、Windows APIで取得を試みる
        if not HAS_PSUTIL:
            print("警告: psutilがないため、Windows APIでシステム情報を取得します。")
            specs = self._detect_specs_windows_api()
            if specs:
                specs.update(self._detect_cpu_detailed())
                specs.update(self._detect_gpu_detailed())
                specs.update(self._detect_cuda_availability())
                return specs
            # フォールバック: デフォルト値
            return {
                'cpu_info': platform.processor() if hasattr(platform, 'processor') else 'Unknown',
                'cpu_cores_physical': 4,
                'cpu_cores_logical': 8,
                'memory_gb': 16.0,
                'is_notebook': True,
                'is_high_end': False,
                'gpu_info': self._detect_gpu(),
                'device_type': 'notebook',
            }
        
        if HAS_PSUTIL and psutil:
            specs = {
                'cpu_info': platform.processor(),
                'cpu_cores_physical': psutil.cpu_count(logical=False),
                'cpu_cores_logical': psutil.cpu_count(logical=True),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'is_notebook': False,
                'is_high_end': False,
                'gpu_info': self._detect_gpu(),
            }
        else:
            # psutilがない場合はWindows APIで取得
            specs = self._detect_specs_windows_api()
            if not specs:
                # フォールバック
                specs = {
                    'cpu_info': platform.processor() if hasattr(platform, 'processor') else 'Unknown',
                    'cpu_cores_physical': 0,
                    'cpu_cores_logical': 0,
                    'memory_gb': 0.0,
                    'is_notebook': False,
                    'is_high_end': False,
                    'gpu_info': self._detect_gpu(),
                }
        
        # CPU詳細情報を追加
        specs.update(self._detect_cpu_detailed())
        
        # GPU詳細情報を追加
        specs.update(self._detect_gpu_detailed())
        
        # CUDA/TensorFlow GPU利用可能性をチェック
        specs.update(self._detect_cuda_availability())
        
        # ノートPC判定（複数の基準で判定）
        # 基準1: CPUコア数とメモリ（低スペック = ノートPCの可能性が高い）
        low_spec = specs['cpu_cores_physical'] <= 4 and specs['memory_gb'] <= 32
        
        # 基準2: 中スペックでも軽量モードが必要な場合（ノートPCの可能性）
        # 8コア以下、64GB以下 → ノートPCの可能性
        medium_spec = specs['cpu_cores_physical'] <= 8 and specs['memory_gb'] <= 64
        
        # 基準3: バッテリー検出（Windows環境）
        has_battery = False
        if sys.platform.startswith('win'):
            try:
                import subprocess
                result = subprocess.run(
                    ['wmic', 'path', 'Win32_Battery', 'get', 'Availability'],
                    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=3
                )
                if result.returncode == 0 and 'Availability' in result.stdout:
                    # バッテリーが見つかった場合
                    lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip() and l.strip().isdigit()]
                    if lines:
                        has_battery = True
            except:
                pass
        
        # 判定ロジック
        # 低スペック OR (中スペック AND バッテリー検出) → ノートPC
        if low_spec or (medium_spec and has_battery):
            specs['is_notebook'] = True
            specs['device_type'] = 'notebook'
        elif medium_spec and not has_battery:
            # 中スペックでバッテリーなし → デスクトップの可能性が高いが、念のため軽量モード推奨
            specs['is_notebook'] = False
            specs['device_type'] = 'desktop'
            specs['recommend_lightweight'] = True  # 軽量モード推奨フラグ
        else:
            specs['is_notebook'] = False
            specs['device_type'] = 'desktop'
        
        # バッテリー情報を追加
        specs['has_battery'] = has_battery
        
        # ハイエンド判定
        if (specs['cpu_cores_physical'] >= 8 and 
            specs['memory_gb'] >= 32 and 
            'RTX' in specs['gpu_info']):
            specs['is_high_end'] = True
        
        return specs
    
    def _detect_cpu_detailed(self):
        """CPUの詳細情報を検出"""
        cpu_details = {
            'cpu_freq_current_mhz': 0,
            'cpu_freq_min_mhz': 0,
            'cpu_freq_max_mhz': 0,
            'cpu_architecture': platform.machine(),
            'cpu_platform': platform.platform(),
            'cpu_bits': platform.architecture()[0],
            'cpu_process': platform.processor(),
        }
        
        # CPU周波数情報
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_details['cpu_freq_current_mhz'] = cpu_freq.current if cpu_freq.current else 0
                cpu_details['cpu_freq_min_mhz'] = cpu_freq.min if cpu_freq.min else 0
                cpu_details['cpu_freq_max_mhz'] = cpu_freq.max if cpu_freq.max else 0
        except:
            pass
        
        # Windows環境での詳細情報取得
        if sys.platform.startswith('win'):
            try:
                import subprocess
                # CPU名を詳細に取得
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'Name'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    encoding='utf-8',
                    errors='replace'
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip() and 'Name' not in line:
                            cpu_details['cpu_full_name'] = line.strip()
                            break
                
                # キャッシュサイズ情報
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'L3CacheSize'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    encoding='utf-8',
                    errors='replace'
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip() and line.strip().isdigit():
                            cpu_details['cpu_l3_cache_mb'] = int(line.strip())
                            break
            except:
                pass
        
        return cpu_details
    
    def _detect_gpu_detailed(self):
        """GPUの詳細情報を検出"""
        gpu_details = {
            'gpu_vram_total_gb': 0,
            'gpu_vram_used_gb': 0,
            'gpu_driver_version': '',
            'gpu_cuda_version': '',
            'gpu_compute_capability': '',
            'gpu_cuda_cores': 0,
        }
        
        # NVIDIA GPU詳細情報
        try:
            import subprocess
            # GPU情報を一括取得
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version,compute_cap', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(', ')
                    if len(parts) >= 5:
                        gpu_details['gpu_name_full'] = parts[0].strip()
                        gpu_details['gpu_vram_total_mb'] = int(float(parts[1].strip())) if parts[1].strip().isdigit() or '.' in parts[1].strip() else 0
                        gpu_details['gpu_vram_used_mb'] = int(float(parts[2].strip())) if parts[2].strip().isdigit() or '.' in parts[2].strip() else 0
                        gpu_details['gpu_driver_version'] = parts[3].strip()
                        gpu_details['gpu_compute_capability'] = parts[4].strip()
                        gpu_details['gpu_vram_total_gb'] = gpu_details['gpu_vram_total_mb'] / 1024
                        gpu_details['gpu_vram_used_gb'] = gpu_details['gpu_vram_used_mb'] / 1024
        except:
            pass
        
        # CUDAバージョン
        try:
            import subprocess
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        parts = line.split('release')
                        if len(parts) > 1:
                            version = parts[1].strip().split(',')[0]
                            gpu_details['gpu_cuda_version'] = version
                            break
        except:
            pass
        
        # GPUtilを使用した詳細情報（利用可能な場合）
        if HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    if not gpu_details.get('gpu_vram_total_gb'):
                        gpu_details['gpu_vram_total_gb'] = gpu.memoryTotal / 1024
                    if not gpu_details.get('gpu_vram_used_gb'):
                        gpu_details['gpu_vram_used_gb'] = gpu.memoryUsed / 1024
                    gpu_details['gpu_temperature'] = gpu.temperature
                    gpu_details['gpu_load'] = gpu.load * 100
            except:
                pass
        
        return gpu_details
    
    def _detect_cuda_availability(self):
        """CUDAとTensorFlow GPUの利用可能性を検出"""
        cuda_info = {
            'cuda_available': False,
            'cuda_device_type': 'CPU',
            'tensorflow_gpu_available': False,
            'tensorflow_devices': [],
            'directml_available': False,
        }
        
        # TensorFlowのGPU検出
        try:
            import tensorflow as tf
            import os
            
            # TensorFlowのログレベルを設定（警告を抑制）
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
            
            # DirectML検出（Windows - 優先的にチェック）
            dml_devices = []
            if sys.platform.startswith('win'):
                try:
                    dml_devices = tf.config.list_physical_devices('DML')
                    if dml_devices:
                        cuda_info['directml_available'] = True
                        cuda_info['cuda_device_type'] = 'DirectML (Windows GPU)'
                        cuda_info['tensorflow_devices'] = [d.name for d in dml_devices]
                        cuda_info['cuda_available'] = True  # DirectMLもGPUとして扱う
                        print("[INFO] DirectML GPU detected:", dml_devices)
                except Exception as e:
                    print(f"[DEBUG] DirectML detection error: {e}")
            
            # CUDA GPU検出（DirectMLが見つからない場合のみ）
            if not dml_devices:
                gpus = []
                try:
                    gpus = tf.config.list_physical_devices('GPU')
                except Exception as e:
                    cuda_info['tensorflow_error'] = f"list_physical_devices error: {str(e)}"
                
                # 代替方法: tf.test.is_gpu_available()を使用
                if not gpus:
                    try:
                        if tf.test.is_gpu_available():
                            # GPUが利用可能だが、list_physical_devicesで検出されない場合
                            # nvidia-smiで確認
                            import subprocess
                            result = subprocess.run(
                                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5
                            )
                            if result.returncode == 0 and result.stdout.strip():
                                # GPUが見つかったがTensorFlowで認識されていない
                                cuda_info['gpu_detected_via_nvidia_smi'] = True
                                cuda_info['tensorflow_gpu_available'] = False
                                cuda_info['cuda_device_type'] = 'GPU (TensorFlow未認識)'
                    except:
                        pass
                
                if gpus:
                    cuda_info['cuda_available'] = True
                    cuda_info['tensorflow_gpu_available'] = True
                    cuda_info['cuda_device_type'] = 'CUDA GPU'
                    cuda_info['tensorflow_devices'] = [f'GPU:{i}' for i in range(len(gpus))]
                    
                    # CUDAバージョン情報
                    try:
                        build_info = tf.sysconfig.get_build_info()
                        if 'cuda_version' in build_info:
                            cuda_info['cuda_build_version'] = build_info['cuda_version']
                        if 'cudnn_version' in build_info:
                            cuda_info['cudnn_version'] = build_info['cudnn_version']
                    except:
                        pass
            
            # TensorFlowで認識されているデバイス一覧
            try:
                all_devices = tf.config.list_physical_devices()
                cuda_info['tensorflow_all_devices'] = [d.name for d in all_devices]
            except:
                pass
                
        except ImportError:
            cuda_info['tensorflow_installed'] = False
        except Exception as e:
            cuda_info['tensorflow_error'] = str(e)
        
        # nvidia-smiでGPUとドライバーの確認（優先度を上げる）
        try:
            import subprocess
            # GPU名とドライバーバージョンを取得
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0].strip():
                    parts = lines[0].split(', ')
                    if len(parts) >= 2:
                        gpu_name = parts[0].strip()
                        driver_version = parts[1].strip()
                        cuda_info['nvidia_gpu_name'] = gpu_name
                        cuda_info['nvidia_driver_version'] = driver_version
                        cuda_info['nvidia_gpu_detected'] = True
                        
                        # ドライバーのバージョンから対応CUDAバージョンを推測
                        try:
                            driver_major = int(driver_version.split('.')[0])
                            # ドライバーバージョンとCUDAバージョンの対応関係
                            if driver_major >= 550:
                                cuda_info['nvidia_cuda_version'] = '12.x+'
                            elif driver_major >= 525:
                                cuda_info['nvidia_cuda_version'] = '11.8+'
                            elif driver_major >= 510:
                                cuda_info['nvidia_cuda_version'] = '11.x'
                            else:
                                cuda_info['nvidia_cuda_version'] = '10.x/11.x'
                        except:
                            cuda_info['nvidia_cuda_version'] = 'Unknown'
                        
                        # nvidia-smiでGPUが見つかったが、TensorFlowで認識されていない場合
                        if not cuda_info.get('tensorflow_gpu_available', False) and not cuda_info.get('cuda_available', False):
                            # GPUは存在するがTensorFlowで認識されていない
                            cuda_info['cuda_device_type'] = 'GPU (CUDA利用可能、TensorFlow未認識)'
                            # 推奨メッセージを追加
                            cuda_info['recommendation'] = f'TensorFlowでGPUを認識するには、CUDA対応のTensorFlow（tensorflow-gpuまたはtensorflow[and-cuda]）のインストールが必要です。現在のTensorFlowはCPU版のためGPUを認識できません。'
        except Exception as e:
            cuda_info['nvidia_smi_error'] = str(e)
            pass
        
        # CUDA Toolkitのバージョン確認（nvcc）
        try:
            import subprocess
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        parts = line.split('release')
                        if len(parts) > 1:
                            version = parts[1].strip().split(',')[0]
                            cuda_info['cuda_toolkit_version'] = version
                            cuda_info['cuda_toolkit_installed'] = True
                            break
        except:
            pass
        
        return cuda_info
    
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
    
    def _detect_specs_windows_api(self):
        """Windows APIでシステムスペックを検出（psutilがない場合）"""
        import subprocess
        import os
        
        specs = {
            'cpu_info': 'Unknown',
            'cpu_cores_physical': 0,
            'cpu_cores_logical': 0,
            'memory_gb': 0.0,
            'is_notebook': False,
            'is_high_end': False,
            'gpu_info': 'Unknown',
            'device_type': 'unknown',
        }
        
        try:
            # CPU名を取得
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'Name'],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    line_stripped = line.strip()
                    if line_stripped and 'Name' not in line_stripped and line_stripped.lower() != 'name':
                        specs['cpu_info'] = line_stripped
                        print(f"[DEBUG] CPU detected via wmic: {line_stripped}")
                        break
            
            # platform.processor()も試行
            if specs['cpu_info'] == 'Unknown':
                try:
                    cpu_from_platform = platform.processor()
                    if cpu_from_platform:
                        specs['cpu_info'] = cpu_from_platform
                        print(f"[DEBUG] CPU detected via platform: {cpu_from_platform}")
                except:
                    pass
            
            # CPUコア数を取得
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'NumberOfCores', 'NumberOfLogicalProcessors'],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    line_stripped = line.strip()
                    # ヘッダー行をスキップ
                    if 'NumberOfCores' in line_stripped or 'NumberOfLogicalProcessors' in line_stripped:
                        continue
                    if line_stripped:
                        parts = line_stripped.split()
                        # 数字の部分を探す
                        nums = [p for p in parts if p.isdigit()]
                        if len(nums) >= 2:
                            specs['cpu_cores_physical'] = int(nums[0])
                            specs['cpu_cores_logical'] = int(nums[1])
                            print(f"[DEBUG] CPU cores detected: Physical={nums[0]}, Logical={nums[1]}")
                            break
                        elif len(nums) == 1:
                            # 論理コアのみ取得できた場合
                            specs['cpu_cores_logical'] = int(nums[0])
                            # 物理コアは論理コアの半分と仮定（HTが有効な場合）
                            if specs['cpu_cores_logical'] > 0:
                                specs['cpu_cores_physical'] = specs['cpu_cores_logical'] // 2
                            print(f"[DEBUG] CPU logical cores detected: {nums[0]}")
            
            # 環境変数からも試行
            if specs['cpu_cores_logical'] == 0:
                try:
                    num_proc = os.environ.get('NUMBER_OF_PROCESSORS', '0')
                    if num_proc.isdigit():
                        specs['cpu_cores_logical'] = int(num_proc)
                        specs['cpu_cores_physical'] = int(num_proc) // 2  # HTが有効な場合の仮定
                        print(f"[DEBUG] CPU cores from env: {specs['cpu_cores_logical']}")
                except:
                    pass
            
            # メモリを取得
            result = subprocess.run(
                ['wmic', 'computersystem', 'get', 'TotalPhysicalMemory'],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip() and line.strip().isdigit():
                        memory_bytes = int(line.strip())
                        specs['memory_gb'] = memory_bytes / (1024**3)
                        break
            
            # GPU情報を取得（必ず実行）
            gpu_info_result = self._detect_gpu()
            if gpu_info_result and gpu_info_result != 'Unknown':
                specs['gpu_info'] = gpu_info_result
            else:
                # フォールバック: nvidia-smiで直接取得
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        specs['gpu_info'] = result.stdout.strip()
                    else:
                        specs['gpu_info'] = 'Unknown'
                except:
                    specs['gpu_info'] = 'Unknown'
            
            # システムタイプを判定（メイン関数と同じロジック）
            # バッテリー検出
            has_battery = False
            try:
                import subprocess
                result = subprocess.run(
                    ['wmic', 'path', 'Win32_Battery', 'get', 'Availability'],
                    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=3
                )
                if result.returncode == 0 and 'Availability' in result.stdout:
                    lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip() and l.strip().isdigit()]
                    if lines:
                        has_battery = True
            except:
                pass
            
            low_spec = specs['cpu_cores_physical'] <= 4 and specs['memory_gb'] <= 32
            medium_spec = specs['cpu_cores_physical'] <= 8 and specs['memory_gb'] <= 64
            
            if low_spec or (medium_spec and has_battery):
                specs['is_notebook'] = True
                specs['device_type'] = 'notebook'
            elif medium_spec and not has_battery:
                specs['is_notebook'] = False
                specs['device_type'] = 'desktop'
                specs['recommend_lightweight'] = True
            else:
                specs['is_notebook'] = False
                specs['device_type'] = 'desktop'
            
            specs['has_battery'] = has_battery
            
            # ハイエンド判定
            if (specs['cpu_cores_physical'] >= 8 and 
                specs['memory_gb'] >= 32 and 
                'RTX' in specs['gpu_info']):
                specs['is_high_end'] = True
            
        except Exception as e:
            print(f"[DEBUG] Windows API detection error: {e}")
            import traceback
            traceback.print_exc()
        
        return specs if (specs['cpu_cores_logical'] > 0 or specs['memory_gb'] > 0) else None
    
    def optimize_config(self):
        """スペックに応じた最適化設定を生成"""
        specs = self.specs
        
        if specs['is_notebook']:
            # ノートPC向け軽量設定
            config = {
                'batch_size': 8,  # 軽量
                'workers': 2,  # CPUワーカー数
                'max_queue_size': 5,  # キューサイズ
                'use_multiprocessing': False,  # マルチプロセス無効
                'epochs': 100,  # 最大学習回数（減らム）
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
                'epochs': 200,  # 最大学習回数
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
                'epochs': 150,  # 中程度の学習回数
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
        if 'has_battery' in self.specs:
            print(f"バッテリー検出: {'あり' if self.specs['has_battery'] else 'なし'}")
        if self.specs.get('recommend_lightweight', False):
            print("※ 軽量モード推奨（中スペックデスクトップ）")
        print(f"ハイエンド: {'はい' if self.specs['is_high_end'] else 'いいえ'}")
        print("=" * 60)
        print("\n【判定基準】")
        print("ノートPC判定:")
        print("  - 低スペック: 物理コア数 ≤ 4 かつ メモリ ≤ 32GB")
        print("  - または中スペック（コア数 ≤ 8、メモリ ≤ 64GB）かつバッテリー検出")
        print("デスクトップ判定:")
        print("  - 上記以外、または中スペックでバッテリーなし")
        print("ハイエンド判定:")
        print("  - 物理コア数 ≥ 8 かつ メモリ ≥ 32GB かつ RTX GPU")
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

