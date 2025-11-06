#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-Class Ultra-High Accuracy Deep Learning with SPARSE MODELING
Complete defect detection system for good, black_spot, chipping, and scratch
WITH CLEAR PROGRESS DISPLAY AND SYSTEM-OPTIMIZED SETTINGS
"""

import os
# Prefer GPU via DirectML on Windows if available
if os.name == 'nt' and 'TF_USE_DIRECTML' not in os.environ:
    os.environ['TF_USE_DIRECTML'] = '1'

# WSL2環境の早期検出とXLA設定（TensorFlowインポート前）
# libdeviceパスを検出してXLAを有効化（無効化ではなく、正しく設定）
# 環境変数を最優先で設定（TensorFlow/Kerasが読み込まれる前に）
# 重要: これらの環境変数はTensorFlow/Kerasがインポートされる前に設定する必要があります
# libdeviceのパスを自動検出（nvidia-nvjitlink-cu12パッケージから）
xla_flags_base = '--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found --xla_gpu_force_compilation_parallelism=1 --xla_gpu_deterministic_ops=true'
try:
    import site
    from pathlib import Path
    # nvidia-nvjitlink-cu12パッケージからlibdeviceを探す
    try:
        import nvidia.nvjitlink
        package_path = Path(nvidia.nvjitlink.__file__).parent
        libdevice_files = list(package_path.rglob('libdevice*.bc'))
        if libdevice_files:
            libdevice_dir = libdevice_files[0].parent
            xla_flags_base += f' --xla_gpu_cuda_data_dir={libdevice_dir}'
            print(f"[INFO] libdeviceパスを検出: {libdevice_dir}")
    except ImportError:
        pass
    # CUDA Toolkitの標準パスも確認
    cuda_paths = ['/usr/local/cuda', '/opt/cuda', '/usr/local/cuda-12.5', '/usr/local/cuda-12.4']
    for cuda_path in cuda_paths:
        if os.path.exists(f'{cuda_path}/nvvm/libdevice'):
            xla_flags_base += f' --xla_gpu_cuda_data_dir={cuda_path}/nvvm'
            print(f"[INFO] CUDA Toolkitのlibdeviceパスを検出: {cuda_path}/nvvm")
            break
except Exception as e:
    print(f"[WARN] libdeviceパス検出エラー: {e}")

os.environ['XLA_FLAGS'] = xla_flags_base
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# WSL2環境でのマルチスレッド問題を回避するため、シングルスレッド処理を強制
# これは、tf.py_functionのコールバック登録問題を回避するため
try:
    if os.path.exists('/proc/version'):
        with open('/proc/version', 'r') as f:
            version_info = f.read()
            if 'microsoft' in version_info.lower() or 'wsl' in version_info.lower():
                os.environ['TF_NUM_INTEROP_THREADS'] = '1'
                os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
                print("[INFO] WSL2環境検出: シングルスレッド処理を強制（マルチスレッド問題回避）")
    if 'WSL_DISTRO_NAME' in os.environ or 'WSL_INTEROP' in os.environ:
        os.environ['TF_NUM_INTEROP_THREADS'] = '1'
        os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
        print("[INFO] WSL2環境検出（環境変数）: シングルスレッド処理を強制（マルチスレッド問題回避）")
except:
    pass

# libdeviceが見つかった場合はXLAを有効化、見つからない場合は無効化
libdevice_found = '--xla_gpu_cuda_data_dir=' in xla_flags_base and len(xla_flags_base.split('--xla_gpu_cuda_data_dir=')) > 1

if libdevice_found:
    # libdeviceが見つかった場合、XLAを有効化
    print("[INFO] libdeviceパスが見つかりました。XLAを有効化します。")
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=true'
    os.environ['TF_XLA_AUTO_JIT'] = '2'  # 自動JITコンパイルを有効化
    os.environ['TF_XLA_ENABLE_XLA_DEVICES'] = '1'
    print("[INFO] XLAを有効化する環境変数を設定しました")
else:
    # libdeviceが見つからない場合、XLAを無効化
    print("[INFO] libdeviceパスが見つかりませんでした。XLAを無効化します。")
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_cpu_global_jit=false'
    os.environ['TF_XLA_AUTO_JIT'] = '0'
    os.environ['TF_DISABLE_XLA'] = '1'
    os.environ['XLA_GPU_DEVICE'] = '0'
    os.environ['XLA_CPU_DEVICE'] = '0'
    os.environ['TF_XLA_ENABLE_XLA_DEVICES'] = '0'
    os.environ['TF_XLA_EXEC'] = '0'
    print("[INFO] XLAを無効化する環境変数を設定しました（libdevice/ptxasエラーを防ぐため）")

# Keras 3用の設定
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 警告を抑制
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import subprocess
import numpy as np
import cv2

# TensorFlowインポート前にXLA設定を再確認（libdeviceが見つかった場合は有効化を維持）
# libdeviceが見つかっていない場合のみ無効化
if not libdevice_found:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_cpu_global_jit=false'
    os.environ['TF_XLA_AUTO_JIT'] = '0'
    os.environ['TF_DISABLE_XLA'] = '1'

import tensorflow as tf

# TensorFlowインポート後、libdeviceの検出結果に基づいてXLA設定を決定
try:
    if libdevice_found:
        # libdeviceが見つかった場合、XLAを有効化
        print("[INFO] TensorFlowインポート後、XLAを有効化しました（libdeviceパス設定済み）")
        # XLA JITを有効化（自動JITコンパイル）
        try:
            tf.config.optimizer.set_jit('autoclustering')  # 自動クラスタリングを有効化
        except:
            pass
        # GPU演算をCPUにフォールバック可能にする（エラー回避）
        try:
            tf.config.set_soft_device_placement(True)
            print("[INFO] GPU演算をCPUにフォールバック可能にしました")
        except:
            pass
    else:
        # libdeviceが見つからない場合、XLAを無効化してeager実行モードにフォールバック
        print("[INFO] TensorFlowインポート後、XLAを無効化しました（libdeviceパスが見つかりません）")
        # XLAコンパイラを無効化
        tf.config.optimizer.set_jit(False)
        # 環境変数を再設定（確実に無効化）
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_cpu_global_jit=false'
        os.environ['TF_XLA_AUTO_JIT'] = '0'
        os.environ['TF_XLA_ENABLE_XLA_DEVICES'] = '0'
        
        # TensorFlow実行モードをeagerに強制（XLAエラー回避のため）
        try:
            # TensorFlow 2.xの実行モードをeagerに設定
            tf.config.run_functions_eagerly(True)
            # tf.data関数もeager実行を強制（重要：XLAエラー回避）
            tf.data.experimental.enable_debug_mode()
            # すべての最適化を無効化（WSL2でのXLAエラー回避）
            tf.config.optimizer.set_experimental_options({
                'disable_meta_optimizer': True,
                'disable_model_pruning': True,
            })
            # GPU演算をCPUにフォールバック可能にする（XLAエラー回避）
            tf.config.set_soft_device_placement(True)
            print("[INFO] TensorFlow実行モードをeagerに強制設定しました")
            print("[INFO] tf.data関数もeager実行モードを有効化しました")
            print("[INFO] すべての最適化を無効化しました（XLAエラー回避）")
        except Exception as e:
            print(f"[WARN] eager実行設定エラー: {e}")
            import traceback
            traceback.print_exc()
    
    # GPUデバイスがある場合、メモリ成長を有効化
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass
    
    # Keras 3でeager実行を強制（グローバル設定）
    try:
        # Keras 3の実行モードを設定
        import keras
        # Keras 3では実行モードを直接設定できないため、環境変数で制御
        os.environ['TF_EAGER_EXECUTION'] = '1'
    except:
        pass
    
    if libdevice_found:
        print(f"[INFO] XLA設定: JIT={tf.config.optimizer.get_jit()} (有効化済み)")
    else:
        print(f"[INFO] XLA設定: JIT={tf.config.optimizer.get_jit()} (無効化済み)")
except Exception as e:
    print(f"[WARN] XLA無効化エラー: {e}")
    # エラー時も環境変数は設定済み
    import traceback
    traceback.print_exc()
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from collections import Counter

# Set UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# システムスペック検出をインポート
try:
    from system_detector import SystemSpecDetector
    HAS_SYSTEM_DETECTOR = True
except ImportError:
    HAS_SYSTEM_DETECTOR = False

# 実行時のCPU/GPU使用率取得（任意）
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

# HWiNFO Shared Memory読み取り（最優先テ最も信頼できる情報源）
try:
    # scripts/hwinfo/に移動したため、パスを調整
    import sys
    from pathlib import Path
    hwinfo_path = Path(__file__).parent / 'hwinfo' / 'hwinfo_reader.py'
    if hwinfo_path.exists():
        sys.path.insert(0, str(hwinfo_path.parent))
    from hwinfo_reader import read_hwinfo_shared_memory
    HAS_HWINFO_READER = True
except Exception:
    try:
        # フォールバック: 直接インポート
        from hwinfo.hwinfo_reader import read_hwinfo_shared_memory
        HAS_HWINFO_READER = True
    except Exception:
        HAS_HWINFO_READER = False

class ClearProgressSparseModelingFourClassWasherInspector:
    # データ読み込み結果をキャッシュ（クラス変数）
    _cached_data = None
    _cached_data_path = None
    
    def __init__(self, data_path="cs_AItraining_data"):
        self.data_path = Path(data_path)
        self.class_names = ['good', 'black_spot', 'chipping', 'scratch']  # 4 classes
        self.models = []
        self.histories = []
        self.viewer_process = None
        # 進捗ステータスの出力先
        self.status_path = Path('logs') / 'training_status.json'
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        
        # 全体の時間管理
        self.total_start_time = None
        self.section_start_time = None
        self.section_name = ""
        
        # 使用率キャッシュ
        self._last_metrics_time = 0.0
        self._last_metrics = {}
        
        # システムスペック検出と最適化設定
        if HAS_SYSTEM_DETECTOR:
            try:
                self.system_detector = SystemSpecDetector()
                self.system_config = self.system_detector.config
                self.system_specs = self.system_detector.specs
                
                print("\n" + "=" * 60)
                print("System spec detection completed")
                print("=" * 60)
                self.system_detector.print_specs()
                self.system_detector.print_config()
                print("=" * 60 + "\n")
                
                # システム設定に基づいてバッチサイズなどを調整
                # 環境変数からバッチサイズを取得（優先度: 環境変数 > システム設定）
                if 'TRAINING_BATCH_SIZE' in os.environ:
                    try:
                        self.batch_size = int(os.environ['TRAINING_BATCH_SIZE'])
                        print(f"[INFO] 環境変数からバッチサイズを取得: {self.batch_size}")
                    except ValueError:
                        self.batch_size = self.system_config['batch_size']
                else:
                    self.batch_size = self.system_config['batch_size']
                self.max_epochs = self.system_config['epochs']
                self.patience = self.system_config['patience']
                self.use_mixed_precision = self.system_config['use_mixed_precision']
                # Windowsでのmultiprocessing問題を回避
                if os.name == 'nt':  # Windows
                    self.workers = 1  # Windowsではworkers=1で安全
                    self.max_queue_size = 10
                    self.use_multiprocessing = False  # WindowsではFalse必須
                else:
                    self.workers = self.system_config.get('workers', max(4, (os.cpu_count() or 8)))
                    self.max_queue_size = self.system_config.get('max_queue_size', 20)
                    self.use_multiprocessing = self.system_config.get('use_multiprocessing', True)
                
            except Exception as e:
                print(f"Warning: System spec detection failed: {e}")
                print("Using default settings.")
                self.system_detector = None
                self.system_config = {}
                self.system_specs = {}
                self.batch_size = 16
                self.max_epochs = 200
                self.patience = 30
                self.use_mixed_precision = False
                # フル活用: workersを最大化（Windows対応）
                if os.name == 'nt':  # Windows
                    self.workers = 1  # Windowsではworkers=1で安全
                    self.max_queue_size = 10
                    self.use_multiprocessing = False  # WindowsではFalse必須
                else:
                    self.workers = os.cpu_count() or 16
                    self.max_queue_size = 32
                    self.use_multiprocessing = True
        else:
            print("Warning: system_detector module not found. Using default settings.")
            self.system_detector = None
            self.system_config = {}
            self.system_specs = {}
            self.batch_size = 16
            self.max_epochs = 200
            self.patience = 30
            self.use_mixed_precision = False
            # フル活用: workersを最大化（Windows対応）
            if os.name == 'nt':  # Windows
                self.workers = 1  # Windowsではworkers=1で安全
                self.max_queue_size = 10
                self.use_multiprocessing = False  # WindowsではFalse必須
            else:
                self.workers = os.cpu_count() or 16
                self.max_queue_size = 32
                self.use_multiprocessing = True

        # XLAとスレッド最適化を有効化（CPU/GPUを最大限活用）
        # WSL2ではptxasやlibdeviceが見つからない場合があるため、XLAを無効化
        try:
            # WSL2環境を検出（LinuxかつWSL特有のパスが存在する場合）
            is_wsl2 = False
            try:
                with open('/proc/version', 'r') as f:
                    version_info = f.read()
                    if 'microsoft' in version_info.lower() or 'wsl' in version_info.lower():
                        is_wsl2 = True
            except:
                pass
            
            # すべての環境でXLAを完全に無効化（libdevice/ptxasエラーを防ぐため）
            # より積極的にXLAを無効化
            tf.config.optimizer.set_jit(False)
            # 環境変数も再設定（確実に無効化）
            os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
            os.environ['TF_XLA_AUTO_JIT'] = '0'
            # XLAサービスを完全に無効化
            os.environ['TF_XLA_CPU_DEVICE'] = '0'
            os.environ['TF_XLA_GPU_DEVICE'] = '0'
            # GPU実行をXLAなしで強制
            try:
                # GPU設定を取得
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        # メモリ増加を許可（XLAを使わない）
                        tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass
            
            # WSL2環境では、すべての最適化を無効化（XLAエラー完全回避）
            if is_wsl2:
                try:
                    # すべての最適化を無効化
                    tf.config.optimizer.set_experimental_options({
                        'disable_meta_optimizer': True,
                        'disable_model_pruning': True,
                    })
                    # tf.data関数もeager実行を強制
                    tf.data.experimental.enable_debug_mode()
                    print("[WSL2] すべての最適化を無効化しました（XLAエラー完全回避）")
                except:
                    pass
            
            if is_wsl2:
                print("[WSL2] XLAを完全に無効化しました（libdevice/ptxasエラーを防ぐため）")
                print("[WSL2] 通常のTensorFlow実行を使用します（パフォーマンスは若干低下しますが、動作は安定します）")
            else:
                print("[INFO] XLAを無効化しました（cuDNNアルゴリズム選択の高速化とlibdeviceエラーを防ぐため）")
                print("[INFO] 通常のTensorFlow実行を使用します（最初のエポックが高速になります）")
            
            # さらに積極的な最適化（XLAが無効でも有効）
            optimizer_options = {
                'disable_meta_optimizer': False,
                'disable_model_pruning': False,
            }
            
            # WSL2環境ではXLAオートチューニングを完全に無効化
            if is_wsl2:
                optimizer_options['min_graph_nodes'] = -1  # オートチューニングを無効化
            
            tf.config.optimizer.set_experimental_options(optimizer_options)
        except Exception as e:
            print(f"[WARN] XLA設定エラー: {e}")
            # エラーが発生した場合はXLAを無効化
            try:
                tf.config.optimizer.set_jit(False)
                os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
                print("[INFO] XLAを無効化しました（エラーのため）")
            except:
                pass
        # CPUスレッドを最大限活用（100%使用率を目指ム）
        try:
            cpu_count = os.cpu_count() or 8
            # 最大限の並列化（100%使用率を目指ム）
            tf.config.threading.set_intra_op_parallelism_threads(cpu_count * 2)  # 論理コア数の2倍で最大限使用
            tf.config.threading.set_inter_op_parallelism_threads(cpu_count)  # 全コアを使用
            print(f"CPU threading optimized for 100%: intra_op={cpu_count * 2}, inter_op={cpu_count}")
        except Exception:
            pass

        # GPU強制検出テ有効化（最大限活用）
        self.use_gpu = False
        try:
            # WSL2環境ではXLAオートチューニングを無効化
            is_wsl2 = False
            try:
                with open('/proc/version', 'r') as f:
                    version_info = f.read()
                    if 'microsoft' in version_info.lower() or 'wsl' in version_info.lower():
                        is_wsl2 = True
            except:
                pass
            
            if is_wsl2:
                # XLAオートチューニングを無効化
                try:
                    tf.config.optimizer.set_experimental_options({
                        'disable_meta_optimizer': False,
                        'disable_model_pruning': False,
                        'min_graph_nodes': -1,  # オートチューニングを無効化
                    })
                except:
                    pass
                # XLAサービスを無効化
                try:
                    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
                except:
                    pass
            
            gpus = tf.config.list_physical_devices('GPU')
            dml_devices = tf.config.list_physical_devices('DML')  # DirectML (Windows GPU)
            all_gpus = gpus + dml_devices
            
            if all_gpus:
                for gpu in all_gpus:
                    try:
                        # メモリ増加を許可（最大限使用）
                        tf.config.experimental.set_memory_growth(gpu, True)
                        # GPUメモリを最大限使用（メモリ制限を設定せず、全メモリを使用）
                        # memory_growth=Trueで全メモリを動的に使用
                        try:
                            # GPUの全メモリを積極的に使用（必要に応じて）
                            # DirectMLの場合は設定できないことがあるので例外処理
                            pass
                        except Exception:
                            pass
                    except Exception:
                        pass
                # 強制GPU使用
                self.use_gpu = True
                print(f"GPU/DirectML detected: {len(all_gpus)} device(s) - Force GPU usage enabled")
                print(f"GPU memory growth enabled for maximum utilization")
            elif gpus:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception:
                        pass
                self.use_gpu = True
                print(f"GPU detected: {len(gpus)} device(s) - Force GPU usage enabled")
            else:
                print("No GPU detected. Running on CPU.")
        except Exception as e:
            print(f"GPU detection warning: {e}")
        
        # SPARSE MODELING parameters（システム設定に基づいて調整）
        force_accuracy = os.environ.get('FORCE_ACCURACY') == '1'
        
        # フル活用: workers/queue/batchを最大化（Windows対応）
        # 最大限の性能を追求
        if force_accuracy:
            if os.name == 'nt':  # Windows
                self.workers = 1  # Windowsではworkers=1で安全（multiprocessing問題回避）
                self.max_queue_size = 100  # キューサイズを最大限に（バッファリング最大化、100%使用率を目指ム）
                self.use_multiprocessing = False  # WindowsではFalse必須
                # WindowsでもCPU使用率を最大限に上げるため、batch_sizeをさらに大きく
                # CPU使用率を上げるために、データ処理負荷を最大化
                # ただし、大きムぎるとメモリエラーになるので、段階的に試行
                cpu_target_batch = max(self.batch_size, 256)  # CPU処理を増やムため大きく
                if self.use_gpu:
                    # GPU使用時はGPUメモリを考慮してバッチサイズを設定
                    # 環境変数で指定されたバッチサイズを優先（既に設定済み）
                    # VRAMが12GB程度なら、バッチサイズは256程度が安全な上限
                    # 環境変数で指定されていない場合のみ調整
                    if 'TRAINING_BATCH_SIZE' not in os.environ:
                        # メモリエラーを避けるため、控えめに設定
                        self.batch_size = min(max(self.batch_size, 128), 256)  # 128-256の範囲に制限
                        print(f"GPU batch_size set to: {self.batch_size} (safe for 12GB VRAM)")
                    else:
                        # 環境変数で指定された場合は、そのまま使用（ただし警告を表示）
                        if self.batch_size > 256:
                            print(f"[WARN] バッチサイズ {self.batch_size} は大きすぎる可能性があります")
                            print(f"[WARN] GPUメモリ不足が発生する場合は、バッチサイズを減らしてください")
                        else:
                            print(f"GPU batch_size: {self.batch_size} (環境変数から取得)")
                    # 混合精度トレーニングを有効化してGPU効率を最大化
                    self.use_mixed_precision = True
                else:
                    # CPUのみの場合でもメモリが許ム限り最大に（CPU使用率を上げる）
                    self.batch_size = max(self.batch_size, cpu_target_batch)
            else:
                self.workers = os.cpu_count() or 16
                self.max_queue_size = 64  # さらに大きく
                self.use_multiprocessing = True
            # GPUがあればバッチサイズを適切に設定（メモリを考慮）
            if self.use_gpu:
                # 環境変数で指定されたバッチサイズを優先
                if 'TRAINING_BATCH_SIZE' not in os.environ:
                    # 12GB VRAMでは256が安全な上限
                    self.batch_size = min(max(self.batch_size, 128), 256)  # 128-256の範囲に制限
                    print(f"GPU optimization: batch_size={self.batch_size}, mixed_precision=True")
                    print(f"[INFO] バッチサイズはGPUメモリに合わせて調整されました")
                else:
                    # 環境変数で指定された場合はそのまま使用
                    print(f"GPU optimization: batch_size={self.batch_size} (環境変数から取得), mixed_precision=True")
                # GPU使用時は混合精度を強制有効化
                self.use_mixed_precision = True
        augmentation_intensity = 'heavy' if force_accuracy else self.system_config.get('augmentation_intensity', 'medium')
        
        if augmentation_intensity == 'light':
            # 軽量: 少ないデータ拡張
            sparse_augmentation_params = {
                'rotation_range': 15,
                'width_shift_range': 0.1,
                'height_shift_range': 0.1,
                'shear_range': 0.1,
                'zoom_range': 0.1,
                'horizontal_flip': True,
                'vertical_flip': False,
                'brightness_range': [0.9, 1.1],
                'channel_shift_range': 0.05,
                'fill_mode': 'nearest',
                'cval': 0.0,
            }
        elif augmentation_intensity == 'heavy':
            # 重い: 強いデータ拡張
            sparse_augmentation_params = {
                'rotation_range': 45,
                'width_shift_range': 0.3,
                'height_shift_range': 0.3,
                'shear_range': 0.3,
                'zoom_range': 0.3,
                'horizontal_flip': True,
                'vertical_flip': True,
                'brightness_range': [0.7, 1.3],
                'channel_shift_range': 0.2,
                'fill_mode': 'nearest',
                'cval': 0.0,
            }
        else:  # medium
            # 中程度: デフォルト設定
            sparse_augmentation_params = {
                'rotation_range': 30,
                'width_shift_range': 0.2,
                'height_shift_range': 0.2,
                'shear_range': 0.2,
                'zoom_range': 0.2,
                'horizontal_flip': True,
                'vertical_flip': False,
                'brightness_range': [0.8, 1.2],
                'channel_shift_range': 0.1,
                'fill_mode': 'nearest',
                'cval': 0.0,
            }
        
        self.sparse_augmentation_params = sparse_augmentation_params
        
        # SPARSE MODELING regularization parameters
        self.sparse_regularization = {
            'l1_lambda': 0.001,  # L1 regularization strength
            'l2_lambda': 0.0001,  # L2 regularization strength
            'dropout_rate': 0.5,  # High dropout for sparsity
            'sparse_threshold': 0.1,  # Threshold for sparse activation
        }
        # 高精度モード: 長期学習テ耐性強化テローダ強化
        if force_accuracy:
            self.max_epochs = max(self.max_epochs, 300)
            self.patience = max(self.patience, 50)
            # Windowsでのmultiprocessing問題を回避
            if os.name != 'nt':  # Windows以外
                self.workers = max(self.workers, (os.cpu_count() or 8))
                self.max_queue_size = max(self.max_queue_size, 32)
            # Windowsの場合は既に設定されているので変更不要
        
        # WSL2環境を検出（混合精度無効化のため）
        is_wsl2_for_precision = False
        try:
            if os.path.exists('/proc/version'):
                with open('/proc/version', 'r') as f:
                    version_info = f.read()
                    if 'microsoft' in version_info.lower() or 'wsl' in version_info.lower():
                        is_wsl2_for_precision = True
            if 'WSL_DISTRO_NAME' in os.environ or 'WSL_INTEROP' in os.environ:
                is_wsl2_for_precision = True
            if os.path.exists('/mnt/c'):
                is_wsl2_for_precision = True
        except:
            pass
        
        # 混合精度トレーニング設定（WSL2環境では無効化してXLAエラーを回避）
        if is_wsl2_for_precision and self.use_gpu:
            # WSL2 GPU環境では混合精度を無効化（XLAエラー回避）
            self.use_mixed_precision = False
            print("[WSL2] 混合精度トレーニングを無効化しました（XLAエラー回避のため）")
            print("[WSL2] float32で学習します（パフォーマンスは若干低下しますが、安定します）")
        elif self.use_mixed_precision or (self.use_gpu and force_accuracy):
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("Mixed precision training enabled (maximum GPU performance)")
                # GPU効率を最大化するための追加設定
                if self.use_gpu:
                    try:
                        # データ型変換の最適化
                        tf.config.experimental.enable_tensor_float_32_execution(True)  # TensorFloat-32を有効化（A100以降）
                    except:
                        pass
            except Exception as e:
                print(f"Warning: Mixed precision setup failed: {e}")

        # トレーニング用デバイス（条件付きでGPUを強制）
        self.training_device = None

    def log_runtime_devices(self):
        """起動時にGPU/CPUデバイス状況をログ出力。"""
        try:
            print("\n[Device Info]")
            print(f"TensorFlow: {tf.__version__}")
            cpus = tf.config.list_physical_devices('CPU')
            gpus = tf.config.list_physical_devices('GPU')
            print(f"CPU: {len(cpus)} device(s) / GPU: {len(gpus)} device(s)")
            if gpus:
                try:
                    details = [tf.config.experimental.get_device_details(g) for g in gpus]
                    names = [d.get('device_name','GPU') for d in details]
                    print("GPU Name(s): ", ', '.join(names))
                except Exception:
                    pass
            else:
                print("No GPU detected (CPU execution)")
        except Exception:
            pass

    def request_gpu_permission(self):
        """ユーザーにGPU使用の許可と選択を求める。環境や非対話時は環境変数で制御。
        優先順位: GPU_USE(1/0) > 対話入力 > 既定(自動判定)
        許可=1のとき FORCE_GPU=1 を設定。
        """
        try:
            # GPUがなければ何もしない
            if not tf.config.list_physical_devices('GPU'):
                return
            # 環境変数で明示指定がある場合
            env_choice = os.environ.get('GPU_USE')
            if env_choice in ('1', '0'):
                if env_choice == '1':
                    os.environ['FORCE_GPU'] = '1'
                    print("GPU_USE=1: GPU usage enabled (FORCE_GPU=1).")
                else:
                    os.environ.pop('FORCE_GPU', None)
                    print("GPU_USE=0: CPU priority (no forced GPU).")
                return
            # 対話可能なら確認
            if sys.stdin and sys.stdin.isatty():
                print("\nGPU available. Use GPU for training? [y/N] ", end='', flush=True)
                try:
                    answer = input().strip().lower()
                except Exception:
                    answer = ''
                if answer in ('y', 'yes', '1'):
                    os.environ['FORCE_GPU'] = '1'
                    print("User approved: Using GPU (FORCE_GPU=1).")
                else:
                    os.environ.pop('FORCE_GPU', None)
                    print("User selected: CPU priority (no forced GPU).")
            else:
                # Non-interactive: default to auto-detect (don't set FORCE)
                print("Non-interactive mode: Skipping GPU dialog (auto-detect).")
        except Exception:
            pass

    def decide_training_device(self):
        """CPU高負荷かつGPU低負荷ならGPUに強制配置（品質影響なし）。"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                self.training_device = None
                return
            # 環境変数で強制指定された場合は最優先
            if os.environ.get('FORCE_GPU') == '1':
                self.training_device = '/GPU:0'
                print("FORCE_GPU=1: Forcing GPU usage.")
                return
            cpu_busy = False
            gpu_idle = True
            if HAS_PSUTIL:
                cpu_percent = psutil.cpu_percent(interval=0.5)
                cpu_busy = cpu_percent is not None and cpu_percent >= 75.0
            if HAS_NVML:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = getattr(util, 'gpu', 0)
                    gpu_idle = gpu_util < 20
                except Exception:
                    gpu_idle = True
            if cpu_busy and gpu_idle:
                self.training_device = '/GPU:0'
                print("Condition met: CPU high load + GPU idle -> Forcing GPU placement.")
            else:
                self.training_device = None
        except Exception:
            self.training_device = None

    def start_status_viewer(self):
        """学習中ステータスビューア（Qt）を自動起動。既に起動済みなら何もしない。"""
        try:
            if self.viewer_process and self.viewer_process.poll() is None:
                return
            # デフォルトはPyQt版。環境変数でtkに切替可能
            viewer_file = 'qt_status_viewer.py'
            if os.environ.get('STATUS_VIEWER_BACKEND') == 'tk':
                viewer_file = 'status_viewer_app.py'
            viewer_path = Path(__file__).resolve().parents[1] / 'dashboard' / viewer_file
            if not viewer_path.exists():
                return
            kwargs = {}
            if os.name == 'nt':
                CREATE_NO_WINDOW = 0x08000000
                kwargs['creationflags'] = CREATE_NO_WINDOW
            self.viewer_process = subprocess.Popen([sys.executable, str(viewer_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
        except Exception:
            pass

    def stop_status_viewer(self):
        """学習終了時にビューアをクリーン停止。"""
        try:
            if self.viewer_process and self.viewer_process.poll() is None:
                self.viewer_process.terminate()
        except Exception:
            pass

    def update_status(self, payload):
        """チャット表示用に進捗をJSONへ書き出ム。"""
        try:
            payload = dict(payload)
            # 使用率を付加（毎回更新して最新の値を取得）
            try:
                # より頻繁に更新して、GPU使用率をリアルタイムに取得
                self._last_metrics = self.sample_system_metrics()
                self._last_metrics_time = time.time()
                if self._last_metrics:
                    payload.update(self._last_metrics)
            except Exception:
                pass
            payload['timestamp'] = time.time()
            with open(self.status_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            # 書き出し失敗は学習を止めない
            pass

    def sample_system_metrics(self):
        """CPU/メモリ/GPU使用率を取得（HWiNFOがなくてもpsutil/pynvmlで代替）。"""
        metrics = {}
        # まずHWiNFO Shared Memoryからすべての情報を取得を試みる（最も正確）
        hwinfo_data = None
        if HAS_HWINFO_READER:
            try:
                hwinfo_data = read_hwinfo_shared_memory()
            except Exception:
                pass
        
        # CPU/メモリ
        try:
            # HWiNFOから取得した値を優先
            if hwinfo_data and hwinfo_data.get('cpu_percent') is not None:
                metrics['cpu_percent'] = float(hwinfo_data['cpu_percent'])
            elif HAS_PSUTIL:
                metrics['cpu_percent'] = float(psutil.cpu_percent(interval=0.0))
            else:
                metrics['cpu_percent'] = None
            
            # メモリ
            if hwinfo_data and hwinfo_data.get('mem_percent') is not None:
                metrics['mem_percent'] = float(hwinfo_data['mem_percent'])
                metrics['mem_used_mb'] = float(hwinfo_data.get('mem_used_mb', 0))
                metrics['mem_total_mb'] = float(hwinfo_data.get('mem_total_mb', 0))
            elif HAS_PSUTIL:
                vm = psutil.virtual_memory()
                metrics['mem_percent'] = float(vm.percent)
                metrics['mem_used_mb'] = round(vm.used / (1024*1024), 1)
                metrics['mem_total_mb'] = round(vm.total / (1024*1024), 1)
            else:
                metrics['mem_percent'] = None
                metrics['mem_used_mb'] = None
                metrics['mem_total_mb'] = None
            
            # CPU温度
            if hwinfo_data and hwinfo_data.get('cpu_temp_c') is not None:
                metrics['cpu_temp_c'] = float(hwinfo_data['cpu_temp_c'])
            elif HAS_PSUTIL:
                try:
                    temps = psutil.sensors_temperatures()
                    cpu_t = None
                    for key, arr in (temps or {}).items():
                        if not arr:
                            continue
                        # 一般的なキー名の候補
                        if key.lower() in ('coretemp', 'acpitz', 'k10temp', 'cpu-thermal', 'amdppm', 'intelpowergadget'):
                            cpu_t = arr[0].current if hasattr(arr[0], 'current') else None
                            break
                    metrics['cpu_temp_c'] = float(cpu_t) if cpu_t is not None else None
                except Exception:
                    metrics['cpu_temp_c'] = None
            else:
                metrics['cpu_temp_c'] = None
        except Exception:
            metrics['cpu_percent'] = None
            metrics['mem_percent'] = None
            metrics['mem_used_mb'] = None
            metrics['mem_total_mb'] = None
            metrics['cpu_temp_c'] = None
        
        # GPU
        try:
            gpu_util = None
            gpu_mem_used = None
            gpu_mem_total = None
            gpu_temp = None
            gpu_power = None
            
            # HWiNFOからGPU情報を取得（最優先テ最も信頼できる情報源）
            # HWiNFOの値は最も信頼できるため、取得できた値は必ず使用（上書きしない）
            if hwinfo_data:
                # HWiNFOから取得できた値は必ず使用（nvidia-smiより優先）
                # 値の検証（異常値チェック）
                if hwinfo_data.get('gpu_util_percent') is not None:
                    val = float(hwinfo_data['gpu_util_percent'])
                    if 0 <= val <= 100:  # 正常な範囲内のみ使用
                        gpu_util = val
                if hwinfo_data.get('gpu_temp_c') is not None:
                    val = float(hwinfo_data['gpu_temp_c'])
                    if 20 <= val <= 150:  # 正常な範囲内のみ使用
                        gpu_temp = val
                if hwinfo_data.get('gpu_mem_used_mb') is not None:
                    val = float(hwinfo_data['gpu_mem_used_mb'])
                    if 0 <= val <= 100000:  # 正常な範囲内のみ使用
                        gpu_mem_used = val
                if hwinfo_data.get('gpu_mem_total_mb') is not None:
                    val = float(hwinfo_data['gpu_mem_total_mb'])
                    if 0 <= val <= 100000:  # 正常な範囲内のみ使用
                        gpu_mem_total = val
                if hwinfo_data.get('gpu_power_w') is not None:
                    val = float(hwinfo_data['gpu_power_w'])
                    if 0 <= val <= 1000:  # 正常な範囲内のみ使用
                        gpu_power = val
            
            # HWiNFOで取得できなかった場合のみ、nvidia-smiで取得を試みる（補完用）
            # HWiNFOは最も信頼できる情報源のため、HWiNFOから取得できた値は上書きしない
            if gpu_util is None:
                try:
                    import subprocess
                    # nvidia-smiでGPU使用率を取得（最新値を取得するためキャッシュを無効化）
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits', '-i', '0'],
                        capture_output=True,
                        text=True,
                        timeout=2,
                        shell=False  # シェル経由でない直接実行
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        # 出力をパース（改行と空白を考慮）
                        output = result.stdout.strip()
                        # 最初の行を取得
                        lines = output.split('\n')
                        if lines:
                            # カンマで分割し、空白を除去
                            parts = [p.strip() for p in lines[0].split(',')]
                            
                            if len(parts) >= 1 and parts[0]:
                                try:
                                    gpu_util = float(parts[0])
                                    # 値の妥当性チェック（0-100の範囲）
                                    if gpu_util < 0:
                                        gpu_util = 0.0
                                    elif gpu_util > 100:
                                        gpu_util = 100.0
                                except (ValueError, TypeError):
                                    gpu_util = None
                            if len(parts) >= 2 and parts[1]:
                                try:
                                    gpu_mem_used = float(parts[1])
                                except (ValueError, TypeError):
                                    gpu_mem_used = None
                            if len(parts) >= 3 and parts[2]:
                                try:
                                    gpu_mem_total = float(parts[2])
                                except (ValueError, TypeError):
                                    gpu_mem_total = None
                except Exception as e:
                    pass
            
            # nvidia-smiで取得できなかった場合、NVMLで試み（NVIDIA GPUのバックアップ）
            if (gpu_util is None or gpu_temp is None or gpu_power is None) and HAS_NVML:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    # GPU使用率
                    if gpu_util is None:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = float(getattr(util, 'gpu', 0))
                    # GPUメモリ
                    if gpu_mem_used is None or gpu_mem_total is None:
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        if gpu_mem_used is None:
                            gpu_mem_used = round(mem_info.used / (1024*1024), 1)
                        if gpu_mem_total is None:
                            gpu_mem_total = round(mem_info.total / (1024*1024), 1)
                    # GPU温度
                    if gpu_temp is None:
                        try:
                            tempC = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            gpu_temp = float(tempC) if tempC is not None else None
                        except Exception:
                            pass
                    # GPU電力
                    if gpu_power is None:
                        try:
                            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                            gpu_power = round(power_mw / 1000.0, 1)
                        except Exception:
                            pass
                except Exception:
                    pass
            
            # それでも取得できなかった場合はNoneのまま（誤った値を表示しない）
            if gpu_util is None:
                gpu_mem_used = None if gpu_mem_used is None else gpu_mem_used
                gpu_mem_total = None if gpu_mem_total is None else gpu_mem_total
            
            metrics['gpu_util_percent'] = gpu_util
            metrics['gpu_mem_used_mb'] = gpu_mem_used
            metrics['gpu_mem_total_mb'] = gpu_mem_total
            metrics['gpu_temp_c'] = gpu_temp
            metrics['gpu_power_w'] = gpu_power
        except Exception:
            metrics['gpu_util_percent'] = None
            metrics['gpu_mem_used_mb'] = None
            metrics['gpu_mem_total_mb'] = None
            metrics['gpu_temp_c'] = None
            metrics['gpu_power_w'] = None
        return metrics
        
    def format_time(self, seconds):
        """見やすく安定した時間表記に変換（変動しすぎない丸め表現）。"""
        if seconds is None:
            return "-"
        try:
            seconds = float(seconds)
        except Exception:
            return "-"
        if seconds >= 2 * 3600:
            hours = int(round(seconds / 3600.0))
            return f"約{hours}時間"
        if seconds >= 10 * 60:
            mins = int(round(seconds / 60.0))
            return f"約{mins}分"
        if seconds >= 60:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}分{secs}秒"
        return f"{int(seconds)}秒"

    def format_eta(self, start_time, progress_percent):
        """進捗率から残り時間と完了予測を安定表記で返す。"""
        if not start_time or not progress_percent or progress_percent <= 0:
            return ("-", "-")
        elapsed = time.time() - start_time
        remaining = (elapsed / progress_percent) * (100 - progress_percent)
        return (self.format_time(remaining), self.format_time(elapsed + remaining))
        
    def print_progress_with_time(self, current, total, prefix="Progress", suffix="", length=50, 
                                 start_time=None, section_start_time=None, section_name="",
                                 overall_total=None, overall_current=None, overall_start_time=None):
        """進行度、残り時間、経過時間を表示するプログレスバー（全体と項目の両方を表示）"""
        if total == 0:
            return
            
        # 現在の項目の進捗
        item_percent = 100 * (current / float(total))
        filled_length = int(length * current // total)
        bar = '=' * filled_length + '-' * (length - filled_length)
        
        # 時間情報の計算（項目単位）
        item_elapsed_str = ""
        item_remaining_str = ""
        
        if section_start_time is not None and current > 0:
            item_elapsed = time.time() - section_start_time
            item_elapsed_str = f"項目経過: {self.format_time(item_elapsed)}"
            
            # 項目の残り時間の推定
            if current > 0:
                avg_time_per_item = item_elapsed / current
                remaining_items = total - current
                remaining_seconds = avg_time_per_item * remaining_items
                item_remaining_str = f"項目残り: {self.format_time(remaining_seconds)}"
        
        # 全体の進捗（指定されている場合）
        overall_info = ""
        if overall_total is not None and overall_current is not None and overall_start_time is not None:
            overall_percent = 100 * (overall_current / float(overall_total))
            overall_elapsed = time.time() - overall_start_time
            overall_elapsed_str = f"全体経過: {self.format_time(overall_elapsed)}"
            
            # 全体の残り時間の推定
            if overall_current > 0:
                avg_time_per_overall_item = overall_elapsed / overall_current
                remaining_overall_items = overall_total - overall_current
                overall_remaining_seconds = avg_time_per_overall_item * remaining_overall_items
                overall_remaining_str = f"全体残り: {self.format_time(overall_remaining_seconds)}"
                overall_info = f"【全体: {overall_percent:.1f}% | {overall_elapsed_str} | {overall_remaining_str}】"
        
        # 表示
        progress_info = f"{prefix}: |{bar}| 項目: {item_percent:.1f}%"
        if item_elapsed_str:
            progress_info += f" | {item_elapsed_str}"
        if item_remaining_str:
            progress_info += f" | {item_remaining_str}"
        if overall_info:
            progress_info += f" | {overall_info}"
        if suffix:
            progress_info += f" - {suffix}"
            
        print(f'\r{progress_info}', end='', flush=True)
        # JSONステータス更新
        try:
            item_elapsed_seconds = (time.time() - section_start_time) if section_start_time else None
            item_remaining_seconds = remaining_seconds if 'remaining_seconds' in locals() else None
            status = {
                'stage': prefix,
                'section': section_name,
                'item_progress_percent': round(item_percent, 1),
                'item_elapsed_seconds': item_elapsed_seconds,
                'item_elapsed_human': self.format_time(item_elapsed_seconds),
                'item_remaining_est_seconds': item_remaining_seconds,
                'item_remaining_human': self.format_time(item_remaining_seconds),
            }
            if overall_total is not None and overall_current is not None and overall_start_time is not None:
                overall_percent = 100 * (overall_current / float(overall_total))
                overall_elapsed = time.time() - overall_start_time
                # 残りと完了予測（安定表記）
                overall_remaining_human, overall_eta_human = self.format_eta(overall_start_time, overall_percent)
                status.update({
                    'overall_progress_percent': round(overall_percent, 1),
                    'overall_elapsed_seconds': overall_elapsed,
                    'overall_elapsed_human': self.format_time(overall_elapsed),
                    'overall_remaining_est_seconds': overall_remaining_seconds if 'overall_remaining_seconds' in locals() else None,
                    'overall_remaining_human': overall_remaining_human,
                    'overall_eta_human': overall_eta_human,
                })
            if suffix:
                status['message'] = suffix
            self.update_status(status)
        except Exception:
            pass
        if current == total:
            print()  # New line when complete
    
    def print_progress(self, current, total, prefix="Progress", suffix="", length=50):
        """後方互換性のための簡易プログレスバー"""
        self.print_progress_with_time(current, total, prefix, suffix, length, 
                                     self.section_start_time, self.section_start_time, self.section_name)
        
    def start_section(self, section_name):
        """新しいセクションを開始"""
        self.section_name = section_name
        self.section_start_time = time.time()
        print(f"\n[{section_name} Started]")
        
    def end_section(self, section_name):
        """セクションを終了"""
        if self.section_start_time:
            elapsed = time.time() - self.section_start_time
            print(f"[{section_name} Completed] Time: {self.format_time(elapsed)}")
            self.section_start_time = None
        
    def load_and_prepare_data(self):
        """Load data with SPARSE MODELING for 4 classes (with caching)"""
        # キャッシュをチェック（同じデータパスで既に読み込まれていれば再利用）
        if (ClearProgressSparseModelingFourClassWasherInspector._cached_data is not None and 
            ClearProgressSparseModelingFourClassWasherInspector._cached_data_path == self.data_path):
            print("\n[INFO] データは既に読み込まれています。キャッシュから再利用します。")
            print("=" * 80)
            X, y, class_counts = ClearProgressSparseModelingFourClassWasherInspector._cached_data
            print(f"[INFO] キャッシュから復元: {len(X)}枚の画像, クラス分布: {class_counts}")
            return X, y, class_counts
        
        # データを読み込む
        self.total_start_time = time.time()
        self.start_section("Data Loading")
        print("=" * 80)
        print("Loading data with new defect samples")
        print("=" * 80)
        
        images = []
        labels = []
        class_counts = {}
        
        total_classes = len(self.class_names)
        class_start_time = time.time()
        
        for class_idx, class_name in enumerate(self.class_names):
            class_section_start = time.time()
            
            # 全体進捗の計算（データ読み込みは全体の10%と仮定）
            overall_data_step = 10  # データ読み込みは10%
            overall_current = int((class_idx / total_classes) * overall_data_step)
            overall_total = 100  # 全体を100%とする
            
            self.print_progress_with_time(class_idx, total_classes, "Loading Classes", 
                                         f"Processing: {class_name}", 
                                         start_time=self.section_start_time,
                                         section_start_time=class_section_start,
                                         section_name=class_name,
                                         overall_current=overall_current,
                                         overall_total=overall_total,
                                         overall_start_time=self.total_start_time)
            
            class_path = self.data_path / 'resin' / class_name
            if not class_path.exists():
                print(f"\nWarning: {class_path} does not exist")
                continue
                
            class_images = []
            print(f"\n[{class_name}] Loading images from: {class_path}")
            
            # Use glob to find all image files recursively
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(class_path.rglob(ext))
            
            print(f"[{class_name}] Found {len(image_files)} image files")
            
            total_files = len(image_files)
            file_start_time = time.time()
            for file_idx, img_file in enumerate(image_files):
                if file_idx % 5 == 0 or file_idx == total_files - 1:  # Update every 5 files or last file
                    # ファイル読み込みの進捗計算
                    file_progress = (file_idx + 1) / total_files
                    class_progress = (class_idx / total_classes) + (file_progress / total_classes)
                    overall_current = int(class_progress * overall_data_step)
                    
                    self.print_progress_with_time(file_idx + 1, total_files, 
                                                 f"{class_name} Loading", 
                                                 f"{file_idx + 1}/{total_files} images",
                                                 start_time=file_start_time,
                                                 section_start_time=self.section_start_time,
                                                 section_name="Data Loading",
                                                 overall_current=overall_current,
                                                 overall_total=overall_total,
                                                 overall_start_time=self.total_start_time)
                
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        class_images.append(img)
                    else:
                        print(f"\nWarning: Failed to load {img_file}")
                except Exception as e:
                    print(f"\nError: Error loading {img_file}: {e}")
            
            class_elapsed = time.time() - class_section_start
            print(f"\n[{class_name}] Completed: Loaded {len(class_images)} images (Time: {self.format_time(class_elapsed)})")
            
            images.extend(class_images)
            labels.extend([class_idx] * len(class_images))
            class_counts[class_name] = len(class_images)
        
        # Data loading completed (10% of total)
        self.print_progress_with_time(total_classes, total_classes, "Loading Classes", "Completed",
                                     start_time=self.section_start_time,
                                     section_start_time=self.section_start_time,
                                     section_name="Data Loading",
                                     overall_current=overall_data_step,
                                     overall_total=100,
                                     overall_start_time=self.total_start_time)
        
        if not images:
            raise ValueError("No images found!")
        
        # Convert to arrays
        print("\nConverting data to arrays...")
        conversion_start = time.time()
        X = np.array(images)
        y = np.array(labels)
        conversion_time = time.time() - conversion_start
        print(f"Conversion completed (Time: {self.format_time(conversion_time)})")
        
        print(f"\nTotal images: {len(X)}")
        print(f"Class distribution: {class_counts}")
        
        # Verify expected counts
        expected_counts = {'good': 1144, 'black_spot': 88, 'chipping': 117, 'scratch': 112}
        print("\nData verification:")
        for class_name, expected_count in expected_counts.items():
            actual_count = class_counts.get(class_name, 0)
            if actual_count != expected_count:
                print(f"Warning: {class_name} expected {expected_count}, got {actual_count}")
            else:
                print(f"OK {class_name}: {actual_count} images (correct)")
        
        self.end_section("Data Loading")
        
        # データをキャッシュに保存（次回の再利用のため）
        ClearProgressSparseModelingFourClassWasherInspector._cached_data = (X, y, class_counts)
        ClearProgressSparseModelingFourClassWasherInspector._cached_data_path = self.data_path
        print(f"[INFO] データをキャッシュに保存しました（次回の再利用のため）")
        
        return X, y, class_counts
    
    def create_sparse_data_generators(self, X, y):
        """Create data generators with SPARSE MODELING"""
        print("\n" + "=" * 60)
        print("CREATING DATA GENERATORS WITH SPARSE MODELING")
        print("=" * 60)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        
        # SPARSE MODELING training data generator
        train_datagen = ImageDataGenerator(
            **self.sparse_augmentation_params,
            rescale=1./255,
        )
        
        # Validation data generator (minimal augmentation)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
        )
        
        return train_datagen, val_datagen, X_train, X_val, y_train, y_val
    
    def build_sparse_ensemble_models(self, num_classes=4):
        """Build ensemble of SPARSE MODELING EfficientNet models for 4 classes"""
        print("\n" + "=" * 60)
        print("BUILDING SPARSE MODELING ENSEMBLE MODELS")
        print("=" * 60)
        
        models_config = [
            {'name': 'EfficientNetB0', 'model': EfficientNetB0, 'input_size': (224, 224, 3)},
            {'name': 'EfficientNetB1', 'model': EfficientNetB1, 'input_size': (240, 240, 3)},
            {'name': 'EfficientNetB2', 'model': EfficientNetB2, 'input_size': (260, 260, 3)},
        ]
        
        ensemble_models = []
        
        for i, config in enumerate(models_config):
            model_name = config['name']  # モデル名を取得
            self.print_progress(i, len(models_config), "Building Models", f"Building {model_name}")
            
            print(f"\nBuilding SPARSE MODELING {model_name} for 4-class classification...")
            
            # Base model
            base_model = config['model'](
                weights=None,  # Don't load pretrained weights
                include_top=False,
                input_shape=config['input_size']
            )
            
            # Freeze initial layers（高精度モードではより多くの層を学習に解放）
            if os.environ.get('FORCE_ACCURACY') == '1':
                # ほぼ全層を学習対象（最初の数層のみ固定）
                for layer in base_model.layers[:5]:
                    layer.trainable = False
            else:
                for layer in base_model.layers[:-30]:
                    layer.trainable = False
            
            # Add SPARSE MODELING layers for 4-class classification
            # GPUデバイスに強制的に配置（モデル構築時から）
            if self.use_gpu:
                try:
                    dml_devices = tf.config.list_physical_devices('DML')
                    gpus = tf.config.list_physical_devices('GPU')
                    if dml_devices:
                        with tf.device('/DML:0'):
                            model = models.Sequential([
                                base_model,
                                layers.GlobalAveragePooling2D(),
                                layers.Dense(1024, activation='relu',
                                           kernel_regularizer=regularizers.l1_l2(
                                               l1=self.sparse_regularization['l1_lambda'],
                                               l2=self.sparse_regularization['l2_lambda']
                                           )),
                                layers.BatchNormalization(),
                                layers.Dropout(self.sparse_regularization['dropout_rate']),
                                layers.Dense(512, activation='relu',
                                           kernel_regularizer=regularizers.l1_l2(
                                               l1=self.sparse_regularization['l1_lambda'],
                                               l2=self.sparse_regularization['l2_lambda']
                                           )),
                                layers.BatchNormalization(),
                                layers.Dropout(self.sparse_regularization['dropout_rate']),
                                layers.Dense(256, activation='relu',
                                           kernel_regularizer=regularizers.l1_l2(
                                               l1=self.sparse_regularization['l1_lambda'],
                                               l2=self.sparse_regularization['l2_lambda']
                                           )),
                                layers.BatchNormalization(),
                                layers.Dropout(self.sparse_regularization['dropout_rate']),
                                layers.Dense(num_classes, activation='softmax',
                                           kernel_regularizer=regularizers.l1_l2(
                                               l1=self.sparse_regularization['l1_lambda'],
                                               l2=self.sparse_regularization['l2_lambda']
                                           ))
                            ])
                            print(f"[{config['name']}] Model built on DirectML GPU")
                    elif gpus:
                        with tf.device('/GPU:0'):
                            model = models.Sequential([
                                base_model,
                                layers.GlobalAveragePooling2D(),
                                layers.Dense(1024, activation='relu',
                                           kernel_regularizer=regularizers.l1_l2(
                                               l1=self.sparse_regularization['l1_lambda'],
                                               l2=self.sparse_regularization['l2_lambda']
                                           )),
                                layers.BatchNormalization(),
                                layers.Dropout(self.sparse_regularization['dropout_rate']),
                                layers.Dense(512, activation='relu',
                                           kernel_regularizer=regularizers.l1_l2(
                                               l1=self.sparse_regularization['l1_lambda'],
                                               l2=self.sparse_regularization['l2_lambda']
                                           )),
                                layers.BatchNormalization(),
                                layers.Dropout(self.sparse_regularization['dropout_rate']),
                                layers.Dense(256, activation='relu',
                                           kernel_regularizer=regularizers.l1_l2(
                                               l1=self.sparse_regularization['l1_lambda'],
                                               l2=self.sparse_regularization['l2_lambda']
                                           )),
                                layers.BatchNormalization(),
                                layers.Dropout(self.sparse_regularization['dropout_rate']),
                                layers.Dense(num_classes, activation='softmax',
                                           kernel_regularizer=regularizers.l1_l2(
                                               l1=self.sparse_regularization['l1_lambda'],
                                               l2=self.sparse_regularization['l2_lambda']
                                           ))
                            ])
                            print(f"[{config['name']}] Model built on CUDA GPU")
                    else:
                        raise ValueError("No GPU devices available")
                except Exception as e:
                    print(f"[{config['name']}] Warning: Failed to build model on GPU, using CPU: {e}")
                    # フォールバック: CPUで構築
                    model = models.Sequential([
                        base_model,
                        layers.GlobalAveragePooling2D(),
                        layers.Dense(1024, activation='relu',
                                   kernel_regularizer=regularizers.l1_l2(
                                       l1=self.sparse_regularization['l1_lambda'],
                                       l2=self.sparse_regularization['l2_lambda']
                                   )),
                        layers.BatchNormalization(),
                        layers.Dropout(self.sparse_regularization['dropout_rate']),
                        layers.Dense(512, activation='relu',
                                   kernel_regularizer=regularizers.l1_l2(
                                       l1=self.sparse_regularization['l1_lambda'],
                                       l2=self.sparse_regularization['l2_lambda']
                                   )),
                        layers.BatchNormalization(),
                        layers.Dropout(self.sparse_regularization['dropout_rate']),
                        layers.Dense(256, activation='relu',
                                   kernel_regularizer=regularizers.l1_l2(
                                       l1=self.sparse_regularization['l1_lambda'],
                                       l2=self.sparse_regularization['l2_lambda']
                                   )),
                        layers.BatchNormalization(),
                        layers.Dropout(self.sparse_regularization['dropout_rate']),
                        layers.Dense(num_classes, activation='softmax',
                                   kernel_regularizer=regularizers.l1_l2(
                                       l1=self.sparse_regularization['l1_lambda'],
                                       l2=self.sparse_regularization['l2_lambda']
                                   ))
                    ])
            else:
                # CPUモード
                model = models.Sequential([
                    base_model,
                    layers.GlobalAveragePooling2D(),
                    layers.Dense(1024, activation='relu',
                               kernel_regularizer=regularizers.l1_l2(
                                   l1=self.sparse_regularization['l1_lambda'],
                                   l2=self.sparse_regularization['l2_lambda']
                               )),
                    layers.BatchNormalization(),
                    layers.Dropout(self.sparse_regularization['dropout_rate']),
                    layers.Dense(512, activation='relu',
                               kernel_regularizer=regularizers.l1_l2(
                                   l1=self.sparse_regularization['l1_lambda'],
                                   l2=self.sparse_regularization['l2_lambda']
                               )),
                    layers.BatchNormalization(),
                    layers.Dropout(self.sparse_regularization['dropout_rate']),
                    layers.Dense(256, activation='relu',
                               kernel_regularizer=regularizers.l1_l2(
                                   l1=self.sparse_regularization['l1_lambda'],
                                   l2=self.sparse_regularization['l2_lambda']
                               )),
                    layers.BatchNormalization(),
                    layers.Dropout(self.sparse_regularization['dropout_rate']),
                    layers.Dense(num_classes, activation='softmax',
                               kernel_regularizer=regularizers.l1_l2(
                                   l1=self.sparse_regularization['l1_lambda'],
                                   l2=self.sparse_regularization['l2_lambda']
                               ))
                ])
            
            # SPARSE MODELING optimizer with weight decay + CosineDecayRestarts
            lr_schedule = optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=1e-3,
                first_decay_steps=50,
                t_mul=2.0,
                m_mul=0.8,
                alpha=1e-5
            )
            # Use Adam optimizer (Keras 3 compatible)
            optimizer = optimizers.Adam(
                learning_rate=lr_schedule,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )
            
            # model.compile()実行前にXLAを確実に無効化
            try:
                tf.config.optimizer.set_jit(False)
                os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_cpu_global_jit=false'
                os.environ['TF_XLA_AUTO_JIT'] = '0'
                print(f"[{model_name}] model.compile()実行前、XLAを無効化しました")
            except Exception as e:
                print(f"[{model_name}] XLA無効化警告（compile前）: {e}")
            
            # XLAを回避するため、eager実行モードを有効化（WSL2でのlibdevice/ptxasエラー回避）
            # GPU使用時は常にeager実行を使用（XLAエラーを完全に回避）
            use_eagerly = False
            is_wsl2 = False
            use_gpu_for_training = self.use_gpu  # GPU使用フラグを確認
            
            try:
                # WSL2環境を検出（複数の方法で確認）
                if os.path.exists('/proc/version'):
                    with open('/proc/version', 'r') as f:
                        version_info = f.read()
                        if 'microsoft' in version_info.lower() or 'wsl' in version_info.lower():
                            is_wsl2 = True
                # 環境変数でも確認
                if 'WSL_DISTRO_NAME' in os.environ or 'WSL_INTEROP' in os.environ:
                    is_wsl2 = True
                # パスでも確認
                if os.path.exists('/mnt/c'):
                    is_wsl2 = True
            except:
                pass
            
            # GPU使用時またはWSL2環境では常にeager実行を使用（重要：XLAエラーを完全に回避）
            if use_gpu_for_training or is_wsl2:
                use_eagerly = True
                if is_wsl2 and use_gpu_for_training:
                    print(f"[{model_name}] WSL2 GPU環境: eager実行モードを強制有効化します（XLA/libdeviceエラー完全回避）")
                elif use_gpu_for_training:
                    print(f"[{model_name}] GPU使用時: eager実行モードを有効化します（XLAエラー回避）")
                elif is_wsl2:
                    print(f"[{model_name}] WSL2環境: eager実行モードを有効化します")
            else:
                # エラー時も安全のためeager実行を使用
                use_eagerly = True
                print(f"[{model_name}] 安全のためeager実行モードを有効化します")
            
            # GPU使用時またはWSL2環境では常にeager実行を強制（XLAエラー完全回避）
            if use_gpu_for_training or is_wsl2:
                use_eagerly = True
                # 環境変数でもeager実行を強制
                os.environ['TF_EAGER_EXECUTION'] = '1'
                print(f"[{model_name}] eager実行モードを強制有効化: use_eagerly={use_eagerly}")
            
            # XLAを最終的に無効化（compile直前）
            tf.config.optimizer.set_jit(False)
            os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_cpu_global_jit=false'
            os.environ['TF_XLA_AUTO_JIT'] = '0'
            
            # GPU使用時またはWSL2環境では、TensorFlow実行モードもeagerに強制
            if use_gpu_for_training or is_wsl2:
                try:
                    tf.config.run_functions_eagerly(True)
                    # tf.data関数もeager実行を強制（重要：XLAエラー回避）
                    tf.data.experimental.enable_debug_mode()
                    print(f"[{model_name}] TensorFlow実行モードをeagerに強制設定しました")
                    print(f"[{model_name}] tf.data関数もeager実行モードを有効化しました")
                except:
                    pass
            
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
                run_eagerly=use_eagerly  # GPU使用時またはWSL2ではTrue（必須）
            )
            
            # compile()直後にeager実行が有効か確認とXLA無効化
            try:
                # モデルがeager実行モードでコンパイルされたか確認
                if hasattr(model, 'run_eagerly'):
                    actual_eager = model.run_eagerly
                    print(f"[{model_name}] model.compile()完了: run_eagerly={actual_eager} (設定値={use_eagerly})")
                    if not actual_eager:
                        print(f"[{model_name}] ERROR: eager実行モードが有効になっていません！強制的に再設定します")
                        # 強制的にeager実行を有効化（Keras 3の方法）
                        try:
                            # モデルの実行モードを強制的に設定
                            model.run_eagerly = True
                            print(f"[{model_name}] 強制的にrun_eagerly=Trueに設定しました")
                        except:
                            print(f"[{model_name}] run_eagerlyの設定に失敗しました")
                else:
                    print(f"[{model_name}] WARNING: run_eagerly属性が見つかりません")
                
                # XLAを再度無効化（確実に）
                tf.config.optimizer.set_jit(False)
                os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_cpu_global_jit=false'
                os.environ['TF_XLA_AUTO_JIT'] = '0'
                print(f"[{model_name}] XLA無効化確認: JIT={tf.config.optimizer.get_jit()}")
            except Exception as e:
                print(f"[{model_name}] eager実行確認エラー: {e}")
                import traceback
                traceback.print_exc()
            
            # compile()実行後もXLAを無効化
            try:
                tf.config.optimizer.set_jit(False)
            except:
                pass
            
            ensemble_models.append({
                'model': model,
                'name': config['name'],
                'input_size': config['input_size']
            })
        
        self.print_progress_with_time(len(models_config), len(models_config), "Building Models", "Complete",
                                     start_time=self.section_start_time,
                                     section_start_time=self.section_start_time,
                                     section_name="Model Building",
                                     overall_current=15,
                                     overall_total=100,
                                     overall_start_time=self.total_start_time)
        return ensemble_models
    
    def train_sparse_ensemble_with_cross_validation(self, X, y):
        """Train SPARSE MODELING ensemble with cross-validation for 4 classes"""
        self.start_section("アンサンブル学習")
        print("\n" + "=" * 60)
        print("TRAINING SPARSE MODELING ENSEMBLE")
        print("=" * 60)
        
        # Create data generators
        gen_start = time.time()
        print("\nCreating data generators...")
        train_datagen, val_datagen, X_train, X_val, y_train, y_val = self.create_sparse_data_generators(X, y)
        gen_time = time.time() - gen_start
        print(f"Data generators created (Time: {self.format_time(gen_time)})")
        
        # Build ensemble models
        build_start = time.time()
        print("\nBuilding models...")
        ensemble_models = self.build_sparse_ensemble_models()
        build_time = time.time() - build_start
        print(f"Models built (Time: {self.format_time(build_time)})")
        
        # Calculate class weights for imbalanced data
        print("\nCalculating class weights...")
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"\nClass weights: {class_weight_dict}")
        print(f"Sparse modeling parameters: {self.sparse_regularization}")
        
        # Train each model in the ensemble
        total_models = len(ensemble_models)
        training_start_time = time.time()
        
        for i, model_config in enumerate(ensemble_models):
            model_train_start = time.time()
            overall_model_progress = (i / total_models) * 100
            print(f"\n" + "=" * 80)
            print(f"Training {i+1}/{total_models}: {model_config['name']}")
            print(f"[Model Progress: {overall_model_progress:.1f}%]")
            if i > 0:
                model_elapsed = time.time() - training_start_time
                avg_model_time = model_elapsed / i
                remaining_models = total_models - i
                estimated_model_remaining = avg_model_time * remaining_models
                print(f"Model elapsed: {self.format_time(model_elapsed)} | Model remaining: {self.format_time(estimated_model_remaining)}")
            
            # Overall progress (through all processes)
            if self.total_start_time:
                total_elapsed = time.time() - self.total_start_time
                # Model training accounts for about 70% of total
                estimated_total_progress = overall_model_progress * 0.7
                if estimated_total_progress > 0:
                    estimated_total_remaining = (total_elapsed / estimated_total_progress) * (100 - estimated_total_progress)
                else:
                    estimated_total_remaining = 0
                print(f"[Overall Progress: {estimated_total_progress:.1f}%] Overall elapsed: {self.format_time(total_elapsed)} | Overall remaining: {self.format_time(estimated_total_remaining)}")
            print("=" * 80)
            
            model = model_config['model']
            model_name = model_config['name']
            
            # チェックポイントから再開する処理
            resume_mode = os.environ.get('TRAINING_RESUME', '0') == '1'
            initial_epoch = 0
            
            if resume_mode:
                checkpoint_dir = Path('checkpoints') / f'clear_sparse_4class_{model_name.lower()}'
                if checkpoint_dir.exists():
                    checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.h5'))
                    if checkpoints:
                        # 最新のチェックポイントを読み込む
                        latest_checkpoint = checkpoints[-1]
                        try:
                            print(f"[{model_name}] チェックポイントから再開: {latest_checkpoint.name}")
                            # チェックポイントからモデルを読み込む
                            from tensorflow.keras.models import load_model
                            model = load_model(str(latest_checkpoint))
                            # エポック番号を取得（ファイル名から）
                            import re
                            match = re.search(r'checkpoint_epoch_(\d+)\.h5', latest_checkpoint.name)
                            if match:
                                initial_epoch = int(match.group(1))
                                print(f"[{model_name}] エポック {initial_epoch} から再開します")
                            else:
                                print(f"[{model_name}] エポック番号を取得できませんでした。最初から開始します。")
                                initial_epoch = 0
                        except Exception as e:
                            print(f"[{model_name}] チェックポイント読み込みエラー: {e}")
                            print(f"[{model_name}] 最初から学習を開始します")
                            initial_epoch = 0
                    else:
                        print(f"[{model_name}] チェックポイントが見つかりません。最初から学習を開始します")
                else:
                    print(f"[{model_name}] チェックポイントディレクトリが存在しません。最初から学習を開始します")
            
            # SPARSE MODELING callbacks with progress tracking
            class ProgressCallback(callbacks.Callback):
                def __init__(self, inspector, model_name, total_epochs, model_idx, total_models, training_start_time, total_start_time):
                    self.inspector = inspector
                    self.model_name = model_name
                    self.total_epochs = total_epochs
                    self.epoch_start_time = None
                    self.model_idx = model_idx
                    self.total_models = total_models
                    self.training_start_time = training_start_time
                    self.total_start_time = total_start_time
                    self.last_update_time = 0
                    self.update_interval = 2.0  # 2秒ごとに更新（動きムぎないように）
                    # エポック処理時間の履歴を保持（最近のエポックを重視するため）
                    self.epoch_times = []  # 各エポックの処理時間（秒）
                    self.recent_epochs_count = 10  # 最近のNエポックを使用して予測
                    
                def on_epoch_begin(self, epoch, logs=None):
                    self.epoch_start_time = time.time()
                    # 最初のエポック開始時に即座に更新（15%で止まらないように）
                    if epoch == 0:
                        try:
                            # 最初のエポック開始時でも進捗を更新
                            overall_current_percent = 15 + 1  # 15%から少し進める
                            self.inspector.update_status({
                                'stage': 'training_epoch',
                                'model_name': model_name,
                                'model_index': self.model_idx + 1,
                                'models_total': self.total_models,
                                'epoch': epoch + 1,
                                'epochs_total': self.total_epochs,
                                'epoch_progress_percent': 0.0,
                                'training_epoch_percent': 0.0,
                                'overall_progress_percent': overall_current_percent,
                                'overall_elapsed_seconds': time.time() - self.total_start_time if self.total_start_time else 0,
                                'message': f'Training {model_name} - Epoch {epoch + 1}/{self.total_epochs} starting...'
                            })
                        except Exception:
                            pass
                    
                def on_epoch_end(self, epoch, logs=None):
                    current_time = time.time()
                    # 更新頻度を制限（動きムぎないように）
                    if current_time - self.last_update_time < self.update_interval and epoch < self.total_epochs - 1:
                        return
                    self.last_update_time = current_time
                    
                    if self.epoch_start_time:
                        epoch_time = time.time() - self.epoch_start_time
                        # エポック処理時間を履歴に追加
                        self.epoch_times.append(epoch_time)
                        # 履歴が長すぎる場合は古いものを削除（メモリ節約）
                        if len(self.epoch_times) > self.recent_epochs_count * 2:
                            self.epoch_times = self.epoch_times[-self.recent_epochs_count:]
                        
                        remaining_epochs = self.total_epochs - (epoch + 1)
                        estimated_epoch_remaining = epoch_time * remaining_epochs
                        
                        # エポックの進捗
                        epoch_progress = ((epoch + 1) / float(self.total_epochs)) * 100
                        
                        # 全体の進捗計算（全モデルの全エポックに対する進捗）
                        epochs_done = (self.model_idx * self.total_epochs) + (epoch + 1)
                        total_epochs = self.total_models * self.total_epochs
                        training_epoch_percent = 100 * (epochs_done / float(total_epochs))
                        
                        # 全体の進捗（データ読み込み10% + モデル構築5% + 訓練70% + 評価15%）
                        overall_current_percent = 15 + int(training_epoch_percent * 0.70)  # 15%起点、訓練は70%の範囲
                        
                        # 全体の時間（全プロセスを通して）
                        total_elapsed = time.time() - self.total_start_time if self.total_start_time else 0
                        
                        # 残り時間の計算（最近のエポック処理時間を重視）
                        estimated_total_remaining = 0
                        estimated_total_time = 0
                        
                        if len(self.epoch_times) >= 3:  # 最低3エポックのデータが必要
                            # 最近のエポック処理時間を使用（処理時間が増加傾向の場合に対応）
                            recent_epochs = min(len(self.epoch_times), self.recent_epochs_count)
                            recent_times = self.epoch_times[-recent_epochs:]
                            # 最近のエポックの平均時間（重み付き：より新しいエポックを重視）
                            weights = [i + 1 for i in range(len(recent_times))]  # 新しいほど重みが大きい
                            weighted_avg = sum(t * w for t, w in zip(recent_times, weights)) / sum(weights)
                            
                            # 残りエポック数を計算
                            remaining_epochs_total = total_epochs - epochs_done
                            
                            # 最近の処理時間ベースで残り時間を計算
                            estimated_total_remaining = weighted_avg * remaining_epochs_total
                            estimated_total_time = total_elapsed + estimated_total_remaining
                        elif overall_current_percent >= 15 and total_elapsed > 10:
                            # 進捗率ベースの計算（初期段階、または履歴が少ない場合）
                            if overall_current_percent > 0:
                                estimated_total_remaining = (total_elapsed / overall_current_percent) * (100 - overall_current_percent)
                                estimated_total_time = total_elapsed + estimated_total_remaining
                        elif total_elapsed > 0 and epochs_done > 0:
                            # エポックベースで計算（初期でも使える、ただし履歴が少ない場合）
                            avg_epoch_time = total_elapsed / epochs_done
                            remaining_epochs_total = total_epochs - epochs_done
                            estimated_total_remaining = avg_epoch_time * remaining_epochs_total
                            estimated_total_time = total_elapsed + estimated_total_remaining
                        
                        acc = logs.get('accuracy', 0) * 100
                        val_acc = logs.get('val_accuracy', 0) * 100
                        lr = None
                        lr_str = "N/A"
                        
                        # 学習率を安全に取得
                        try:
                            lr_raw = logs.get('lr', None)
                            # logs['lr']がスケジューラオブジェクトの場合は直接使用不可
                            if lr_raw is not None:
                                if isinstance(lr_raw, (int, float)):
                                    lr = float(lr_raw)
                                elif hasattr(lr_raw, 'numpy'):
                                    lr = float(lr_raw.numpy())
                                elif hasattr(lr_raw, '__call__'):
                                    # スケジューラの場合、現在のステップで評価
                                    step = tf.cast(model.optimizer.iterations, tf.float32)
                                    lr = float(lr_raw(step))
                                else:
                                    # それ以外の場合はoptimizerから取得を試みる
                                    raise ValueError("Cannot extract LR from logs")
                            
                            if lr is None:
                                # optimizerから直接取得を試みる
                                if hasattr(model.optimizer, 'learning_rate'):
                                    lr_obj = model.optimizer.learning_rate
                                    if isinstance(lr_obj, (int, float)):
                                        lr = float(lr_obj)
                                    elif hasattr(lr_obj, 'numpy'):
                                        lr = float(lr_obj.numpy())
                                    elif hasattr(lr_obj, '__call__'):
                                        # スケジューラの場合、現在のステップで評価
                                        step = tf.cast(model.optimizer.iterations, tf.float32)
                                        lr = float(lr_obj(step))
                                    else:
                                        lr = None
                        except Exception as e:
                            # エラーが発生した場合は無視してN/Aを表示
                            lr = None
                        
                        if lr is not None and isinstance(lr, (int, float)) and not isinstance(lr, bool):
                            lr_str = f"{float(lr):.2e}"
                        
                        # 固定幅のフォーマットで表示（段差をなくム）
                        print(f"\r[{model_name}] Epoch {epoch + 1:3d}/{self.total_epochs} | "
                              f"エポック: {epoch_progress:5.1f}% | "
                              f"訓練全体: {training_epoch_percent:5.1f}% ({epochs_done:3d}/{total_epochs}エポック) | "
                              f"【全体進捗: {overall_current_percent:3d}%】 | "
                              f"全体経過: {self.inspector.format_time(total_elapsed):>8s} | "
                              f"全体残り: {self.inspector.format_time(estimated_total_remaining):>8s} | "
                              f"完了予測: {self.inspector.format_time(estimated_total_time):>8s} | "
                              f"精度: {acc:5.2f}% | 検証: {val_acc:5.2f}% | LR: {lr_str:>8s}   ", 
                              end='', flush=True)
                        
                        # 最終エポックでは改行
                        if epoch == self.total_epochs - 1:
                            print()
                        # JSONステータス更新
                        try:
                            self.inspector.update_status({
                                'stage': 'training_epoch',
                                'model_name': model_name,
                                'model_index': self.model_idx + 1,
                                'models_total': self.total_models,
                                'epoch': epoch + 1,
                                'epochs_total': self.total_epochs,
                                'epoch_progress_percent': round(epoch_progress, 1),
                                'training_epoch_percent': round(training_epoch_percent, 1),
                                'overall_progress_percent': overall_current_percent,
                                'overall_elapsed_seconds': total_elapsed,
                                'overall_elapsed_human': self.inspector.format_time(total_elapsed),
                                'overall_remaining_est_seconds': estimated_total_remaining,
                                'overall_remaining_human': self.inspector.format_time(estimated_total_remaining),
                                'overall_estimated_total_seconds': estimated_total_time,
                                'overall_eta_human': self.inspector.format_time(estimated_total_time),
                                'accuracy_percent': round(acc, 2),
                                'val_accuracy_percent': round(val_acc, 2),
                                'learning_rate': lr_str,
                            })
                        except Exception:
                            pass
            
            # チェックポイント保存ディレクトリを作成
            checkpoint_dir = Path('checkpoints') / f'clear_sparse_4class_{model_name.lower()}'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # 最良モデル保存（従来通り）
            best_model_path = f'clear_sparse_best_4class_{model_name.lower()}_model.h5'
            
            # エポックごとのチェックポイント保存用コールバック
            class CheckpointCallback(callbacks.Callback):
                def __init__(self, checkpoint_dir, model_name):
                    self.checkpoint_dir = checkpoint_dir
                    self.model_name = model_name
                    self.last_checkpoint_path = None
                
                def on_epoch_end(self, epoch, logs=None):
                    # エポックごとにチェックポイントを保存
                    checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1:04d}.h5'
                    try:
                        # HDF5警告を抑制して保存
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning, message='.*HDF5.*')
                            self.model.save(str(checkpoint_path), save_format='h5')
                        self.last_checkpoint_path = checkpoint_path
                        # 古いチェックポイントを削除（最新5つだけ保持）
                        self._cleanup_old_checkpoints(epoch + 1)
                    except Exception as e:
                        print(f"[WARN] チェックポイント保存エラー: {e}")
                
                def _cleanup_old_checkpoints(self, current_epoch):
                    """古いチェックポイントを削除（最新5つだけ保持）"""
                    try:
                        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.h5'))
                        if len(checkpoints) > 5:
                            # 最新5つ以外を削除
                            for old_checkpoint in checkpoints[:-5]:
                                try:
                                    old_checkpoint.unlink()
                                except:
                                    pass
                    except:
                        pass
            
            checkpoint_callback = CheckpointCallback(checkpoint_dir, model_name)
            
            # 最良モデル保存コールバック（HDF5警告を抑制）
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, message='.*HDF5.*')
                # Keras 3ではsave_formatパラメータは不要（拡張子から自動判定）
                best_checkpoint = callbacks.ModelCheckpoint(
                    best_model_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=0
                )
            
            callbacks_list = [
                ProgressCallback(self, model_name, self.max_epochs, i, total_models, training_start_time, self.total_start_time),
                checkpoint_callback,  # チェックポイント保存コールバック
                callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=self.patience,
                    restore_best_weights=True,
                    verbose=0  # カスタムプログレスで表示するので0に
                ),
                # ReduceLROnPlateauは学習率スケジューラと競合するため削除
                # CosineDecayRestartsが学習率を管理する
                best_checkpoint,  # 最良モデル保存
                callbacks.CSVLogger(f'clear_sparse_training_log_4class_{model_name.lower()}.csv')
            ]
            
            # Resize data for different input sizes
            if model_config['input_size'] != (224, 224, 3):
                new_size = model_config['input_size'][0]
                print(f"\n[{model_name}] Resizing data to {new_size}x{new_size}...")
                resize_start = time.time()
                X_train_resized = np.array([cv2.resize(img, (new_size, new_size)) for img in X_train])
                X_val_resized = np.array([cv2.resize(img, (new_size, new_size)) for img in X_val])
                resize_time = time.time() - resize_start
                print(f"[{model_name}] Resize completed (Time: {self.format_time(resize_time)})")
            else:
                X_train_resized = X_train
                X_val_resized = X_val
            
            # GPU強制使用（フル活用）- モデル構築時からGPUを使用
            device_scope = None
            gpu_device_name = None
            if self.use_gpu:
                try:
                    gpus = tf.config.list_physical_devices('GPU')
                    dml_devices = tf.config.list_physical_devices('DML')
                    if dml_devices:
                        device_scope = tf.device('/DML:0')
                        gpu_device_name = 'DirectML'
                        print(f"\n[{model_name}] Using DirectML GPU for training")
                        # 強制的にDMLデバイスを使用（すべての演算をDMLに配置）
                        tf.config.set_visible_devices(dml_devices[0], 'GPU')
                    elif gpus:
                        device_scope = tf.device('/GPU:0')
                        gpu_device_name = 'CUDA GPU'
                        print(f"\n[{model_name}] Using CUDA GPU for training")
                        # 強制的にGPUデバイスを使用（すべての演算をGPUに配置）
                        tf.config.set_visible_devices(gpus[0], 'GPU')
                    # GPUを強制的に使用（CPUを無効化しないが、GPUを優先）
                    print(f"[{model_name}] GPU device forced: {gpu_device_name}")
                except Exception as e:
                    print(f"\n[{model_name}] GPU device setup warning: {e}")
            else:
                print(f"\n[{model_name}] Using CPU for training (GPU not available)")
            
            print(f"\n[{model_name}] Training started")
            print(f"[{model_name}] Training samples: {len(X_train_resized)}")
            print(f"[{model_name}] Validation samples: {len(X_val_resized)}")
            print(f"[{model_name}] Batch size: {self.batch_size}")
            print(f"[{model_name}] Max epochs: {self.max_epochs}")
            if gpu_device_name:
                print(f"[{model_name}] Device: {gpu_device_name} (GPU training enabled)")
            else:
                print(f"[{model_name}] Device: CPU (workers={self.workers}, queue={self.max_queue_size})")
            
            # WSL2環境またはGPU使用時のマルチスレッド問題を回避するため、ImageDataGeneratorの代わりにtf.data.Datasetを使用
            is_wsl2_env = False
            try:
                if os.path.exists('/proc/version'):
                    with open('/proc/version', 'r') as f:
                        version_info = f.read()
                        if 'microsoft' in version_info.lower() or 'wsl' in version_info.lower():
                            is_wsl2_env = True
                if 'WSL_DISTRO_NAME' in os.environ or 'WSL_INTEROP' in os.environ:
                    is_wsl2_env = True
            except:
                pass
            
            # WSL2環境またはGPU使用時は、ImageDataGeneratorの代わりにtf.data.Datasetを使用（マルチスレッド問題回避）
            if is_wsl2_env or self.use_gpu:
                print(f"[{model_name}] WSL2/GPU環境検出: ImageDataGeneratorの代わりにtf.data.Datasetを使用（マルチスレッド問題回避）")
                
                # データ拡張を適用する関数（SPARSE MODELING用）
                def apply_augmentation(x, y):
                    """データ拡張を適用（SPARSE MODELING用）"""
                    x = tf.cast(x, tf.float32)
                    # ランダムに拡張を適用
                    if tf.random.uniform([]) > 0.5:
                        x = tf.image.flip_left_right(x)
                    if tf.random.uniform([]) > 0.5:
                        x = tf.image.flip_up_down(x)
                    # 明度調整
                    if tf.random.uniform([]) > 0.5:
                        x = tf.image.random_brightness(x, max_delta=0.2)
                    # コントラスト調整
                    if tf.random.uniform([]) > 0.5:
                        x = tf.image.random_contrast(x, lower=0.8, upper=1.2)
                    # 色相調整
                    if tf.random.uniform([]) > 0.5:
                        x = tf.image.random_hue(x, max_delta=0.1)
                    # 飽和度調整
                    if tf.random.uniform([]) > 0.5:
                        x = tf.image.random_saturation(x, lower=0.8, upper=1.2)
                    # 値の範囲を0-1に制限
                    x = tf.clip_by_value(x, 0.0, 1.0)
                    return x, y
                
                # tf.data.Datasetを作成（NumPy配列から直接）
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train_resized, y_train))
                val_dataset = tf.data.Dataset.from_tensor_slices((X_val_resized, y_val))
                
                # 正規化（0-1にスケール）を先に適用
                train_dataset = train_dataset.map(
                    lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                    num_parallel_calls=1  # シングルスレッドで処理（マルチスレッド問題回避）
                )
                val_dataset = val_dataset.map(
                    lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                    num_parallel_calls=1
                )
                
                # データ拡張を適用（訓練データのみ）
                train_dataset = train_dataset.map(
                    apply_augmentation,
                    num_parallel_calls=1,  # シングルスレッドで処理（マルチスレッド問題回避）
                    deterministic=False
                )
                
                # バッチ化とシャッフル（buffer_sizeを適切なサイズに調整）
                shuffle_buffer_size = min(1000, len(X_train_resized))  # 最大1000に制限
                train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer_size)
                train_dataset = train_dataset.batch(self.batch_size)
                train_dataset = train_dataset.prefetch(buffer_size=1)  # プリフェッチを最小限に
                
                val_dataset = val_dataset.batch(self.batch_size)
                val_dataset = val_dataset.prefetch(buffer_size=1)
                
                train_gen = train_dataset
                val_gen = val_dataset
            else:
                # 通常の環境ではImageDataGeneratorを使用
                train_gen = train_datagen.flow(
                    X_train_resized, y_train, 
                    batch_size=self.batch_size
                )
                val_gen = val_datagen.flow(
                    X_val_resized, y_val, 
                    batch_size=self.batch_size
                )
            
            # 学習開始前にステータスを更新
            try:
                status_data = {
                    'stage': 'Training',
                    'section': f'Training {model_name}',
                    'message': f'Training {model_name} ({i+1}/{total_models})',
                    'item_progress_percent': 0.0,
                    'overall_progress_percent': 15.0 + (i / total_models) * 70.0,
                    'section_start_time': time.time(),
                    'overall_current': 15 + int((i / total_models) * 70),
                    'overall_total': 100,
                    'overall_start_time': self.total_start_time if self.total_start_time else time.time()
                }
                if hasattr(self, 'update_status'):
                    self.update_status(status_data)
                else:
                    # update_statusが存在しない場合は直接JSONファイルに書き込む
                    status_path = Path('logs') / 'training_status.json'
                    if status_path.exists():
                        with open(status_path, 'r', encoding='utf-8') as f:
                            status = json.load(f)
                        status.update(status_data)
                        status['timestamp'] = time.time()
                        with open(status_path, 'w', encoding='utf-8') as f:
                            json.dump(status, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Warning: Failed to update status: {e}")
            
            # CPU前処理の並列度を最大化（Windows対応）
            # Windowsではmultiprocessingを使わず、workers=1で安全に動作
            # GPU使用時はバッチサイズを最大限活用してGPU使用率を上げる
            steps_per_epoch = max(1, len(X_train_resized) // self.batch_size)  # 0を避ける
            validation_steps = max(1, len(X_val_resized) // self.batch_size)  # 0を避ける
            
            # Keras 3/TensorFlow 2.16+ではworkers/use_multiprocessing/max_queue_sizeは
            # model.fit()ではサポートされていません（DataGenerator側で設定）
            fit_kwargs = {
                'steps_per_epoch': steps_per_epoch,
                'epochs': self.max_epochs,
                'initial_epoch': initial_epoch,  # チェックポイントから再開する場合の開始エポック
                'validation_data': val_gen,
                'validation_steps': validation_steps,
                'class_weight': class_weight_dict,
                'callbacks': callbacks_list,
                'verbose': 1,  # 最初のエポックが始まったことを確認するため1に変更
            }
            
            # Keras 3では自動的に最適化されるため、workers情報はログのみ
            if os.name == 'nt':  # Windows
                print(f"[{model_name}] Windows detected: DataGenerator will use default settings for stability")
            else:
                print(f"[{model_name}] DataGenerator will use default settings (workers={self.workers} was configured but not used in fit())")
            
            # GPU使用時は追加の最適化を適用
            if self.use_gpu:
                print(f"[{model_name}] GPU optimization: batch_size={self.batch_size}, mixed_precision={self.use_mixed_precision}")
                print(f"[{model_name}] GPU: steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}")
            
            print(f"\n[{model_name}] Starting model.fit()...")
            
            # model.fit()実行前にXLAを確実に無効化（GPU使用時は特に重要）
            try:
                tf.config.optimizer.set_jit(False)
                os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_cpu_global_jit=false'
                os.environ['TF_XLA_AUTO_JIT'] = '0'
                # GPU使用時は追加の無効化
                if self.use_gpu:
                    # GPU実行をXLAなしで強制
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        for gpu in gpus:
                            try:
                                tf.config.experimental.set_memory_growth(gpu, True)
                            except:
                                pass
                # tf.data関数もeager実行を強制（重要：XLAエラー回避）
                try:
                    tf.config.run_functions_eagerly(True)
                    tf.data.experimental.enable_debug_mode()
                    # GPU演算をCPUにフォールバック可能にする（XLAエラー回避）
                    tf.config.set_soft_device_placement(True)
                    print(f"[{model_name}] tf.data関数もeager実行モードを有効化しました")
                    print(f"[{model_name}] GPU演算をCPUにフォールバック可能にしました（XLAエラー回避）")
                except:
                    pass
                print(f"[{model_name}] XLAを無効化しました（model.fit()実行前、GPU使用={self.use_gpu}）")
            except Exception as e:
                print(f"[{model_name}] XLA無効化警告: {e}")
            
            # model.fit()開始直後にステータス更新（プロセスが生きていることを確認）
            try:
                self.update_status({
                    'stage': 'Training',
                    'section': f'Training {model_name}',
                    'message': f'Training {model_name} - Starting model.fit()...',
                    'item_progress_percent': 0.0,
                    'overall_progress_percent': 15.0 + (i / total_models) * 70.0,
                    'timestamp': time.time()
                })
            except Exception:
                pass
            
            # WSL2環境またはGPU使用時のマルチスレッド問題を回避するため、シングルスレッド処理を強制
            original_tf_num_interop_threads = os.environ.get('TF_NUM_INTEROP_THREADS')
            original_tf_num_intraop_threads = os.environ.get('TF_NUM_INTRAOP_THREADS')
            
            if is_wsl2_env or self.use_gpu:
                # シングルスレッド処理を強制（マルチスレッド問題回避）
                os.environ['TF_NUM_INTEROP_THREADS'] = '1'
                os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
                print(f"[{model_name}] シングルスレッド処理を強制（マルチスレッド問題回避）")
            
            try:
                if device_scope:
                    with device_scope:
                        history = model.fit(train_gen, **fit_kwargs)
                else:
                    history = model.fit(train_gen, **fit_kwargs)
            except Exception as e:
                print(f"\n[{model_name}] ERROR during model.fit(): {e}")
                import traceback
                traceback.print_exc()
                # エラー時もステータス更新
                try:
                    self.update_status({
                        'stage': 'Error',
                        'message': f'Error: {str(e)[:100]}',
                        'overall_progress_percent': 15.0,
                        'timestamp': time.time()
                    })
                except Exception:
                    pass
                
                # エラーが発生した場合、チェックポイントから再開を試みる
                checkpoint_dir = Path('checkpoints') / f'clear_sparse_4class_{model_name.lower()}'
                if checkpoint_dir.exists():
                    checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.h5'))
                    if checkpoints:
                        last_checkpoint = checkpoints[-1]
                        print(f"\n[{model_name}] エラーが発生しましたが、チェックポイントが見つかりました: {last_checkpoint.name}")
                        print(f"[{model_name}] チェックポイントから再開するには、アプリの「学習再開」ボタンを使用してください")
                        print(f"[{model_name}] または、次のコマンドで再開できます:")
                        print(f"[{model_name}] python scripts/train_4class_sparse_ensemble.py --resume")
                
                raise
            finally:
                # 環境変数を元に戻す
                if is_wsl2_env or self.use_gpu:
                    try:
                        if original_tf_num_interop_threads is not None:
                            os.environ['TF_NUM_INTEROP_THREADS'] = original_tf_num_interop_threads
                        else:
                            os.environ.pop('TF_NUM_INTEROP_THREADS', None)
                        if original_tf_num_intraop_threads is not None:
                            os.environ['TF_NUM_INTRAOP_THREADS'] = original_tf_num_intraop_threads
                        else:
                            os.environ.pop('TF_NUM_INTRAOP_THREADS', None)
                    except:
                        pass
            
            model_train_time = time.time() - model_train_start
            print(f"\n[{model_name}] Training completed! Time: {self.format_time(model_train_time)}")
            
            self.models.append(model)
            self.histories.append(history)
            
            # Update overall progress
            overall_progress = ((i + 1) / total_models) * 100
            print(f"\nOverall progress: {overall_progress:.1f}% ({i+1}/{total_models})")
            if i + 1 < total_models:
                avg_time = (time.time() - training_start_time) / (i + 1)
                remaining = avg_time * (total_models - i - 1)
                print(f"Estimated remaining time: {self.format_time(remaining)}")
        
        total_training_time = time.time() - training_start_time
        print(f"\nAll models training completed!")
        print(f"Total training time: {self.format_time(total_training_time)}")
        
        self.end_section("Ensemble Training")
        return self.histories
    
    def create_sparse_ensemble_predictions(self, X_test):
        """Create SPARSE MODELING ensemble predictions for 4 classes"""
        print("\n" + "=" * 60)
        print("CREATING SPARSE MODELING ENSEMBLE PREDICTIONS")
        print("=" * 60)
        
        predictions = []
        total_models = len(self.models)
        # 予測フェーズの時間計測開始
        predict_start_time = time.time()
        
        for i, model in enumerate(self.models):
            model_name = f"Sparse_Model_{i+1}"
            self.print_progress(i, total_models, "Making Predictions", f"Predicting with {model_name}")
            
            # Resize if necessary
            if hasattr(model, 'input_shape'):
                input_size = model.input_shape[1]
                if input_size != 224:
                    X_test_resized = np.array([cv2.resize(img, (input_size, input_size)) for img in X_test])
                else:
                    X_test_resized = X_test
            else:
                X_test_resized = X_test
            
            # Predict
            pred = model.predict(X_test_resized, verbose=0)
            predictions.append(pred)
        
        self.print_progress_with_time(total_models, total_models, "Creating Predictions", "Completed",
                                     start_time=predict_start_time,
                                     section_start_time=self.section_start_time,
                                     section_name="Prediction",
                                     overall_current=90,
                                     overall_total=100,
                                     overall_start_time=self.total_start_time)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
        
        return ensemble_pred, ensemble_pred_classes
    
    def evaluate_sparse_ensemble(self, X_test, y_test):
        """Evaluate SPARSE MODELING ensemble performance for 4 classes"""
        print("\n" + "=" * 80)
        print("Evaluation")
        print("=" * 80)
        
        # Create ensemble predictions
        ensemble_pred, ensemble_pred_classes = self.create_sparse_ensemble_predictions(X_test)
        
        # Calculate accuracy
        accuracy = np.mean(ensemble_pred_classes == y_test)
        
        print(f"\nEnsemble accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Individual model accuracies
        total_models = len(self.models)
        eval_start_time = time.time()
        
        # Evaluation is in 90-100% range
        for i, model in enumerate(self.models):
            overall_current = 90 + int((i / total_models) * 10)  # 90-100%
            self.print_progress_with_time(i, total_models, "Evaluation", 
                                         f"Evaluating model {i+1}",
                                         start_time=eval_start_time,
                                         section_start_time=self.section_start_time,
                                         section_name="Evaluation",
                                         overall_current=overall_current,
                                         overall_total=100,
                                         overall_start_time=self.total_start_time)
            
            if hasattr(model, 'input_shape'):
                input_size = model.input_shape[1]
                if input_size != 224:
                    X_test_resized = np.array([cv2.resize(img, (input_size, input_size)) for img in X_test])
                else:
                    X_test_resized = X_test
            else:
                X_test_resized = X_test
            
            pred = model.predict(X_test_resized)
            pred_classes = np.argmax(pred, axis=1)
            model_acc = np.mean(pred_classes == y_test)
            print(f"\nSparse Model {i+1} Accuracy: {model_acc:.4f} ({model_acc*100:.2f}%)")
        
        self.print_progress_with_time(total_models, total_models, "評価", "完了",
                                     start_time=eval_start_time,
                                     section_start_time=self.section_start_time,
                                     section_name="評価",
                                     overall_current=100,
                                     overall_total=100,
                                     overall_start_time=self.total_start_time)
        
        # Detailed evaluation
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(y_test, ensemble_pred_classes)
        
        print("\nSPARSE MODELING Ensemble Confusion Matrix:")
        print(cm)
        
        print("\nSPARSE MODELING Ensemble Classification Report:")
        print(classification_report(y_test, ensemble_pred_classes, target_names=self.class_names))
        
        return accuracy
    
    def save_sparse_ensemble_models(self):
        """Save all SPARSE MODELING ensemble models for 4 classes"""
        print("\n" + "=" * 60)
        print("SAVING SPARSE MODELING ENSEMBLE MODELS")
        print("=" * 60)
        
        total_models = len(self.models)
        for i, model in enumerate(self.models):
            self.print_progress(i, total_models, "Saving Models", f"Saving model {i+1}")
            
            model_name = f"clear_sparse_ensemble_4class_model_{i+1}.h5"
            model.save(model_name)
            print(f"\nSaved {model_name}")
        
        self.print_progress(total_models, total_models, "Saving Models", "Complete")
        
        # Save ensemble info
        ensemble_info = {
            'model_name': 'CLEAR PROGRESS SPARSE MODELING 4-Class Ultra-High Accuracy Ensemble Washer Inspector',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_classes': 4,
            'class_names': self.class_names,
            'expected_data_counts': {'good': 1144, 'black_spot': 88, 'chipping': 117, 'scratch': 112},
            'num_models': len(self.models),
            'model_types': ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2'],
            'sparse_augmentation_params': self.sparse_augmentation_params,
            'sparse_regularization_params': self.sparse_regularization,
            'description': 'CLEAR PROGRESS SPARSE MODELING ensemble of EfficientNet models for 4-class defect detection with L1/L2 regularization and high dropout'
        }
        
        with open('clear_sparse_ensemble_4class_info.json', 'w', encoding='utf-8') as f:
            json.dump(ensemble_info, f, indent=2, ensure_ascii=False)
        
        print("\nSPARSE MODELING ensemble information saved to clear_sparse_ensemble_4class_info.json")

def main():
    """Main training function for 4 classes with SPARSE MODELING and CLEAR PROGRESS"""
    print("=" * 80)
    print("SPARSE MODELING Training with New Defect Data and Retraining")
    print("=" * 80)
    
    # Initialize inspector
    inspector = ClearProgressSparseModelingFourClassWasherInspector()
    inspector.total_start_time = time.time()
    # 学習中ビューア自動起動
    inspector.start_status_viewer()
    # GPU使用許可の確認
    inspector.request_gpu_permission()
    # デバイス情報をログ
    inspector.log_runtime_devices()
    
    try:
        # Step 1: Load all data including new defect samples
        print("\n[Step 1/2] Loading all data including new defect samples...")
        X, y, class_counts = inspector.load_and_prepare_data()
        
        # Step 2: Execute sparse modeling training
        print("\n[Step 2/2] Executing sparse modeling training...")
        print("=" * 80)
        histories = inspector.train_sparse_ensemble_with_cross_validation(X, y)
        
        # Evaluation and saving
        print("\n[Evaluation] Evaluating model performance...")
        inspector.start_section("Evaluation Data Split")
        eval_split_start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        eval_split_time = time.time() - eval_split_start
        print(f"Split completed (Time: {inspector.format_time(eval_split_time)})")
        inspector.end_section("Evaluation Data Split")
        
        inspector.start_section("Model Evaluation")
        test_acc = inspector.evaluate_sparse_ensemble(X_test, y_test)
        inspector.end_section("Model Evaluation")
        
        inspector.start_section("Model Saving")
        inspector.save_sparse_ensemble_models()
        inspector.end_section("Model Saving")
        
        # Calculate total time
        total_elapsed = time.time() - inspector.total_start_time
        
        print("\n" + "=" * 80)
        print("Step 1 Completed: Sparse Modeling Training")
        print("=" * 80)
        print(f"Final sparse modeling ensemble test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Total time: {inspector.format_time(total_elapsed)}")
        print("All CLEAR PROGRESS SPARSE MODELING 4-class ensemble models saved!")
        
        # Step 3: Retraining with all data (including existing data)
        print("\n" + "=" * 80)
        print("[Step 3] Retraining with all data (including existing data)")
        print("=" * 80)
        
        # Reset model list for retraining
        inspector.models = []
        inspector.histories = []
        retrain_start_time = time.time()
        inspector.total_start_time = retrain_start_time  # Reset retraining start time
        
        print("Starting retraining with all data...")
        print("(Using already loaded all data)")
        
        # Execute retraining
        retrain_histories = inspector.train_sparse_ensemble_with_cross_validation(X, y)
        
        # Evaluate retrained models
        print("\n[Retraining Evaluation] Evaluating retrained model performance...")
        X_train_retrain, X_test_retrain, y_train_retrain, y_test_retrain = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        retrain_test_acc = inspector.evaluate_sparse_ensemble(X_test_retrain, y_test_retrain)
        
        # Save retrained models
        print("\nSaving retrained models...")
        for i, model in enumerate(inspector.models):
            model_name = f"retrain_sparse_ensemble_4class_model_{i+1}.h5"
            model.save(model_name)
            print(f"Saved: {model_name}")
        
        retrain_total_time = time.time() - retrain_start_time
        
        print("\n" + "=" * 80)
        print("All training completed!")
        print("=" * 80)
        print(f"Step 1: Sparse modeling accuracy = {test_acc*100:.2f}% (Time: {inspector.format_time(total_elapsed)})")
        print(f"Step 2: Retraining accuracy = {retrain_test_acc*100:.2f}% (Time: {inspector.format_time(retrain_total_time)})")
        print(f"Total time: {inspector.format_time(total_elapsed + retrain_total_time)}")
        print("\nSaved models:")
        print("  - clear_sparse_ensemble_4class_model_*.h5 (Sparse modeling models)")
        print("  - retrain_sparse_ensemble_4class_model_*.h5 (Retrained models)")
        print("=" * 80)
        
        # 学習完了後のリモート転送
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from remote_transfer import RemoteTransfer
            transfer = RemoteTransfer()
            transfer.transfer_on_complete()
        except Exception as e:
            print(f"[WARN] リモート転送エラー (無視して続行): {e}")
        
    except Exception as e:
        print(f"\nError during sparse modeling training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 学習終了時にビューア停止
        try:
            inspector.stop_status_viewer()
        except Exception:
            pass

if __name__ == "__main__":
    main()
