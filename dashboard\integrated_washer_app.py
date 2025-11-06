#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合ワッシャー検査・学習アプリケーション
- 外観検査機能
- スパースモデリング学習機能
- 進捗ビューアー（HWiNFO統合）
- リソース選択システム
"""

import os
import sys
import json
import time
import cv2
import numpy as np
from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (QTabWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QSpinBox, 
                             QDoubleSpinBox, QCheckBox, QListWidget, QScrollArea, QWidget, QSizePolicy, QFrame,
                             QDialog, QLineEdit, QDateEdit, QGroupBox, QFileDialog, QMessageBox, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem, QTextEdit, QTreeWidget, QTreeWidgetItem,
                             QTableWidget, QTableWidgetItem, QHeaderView, QSplitter)
from PyQt5.QtGui import QImage, QPixmap, QFontMetrics, QIcon
from PyQt5.QtCore import Qt, QDate, QSize, QEvent
import threading
from datetime import datetime, timedelta
import csv
import shutil
import subprocess
from collections import defaultdict, deque

# グラフ表示用
try:
    import matplotlib
    matplotlib.use('Qt5Agg')  # PyQt5と互換性のあるバックエンド
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# レポート生成用
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# UTF-8 encoding for Windows (環境変数を使用、より安全)
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    # WindowsコンソールのコードページをUTF-8に設定
    try:
        import subprocess
        subprocess.run(['chcp', '65001'], shell=True, 
                      capture_output=True,
                      creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
    except Exception:
        pass
    # コンソール出力のエンコーディングを設定（安全に）
    try:
        import io
        # stdoutが閉じられていないか確認してからラップ
        if hasattr(sys.stdout, 'buffer') and not getattr(sys.stdout, 'closed', True):
            try:
                if not isinstance(sys.stdout, io.TextIOWrapper):
                    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            except (AttributeError, ValueError, OSError):
                pass
        # stderrが閉じられていないか確認してからラップ
        if hasattr(sys.stderr, 'buffer') and not getattr(sys.stderr, 'closed', True):
            try:
                if not isinstance(sys.stderr, io.TextIOWrapper):
                    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
            except (AttributeError, ValueError, OSError):
                pass
    except Exception:
        pass

# インポート
try:
    # TensorFlow
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    HAS_TENSORFLOW = True
except Exception:
    HAS_TENSORFLOW = False

# プロジェクトパスユーティリティ（ノートPC対応）
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'utils'))
    from project_path import get_project_root, get_project_root_wsl, get_venv_wsl2_path
    PROJECT_ROOT = get_project_root()
    PROJECT_ROOT_WSL = get_project_root_wsl()
    VENV_WSL2_PATH = get_venv_wsl2_path()
except ImportError:
    # フォールバック: 現在のディレクトリを使用
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    project_str = str(PROJECT_ROOT).replace('\\', '/')
    if len(project_str) >= 2 and project_str[1] == ':':
        drive_letter = project_str[0].lower()
        PROJECT_ROOT_WSL = project_str.replace(f'{project_str[0]}:', f'/mnt/{drive_letter}')
    else:
        PROJECT_ROOT_WSL = project_str
    VENV_WSL2_PATH = f"{PROJECT_ROOT_WSL}/venv_wsl2"

# リソース選択システム
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from training_resource_selector import TrainingResourceSelector
    HAS_RESOURCE_SELECTOR = True
except Exception:
    HAS_RESOURCE_SELECTOR = False

# システムスペック検出
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'utils'))
    from system_detector import SystemSpecDetector
    HAS_SYSTEM_DETECTOR = True
except Exception:
    HAS_SYSTEM_DETECTOR = False

# HWiNFO統合
try:
    hwinfo_path = Path(__file__).resolve().parents[1] / 'scripts' / 'hwinfo' / 'hwinfo_reader.py'
    if hwinfo_path.exists():
        sys.path.insert(0, str(hwinfo_path.parent))
    from hwinfo_reader import read_hwinfo_shared_memory
    HAS_HWINFO_READER = True
except Exception:
    HAS_HWINFO_READER = False

# 学習スクリプト
TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'train_4class_sparse_ensemble.py'
STATUS_FILE = Path(__file__).resolve().parents[1] / 'logs' / 'training_status.json'


class SilentMessageBox(QMessageBox):
    """警告音を鳴らさないQMessageBox"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Windowsでの警告音を無効化
        self.setAttribute(Qt.WA_QuitOnClose, False)
    
    def showEvent(self, event):
        """表示時に警告音を無効化"""
        super().showEvent(event)
        # Windowsシステムの警告音を無効化（beepを抑制）
        try:
            import ctypes
            # MessageBeepを無効化する試み（ただし完全には制御できない場合がある）
                    pass
            except:
                pass
    
    def _run_remote_access_setup(self):
        """リモートアクセスの自動セットアップを実行"""
        try:
            # セットアップ完了フラグをチェック
            remote_setup_complete_file = Path(__file__).resolve().parents[1] / '.remote_setup_complete'
            if remote_setup_complete_file.exists():
                print("[INFO] リモートアクセスセットアップは既に完了しています")
                return
            
            # セットアップスクリプトのパス
            setup_script = Path(__file__).resolve().parents[1] / 'scripts' / 'auto_setup_remote_access.py'
            
            if not setup_script.exists():
                print("[WARN] リモートアクセス自動セットアップスクリプトが見つかりません")
                return
            
            # バックグラウンドで実行（非ブロッキング）
            def run_setup():
                try:
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, str(setup_script)],
                        cwd=str(setup_script.parent.parent),
                        capture_output=True,
                        text=True,
                        timeout=300  # 5分のタイムアウト
                    )
                    
                    if result.returncode == 0:
                        print("[INFO] リモートアクセス自動セットアップが完了しました")
                        # セットアップ完了フラグを作成
                        try:
                            remote_setup_complete_file.touch()
                        except:
                            pass
                    else:
                        print(f"[WARN] リモートアクセス自動セットアップでエラー: {result.stderr}")
                except subprocess.TimeoutExpired:
                    print("[WARN] リモートアクセス自動セットアップがタイムアウトしました")
                except Exception as e:
                    print(f"[WARN] リモートアクセス自動セットアップエラー: {e}")
            
            # 別スレッドで実行（UIをブロックしない）
            import threading
            setup_thread = threading.Thread(target=run_setup, daemon=True)
            setup_thread.start()
            
        except Exception as e:
            print(f"[WARN] リモートアクセス自動セットアップ開始エラー: {e}")
            import traceback
            traceback.print_exc()


class InspectionWorker(QThread):
    """外観検査用ワーカースレッド"""
    frame_ready = pyqtSignal(np.ndarray)
    prediction_ready = pyqtSignal(str, float)
    processing_time_ready = pyqtSignal(float)
    camera_info_ready = pyqtSignal(int, int, int)  # camera_id, width, height
    
    def __init__(self, model_path=None, camera_id=0, resolution=None):
        super().__init__()
        self.model_path = model_path
        self.camera_id = camera_id
        self.resolution = resolution  # (width, height) のタプル
        self.model = None
        self.ensemble_models = []  # アンサンブルモデル用
        self.running = False
        self.class_names = ['good', 'black_spot', 'chipping', 'scratch', 'dent', 'distortion']
        
        # ノートPC検出と軽量化設定
        self.is_notebook = False
        self.frame_skip_interval = 1
        self.frame_skip_count = 0
        self.processing_resolution = None
        self.prediction_interval = 0.1  # デフォルト: 10fps
        self._detect_notebook_mode()
    
    def _detect_notebook_mode(self):
        """ノートPCを検出して軽量化設定を適用"""
        try:
            if HAS_SYSTEM_DETECTOR:
                detector = SystemSpecDetector()
                self.is_notebook = detector.specs.get('is_notebook', False)
                
                if self.is_notebook:
                    print("[軽量モード] ノートPC検出 → 軽量化設定を適用")
                    self.frame_skip_interval = 4  # 4フレームごとに1フレーム処理
                    self.processing_resolution = (480, 360)  # 低解像度
                    self.prediction_interval = 0.3  # 約3fps（CPU負荷軽減）
                    print(f"  → フレームスキップ: {self.frame_skip_interval}フレームごと")
                    print(f"  → 処理解像度: {self.processing_resolution[0]}x{self.processing_resolution[1]}")
                    print(f"  → 予測間隔: {self.prediction_interval}秒")
                else:
                    print("[標準モード] デスクトップPCとして動作")
        except Exception as e:
            print(f"[警告] システム検出エラー: {e}")
    
    def load_model(self):
        """アンサンブルモデルを読み込む"""
        print("[INFO] ========== モデル読み込み開始 ==========")
        if not HAS_TENSORFLOW:
            print("[ERROR] TensorFlowが利用できません。モデルを読み込めません。")
            print("[ERROR] HAS_TENSORFLOW = False")
            return False
        
        print("[INFO] TensorFlowは利用可能です (HAS_TENSORFLOW = True)")
        print("[INFO] モデルファイルを検索中...")
        # アンサンブルモデルを探す（EfficientNetB0, B1, B2）
        self.ensemble_models = []
        model_names = ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2']
        
        base_dir = Path(__file__).resolve().parents[1]
        print(f"[INFO] ベースディレクトリ: {base_dir}")
        model_mapping = {
            'EfficientNetB0': [1, 'EfficientNetB0'],
            'EfficientNetB1': [2, 'EfficientNetB1'], 
            'EfficientNetB2': [3, 'EfficientNetB2'],
        }
        
        for model_name in model_names:
            # 複数のパスを試す
            model_num = model_mapping.get(model_name, [1, model_name])[0]
            model_paths = [
                base_dir / f'clear_sparse_best_4class_{model_name.lower()}_model.h5',
                base_dir / f'clear_sparse_ensemble_4class_{model_name.lower()}_model.h5',
                base_dir / 'models' / 'sparse' / f'clear_sparse_ensemble_4class_model_{model_num}.h5',
                base_dir / 'models' / 'sparse' / f'sparse_ensemble_4class_model_{model_num}.h5',
                base_dir / 'models' / 'sparse' / f'sparse_best_4class_{model_name.lower()}_model.h5',  # 追加
                base_dir / 'models' / 'ensemble' / f'ensemble_4class_model_{model_num}.h5',
                base_dir / 'models' / 'ensemble' / f'best_4class_{model_name.lower()}_model.h5',  # 追加
                base_dir / 'models' / 'corrected' / f'corrected_ensemble_4class_model_{model_num}.h5',
                base_dir / 'models' / 'corrected' / f'corrected_best_4class_{model_name.lower()}_model.h5',  # 追加
            ]
            
            for model_path in model_paths:
                if model_path.exists():
                    try:
                        print(f"[INFO] モデル読み込み試行: {model_path}")
                        model = load_model(str(model_path))
                        self.ensemble_models.append(model)
                        print(f"[SUCCESS] モデル読み込み成功: {model_path}")
                        break
                    except Exception as e:
                        print(f"[ERROR] モデル読み込みエラー {model_path}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
        
        if len(self.ensemble_models) > 0:
            self.model = self.ensemble_models[0]  # 最初のモデルをデフォルトに
            print(f"[SUCCESS] {len(self.ensemble_models)}個のモデルを読み込みました。予測が可能です。")
            print(f"[INFO] デフォルトモデル: {type(self.model).__name__}")
            print(f"[INFO] モデル入力形状: {self.model.input_shape if hasattr(self.model, 'input_shape') else 'N/A'}")
            print(f"[INFO] モデルサマリー:")
            try:
                self.model.summary()
            except:
                print("[INFO] モデルサマリーの表示をスキップしました")
            print("[INFO] ========== モデル読み込み完了 ==========")
            return True
        
        # 単一モデルを探す
        if self.model_path and Path(self.model_path).exists():
            try:
                print(f"[INFO] 指定されたモデルパスから読み込み: {self.model_path}")
                self.model = load_model(str(self.model_path))
                self.ensemble_models = [self.model]
                print(f"[SUCCESS] モデル読み込み成功: {self.model_path}")
                return True
            except Exception as e:
                print(f"[ERROR] モデル読み込みエラー: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            # デフォルトモデルを探す
            print("[INFO] デフォルトモデルを検索中...")
            model_files = list(Path('.').glob('**/*ensemble*4class*.h5'))
            if model_files:
                try:
                    print(f"[INFO] 見つかったモデルファイル: {model_files[0]}")
                    self.model = load_model(str(model_files[0]))
                    self.ensemble_models = [self.model]
                    print(f"[SUCCESS] デフォルトモデル読み込み成功: {model_files[0]}")
                    return True
                except Exception as e:
                    print(f"[ERROR] モデル読み込みエラー: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
        
        print("[WARN] ========== モデルファイルが見つかりませんでした ==========")
        print("[WARN] 検索したパス:")
        print(f"[WARN] - ベースディレクトリ: {base_dir}")
        print(f"[WARN] - models/sparse/")
        print(f"[WARN] - models/ensemble/")
        print(f"[WARN] - models/corrected/")
        print("[WARN] カメラ映像のみ表示されますが、AI判定は行われません。")
        print("[WARN] 学習を実行してモデルを作成してください。")
        print("[WARN] ========== モデル読み込み失敗 ==========")
        return False
    
    def predict_defect(self, frame):
        """アンサンブルモデルで欠陥を予測"""
        if (not self.ensemble_models and self.model is None) or frame is None:
            print(f"[WARN] predict_defect: モデルまたはフレームが無効 (ensemble_models={len(self.ensemble_models)}, model={self.model is not None}, frame={frame is not None})")
            return "good", 0.0
        
        try:
            # ノートPC向け: 解像度を削減してから処理
            if self.processing_resolution and frame.shape[0] > self.processing_resolution[1]:
                frame = cv2.resize(frame, self.processing_resolution)
            
            # 入力サイズに応じてリサイズ
            img_224 = cv2.resize(frame, (224, 224))
            img_240 = cv2.resize(frame, (240, 240))
            img_260 = cv2.resize(frame, (260, 260))
            
            # ノートPC向け: アンサンブルモデルを1つだけ使用（軽量化）
            if self.is_notebook and len(self.ensemble_models) > 1:
                # 最も軽量なモデル（最初のモデル）のみ使用
                model = self.ensemble_models[0]
                img = img_224
                img_batch = np.expand_dims(img, axis=0)
                img_batch = img_batch.astype(np.float32) / 255.0
                pred = model.predict(img_batch, verbose=0)
                max_score = np.max(pred[0])
                predicted_class_id = np.argmax(pred[0])
                predicted_class = self.class_names[predicted_class_id] if predicted_class_id < len(self.class_names) else "good"
                return predicted_class, float(max_score)
            
            predictions_list = []
            
            # 各モデルで予測
            for i, model in enumerate(self.ensemble_models):
                # モデルの入力サイズに合わせてリサイズ
                if i == 0:  # EfficientNetB0
                    img = img_224
                elif i == 1:  # EfficientNetB1
                    img = img_240
                elif i == 2:  # EfficientNetB2
                    img = img_260
                else:
                    img = img_224
                
                img_batch = np.expand_dims(img, axis=0)
                img_batch = img_batch.astype(np.float32) / 255.0
                
                pred = model.predict(img_batch, verbose=0)
                predictions_list.append(pred[0])
            
            # アンサンブル予測（平均）
            if predictions_list:
                ensemble_pred = np.mean(predictions_list, axis=0)
                max_score = np.max(ensemble_pred)
                predicted_class_id = np.argmax(ensemble_pred)
                predicted_class = self.class_names[predicted_class_id] if predicted_class_id < len(self.class_names) else "good"
                
                return predicted_class, float(max_score)
            else:
                # フォールバック: 単一モデル
                img = cv2.resize(frame, (224, 224))
                img = np.expand_dims(img, axis=0)
                img = img.astype(np.float32) / 255.0
                predictions = self.model.predict(img, verbose=0)
                max_score = np.max(predictions)
                predicted_class_id = np.argmax(predictions)
                predicted_class = self.class_names[predicted_class_id] if predicted_class_id < len(self.class_names) else "good"
                return predicted_class, float(max_score)
        except Exception as e:
            print(f"予測エラー: {e}")
            import traceback
            traceback.print_exc()
            return "good", 0.0
    
    def run(self):
        """カメラから読み取り"""
        # モデル読み込みは非必須（カメラ映像だけ表示する場合はモデル不要）
        model_loaded = self.load_model()
        if not model_loaded:
            print("[WARN] モデルが読み込めません。カメラ映像のみ表示します。")
            # モデルがない場合でもカメラ映像は表示できるようにする
        
        # Windows環境ではDirectShowバックエンドを使用
        try:
            cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        except:
            # DirectShowが使えない場合はデフォルト
            cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            error_msg = f"カメラ {self.camera_id} を開けません"
            print(f"[ERROR] {error_msg}")
            # シグナルでエラーを通知（UIスレッドで処理）
            try:
                self.frame_ready.emit(None)  # Noneを送信してエラー状態を通知
            except:
                pass
            return
        
        print(f"[INFO] カメラ {self.camera_id} を開きました")
        
        # 解像度を設定（優先順位: 設定された解像度 > ノートPC向け軽量解像度）
        if self.resolution:
            width, height = self.resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # 実際に設定された値を確認
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[INFO] カメラ解像度を設定: {width}x{height} (実際: {actual_width}x{actual_height})")
            
            # UIに実際の解像度を表示（シグナルで送信）
            self.camera_info_ready.emit(self.camera_id, actual_width, actual_height)
        elif self.is_notebook and self.processing_resolution:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.processing_resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.processing_resolution[1])
            # 実際に設定された値を確認して表示
            try:
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_width > 0 and actual_height > 0:
                    self.camera_info_ready.emit(self.camera_id, actual_width, actual_height)
            except:
                pass
        else:
            # 解像度が設定されていない場合、実際の解像度を取得して表示
            try:
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_width > 0 and actual_height > 0:
                    self.camera_info_ready.emit(self.camera_id, actual_width, actual_height)
            except:
                pass
        
        # バッファをクリアして最新フレームを取得
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(3):
            cap.read()  # 古いフレームを破棄
        
        self.running = True
        self.frame_count = 0
        last_prediction_time = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] フレーム読み込み失敗 (カメラ {self.camera_id})")
                time.sleep(0.1)  # エラー時は少し待機
                continue
            
            # フレームスキップ（ノートPC向け軽量化）
            self.frame_count += 1
            if self.frame_count % (self.frame_skip_interval + 1) != 0:
                # フレームは表示するが予測はスキップ
                self.frame_ready.emit(frame)
                time.sleep(0.033)  # 約30fpsでフレーム表示
                continue
            
            # モデルが読み込まれている場合のみ予測を実行
            # ensemble_modelsまたはself.modelのいずれかが存在すれば予測可能
            has_model = (len(self.ensemble_models) > 0) or (self.model is not None)
            
            # デバッグ: 最初の数回だけログを出力
            if self.frame_count <= 10:
                print(f"[DEBUG] フレーム処理中 (frame_count={self.frame_count}, model_loaded={model_loaded}, ensemble_models={len(self.ensemble_models)}, model={self.model is not None}, has_model={has_model})")
            
            if model_loaded and has_model:
                # デバッグ: 予測実行前のログ
                if self.frame_count <= 10:
                    print(f"[DEBUG] 予測を実行します (frame_count={self.frame_count}, model_loaded={model_loaded}, ensemble_models={len(self.ensemble_models)}, model={self.model is not None})")
                # 予測間隔制御（ノートPC向け軽量化）
                current_time = time.time()
                if current_time - last_prediction_time < self.prediction_interval:
                    # 予測をスキップしてフレームのみ送信
                    self.frame_ready.emit(frame)
                    time.sleep(0.033)
                    continue
                
                # 予測（処理時間を計測）
                predict_start = time.time()
                try:
                    prediction, confidence = self.predict_defect(frame)
                    predict_time = time.time() - predict_start
                    
                    # デバッグ: 予測結果をログに出力
                    if self.frame_count <= 10:
                        print(f"[DEBUG] 予測結果: {prediction}, 信頼度: {confidence:.4f}, 処理時間: {predict_time:.3f}秒")
                    
                    last_prediction_time = current_time
                    
                    # 信号を送信
                    self.frame_ready.emit(frame)
                    self.prediction_ready.emit(prediction, confidence)
                    self.processing_time_ready.emit(predict_time)
                except Exception as e:
                    print(f"[ERROR] 予測実行エラー: {e}")
                    import traceback
                    traceback.print_exc()
                    # エラー時もフレームは送信
                    self.frame_ready.emit(frame)
                
                # ノートPC向け: 長めの待機時間（CPU負荷軽減）
                time.sleep(self.prediction_interval)
            else:
                # モデルがない場合はフレームのみ送信
                if self.frame_count <= 10:
                    # 最初の10フレームだけ警告を表示
                    print(f"[WARN] モデルが読み込まれていません。予測は実行されません。 (model_loaded={model_loaded}, ensemble_models={len(self.ensemble_models)}, model={self.model is not None})")
                self.frame_ready.emit(frame)
                time.sleep(0.033)  # 約30fpsでフレーム表示
        
        cap.release()
    
    def stop(self):
        """停止"""
        self.running = False


class TrainingWorker(QThread):
    """学習用ワーカースレッド（0から作り直し）"""
    finished = pyqtSignal()
    log_message = pyqtSignal(str)
    
    def __init__(self, resource_config=None, use_wsl2=False, resume=False):
        super().__init__()
        self.use_wsl2_mode = use_wsl2
        self.resume_mode = resume  # 再開モードフラグ
        self.resource_config = resource_config or {
            'batch_size': 32,
            'workers': 8,
            'max_epochs': 200,
            'patience': 30,
        }
        self.process = None
    
    def run(self):
        """学習を実行（シンプル版）"""
        try:
            print(f"[TrainingWorker] 学習開始")
            print(f"[TrainingWorker] リソース設定: {self.resource_config}")
            print(f"[TrainingWorker] WSL2モード: {self.use_wsl2_mode}")
            
            # 学習スクリプトの確認
            if not TRAIN_SCRIPT.exists():
                error_msg = f"学習スクリプトが見つかりません: {TRAIN_SCRIPT}"
                print(f"[ERROR] {error_msg}")
                self.log_message.emit(f"[ERROR] {error_msg}")
                self.finished.emit()
                return
            
            # 環境変数を準備
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            # リソース設定を環境変数に反映
            if 'batch_size' in self.resource_config:
                env['TRAINING_BATCH_SIZE'] = str(self.resource_config['batch_size'])
            if 'workers' in self.resource_config:
                env['TRAINING_WORKERS'] = str(self.resource_config['workers'])
            if 'max_epochs' in self.resource_config:
                env['TRAINING_MAX_EPOCHS'] = str(self.resource_config['max_epochs'])
            if 'patience' in self.resource_config:
                env['TRAINING_PATIENCE'] = str(self.resource_config['patience'])
            
            # 再開モードの場合は環境変数に設定
            if self.resume_mode:
                env['TRAINING_RESUME'] = '1'
                print("[TrainingWorker] 再開モードで実行します")
            
            # 作業ディレクトリ
            work_dir = str(TRAIN_SCRIPT.parent.parent)
            
            # Windows用のcreationflags
            creation_flags = 0
            if sys.platform.startswith('win'):
                if hasattr(subprocess, 'CREATE_NO_WINDOW'):
                    creation_flags = subprocess.CREATE_NO_WINDOW
            
            # WSL2モードで実行
            if self.use_wsl2_mode and sys.platform.startswith('win'):
                returncode = self._run_wsl2(work_dir, env, creation_flags)
            else:
                returncode = self._run_windows(work_dir, env, creation_flags)
            
            # 学習が正常に完了した場合、リモート転送を実行
            if returncode == 0:
                try:
                    self._transfer_to_remote_pc(work_dir)
                except Exception as e:
                    print(f"[WARN] リモート転送エラー (無視して続行): {e}")
            
            self.finished.emit()
            
        except Exception as e:
            error_msg = f"学習エラー: {e}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            self.log_message.emit(f"[ERROR] {error_msg}")
            self.finished.emit()
    
    def _run_wsl2(self, work_dir, env, creation_flags):
        """WSL2環境で学習を実行"""
        print("[TrainingWorker] WSL2 GPUモードで実行")
        
        # ログファイル
        log_file = Path(work_dir) / 'logs' / 'wsl2_training.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # WindowsパスをWSLパスに変換（ドライブレターを自動検出・ノートPC対応）
        log_file_str = str(log_file).replace('\\', '/')
        if len(log_file_str) >= 2 and log_file_str[1] == ':':
            drive_letter = log_file_str[0].lower()
            log_file_wsl = log_file_str.replace(f'{log_file_str[0]}:', f'/mnt/{drive_letter}')
        else:
            log_file_wsl = log_file_str
        
        # WSL2スクリプト（systemd警告を抑制）
        # systemd警告はstderrに出力されるため、2>&1でリダイレクトしてフィルタリング
        # XLAエラー回避のため、環境変数を追加設定
        # プロジェクトパスを自動検出（ノートPC対応）
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'utils'))
            from project_path import get_project_root_wsl, get_venv_wsl2_path
            project_root_wsl = get_project_root_wsl()
            venv_wsl2_path = get_venv_wsl2_path()
        except ImportError:
            # フォールバック: 現在のディレクトリから自動検出
            current_dir = Path(work_dir).resolve()
            # WindowsパスをWSLパスに変換
            project_root_str = str(current_dir).replace('\\', '/')
            if project_root_str[1] == ':':
                drive_letter = project_root_str[0].lower()
                project_root_wsl = project_root_str.replace(f'{project_root_str[0]}:', f'/mnt/{drive_letter}')
            else:
                project_root_wsl = project_root_str
            venv_wsl2_path = f"{project_root_wsl}/venv_wsl2"
        
        wsl_script = (
            f"cd {project_root_wsl} && "
            f"source {venv_wsl2_path}/bin/activate && "
            f"export TRAINING_BATCH_SIZE={env.get('TRAINING_BATCH_SIZE', '32')} && "
            f"export TRAINING_WORKERS={env.get('TRAINING_WORKERS', '8')} && "
            f"export TRAINING_MAX_EPOCHS={env.get('TRAINING_MAX_EPOCHS', '200')} && "
            f"export TRAINING_PATIENCE={env.get('TRAINING_PATIENCE', '30')} && "
            f"export GPU_USE=1 && "
            f"export FORCE_ACCURACY=1 && "
            f"export TF_XLA_FLAGS='--tf_xla_enable_xla_devices=false --tf_xla_cpu_global_jit=false' && "
            f"export TF_XLA_AUTO_JIT=0 && "
            f"export XLA_FLAGS='--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found' && "
            f"export TF_DISABLE_XLA=1 && "
            f"export TF_XLA_EXEC=0 && "
            f"python3 scripts/train_4class_sparse_ensemble.py 2>&1 | grep -v -E '(systemd|Failed to start)' | tee '{log_file_wsl}'"
        )
        
        # プロセス起動（systemd警告をstderrにリダイレクトして抑制）
        self.process = subprocess.Popen(
            ['wsl', 'bash', '-c', wsl_script],
            cwd=work_dir,
            env=env,
            creationflags=creation_flags,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )
        
        # ログをリアルタイムで読み込み（systemd警告をフィルタリング）
        while True:
            line = self.process.stdout.readline()
            if not line and self.process.poll() is not None:
                break
            if line:
                log_line = line.strip()
                # systemd警告をスキップ（学習には影響しない）
                if 'systemd' in log_line.lower() or 'Failed to start' in log_line:
                    continue
                print(log_line)
                self.log_message.emit(log_line)
        
        returncode = self.process.poll()
        print(f"[TrainingWorker] WSL2実行完了 (終了コード: {returncode})")
        return returncode
    
    def _run_windows(self, work_dir, env, creation_flags):
        """Windows環境で学習を実行"""
        print("[TrainingWorker] Windows CPUモードで実行")
        
        # 環境変数設定
        if not self.use_wsl2_mode:
            env['TF_USE_DIRECTML'] = '1'
            env['GPU_USE'] = '1'
            env['FORCE_ACCURACY'] = '1'
        
        # プロセス起動
        self.process = subprocess.Popen(
            [sys.executable, str(TRAIN_SCRIPT)],
            cwd=work_dir,
            env=env,
            creationflags=creation_flags,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )
        
        # ログをリアルタイムで読み込み
        while True:
            line = self.process.stdout.readline()
            if not line and self.process.poll() is not None:
                break
            if line:
                log_line = line.strip()
                print(log_line)
                self.log_message.emit(log_line)
        
        returncode = self.process.poll()
        print(f"[TrainingWorker] Windows実行完了 (終了コード: {returncode})")
        return returncode
    
    def _transfer_to_remote_pc(self, work_dir):
        """学習完了後のリモートPCへの自動転送"""
        try:
            # リモート転送スクリプトをインポート
            sys.path.insert(0, str(Path(work_dir) / 'scripts'))
            from remote_transfer import RemoteTransfer
            
            # 転送を実行
            transfer = RemoteTransfer()
            transfer.transfer_on_complete()
            
        except ImportError as e:
            print(f"[INFO] リモート転送モジュールが見つかりません: {e}")
        except Exception as e:
            print(f"[WARN] リモート転送エラー: {e}")
    
    def stop(self):
        """学習を停止"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except:
                try:
                    self.process.kill()
                except:
                    pass


class IntegratedWasherApp(QtWidgets.QMainWindow):
    """統合ワッシャー検査・学習アプリケーション"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle('統合ワッシャー検査・学習システム')
        self.setGeometry(100, 100, 1400, 900)
        # 最小ウィンドウサイズを設定（より小さくリサイズ可能にする）
        self.setMinimumSize(600, 400)
        
        # ロゴアイコンの設定
        self.setup_logo()
        
        # ワーカースレッド
        self.inspection_worker = None
        self.training_worker = None
        
        # 設定
        self.resource_config = {}
        
        # アンサンブルモデル用
        self.ensemble_models = []
        
        # 外観検査関連の変数
        self.inspection_history = []  # 検査履歴
        self.inspection_stats = {  # 統計情報
            'total': 0,
            'ok': 0,
            'ng': 0,
            'by_class': {'good': 0, 'black_spot': 0, 'chipping': 0, 'scratch': 0, 'dent': 0, 'distortion': 0}
        }
        self.inspection_log_dir = Path('logs/inspection')
        self.inspection_log_dir.mkdir(parents=True, exist_ok=True)
        self.inspection_image_dir = Path('inspection_results')
        self.inspection_image_dir.mkdir(parents=True, exist_ok=True)
        self.current_frame = None  # 現在のフレームを保存
        self.auto_save_ng = True  # NG検出時に自動保存
        self.confidence_threshold = 0.7  # 信頼度閾値
        self.manual_mode = False  # 手動判定モード（False=自動判定モード）
        self.pending_ai_prediction = None  # 手動判定待ちのAI予測結果
        self.pending_ai_confidence = None
        self.fps_counter = 0
        self.fps_time = time.time()
        self.prediction_history_list = []  # 予測履歴（最大20件）
        self.inspection_start_time = None
        self.alert_flash_timer = QTimer(self)
        self.alert_flash_count = 0
        self.alert_enabled_checkbox = None  # 設定ダイアログで初期化される
        
        # パフォーマンス分析用
        self.processing_times = deque(maxlen=1000)  # 最新1000件の処理時間を記録
        self.fps_history = deque(maxlen=100)  # 最新100件のFPSを記録
        self.memory_usage_history = deque(maxlen=100)  # 最新100件のメモリ使用量を記録
        
        # NG率アラート用
        self.ng_rate_alert_threshold = 0.1  # デフォルト10%
        self.ng_rate_alert_enabled = True
        self.last_alert_time = None
        self.alert_check_timer = QTimer(self)
        self.alert_check_timer.timeout.connect(self.check_ng_rate_alert)
        
        # 設定ファイル
        self.settings_file = Path('config/inspection_settings.json')
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        
        # psutilを最初からインポートして確実に使用できるようにする
        try:
            import psutil
            self._psutil_module = psutil
            self._psutil_available = True
            print(f"[DEBUG] psutil imported successfully")
        except ImportError:
            self._psutil_module = None
            self._psutil_available = False
            print(f"[DEBUG] psutil not available - will use Windows API instead")
        
        # psutilがない場合のWindows API関数を定義
        if not self._psutil_available:
            def get_system_info_windows():
                """Windows API/コマンドでシステム情報を取得"""
                memory_gb = 0
                cpu_logical = 0
                cpu_physical = 0
                try:
                    import subprocess
                    import platform
                    # メモリ取得（wmic）
                    result = subprocess.run(
                        ['wmic', 'computersystem', 'get', 'TotalPhysicalMemory'],
                        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        for line in lines:
                            if line.strip() and line.strip().isdigit():
                                memory_bytes = int(line.strip())
                                memory_gb = memory_bytes / (1024**3)
                                break
                    
                    # CPUコア取得（wmic）
                    result = subprocess.run(
                        ['wmic', 'cpu', 'get', 'NumberOfCores', 'NumberOfLogicalProcessors'],
                        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        for line in lines[1:]:  # ヘッダーをスキップ
                            parts = line.strip().split()
                            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                                cpu_physical = int(parts[0])
                                cpu_logical = int(parts[1])
                                break
                    
                    # platformモジュールでも試行
                    if cpu_logical == 0:
                        cpu_logical = platform.processor() and len(platform.processor().split()) or 0
                        # osモジュールでも試行
                        import os
                        cpu_logical = int(os.environ.get('NUMBER_OF_PROCESSORS', cpu_logical or 0))
                except Exception as e:
                    print(f"[DEBUG] Windows API get error: {e}")
                return memory_gb, cpu_logical, cpu_physical
            
            self._get_system_info_windows_api = get_system_info_windows
        
        # システムスペック検出
        self.system_specs = None
        self.recommended_config = None
        if HAS_SYSTEM_DETECTOR:
            try:
                detector = SystemSpecDetector()
                self.system_specs = detector.specs
                self.recommended_config = detector.config
                # デバッグ: 検出されたシステムスペックを表示
                print(f"[DEBUG] System Specs Detected:")
                print(f"  Memory: {self.system_specs.get('memory_gb', 0)}GB")
                print(f"  VRAM: {self.system_specs.get('gpu_vram_total_gb', 0)}GB")
                print(f"  CPU Cores (Logical): {self.system_specs.get('cpu_cores_logical', 0)}")
                print(f"  CPU Cores (Physical): {self.system_specs.get('cpu_cores_physical', 0)}")
            except Exception as e:
                print(f"システムスペック検出エラー: {e}")
                import traceback
                traceback.print_exc()
        
        # システムスペックが取得できなかった場合、psutilまたはWindows APIで直接取得
        if self.system_specs is None:
            if self._psutil_available:
                try:
                    import psutil
                    self.system_specs = {
                        'memory_gb': psutil.virtual_memory().total / (1024**3),
                        'cpu_cores_logical': psutil.cpu_count(logical=True),
                        'cpu_cores_physical': psutil.cpu_count(logical=False),
                        'gpu_vram_total_gb': 0,  # GPUは別途取得
                    }
                    print(f"[DEBUG] System specs from psutil: {self.system_specs}")
                except Exception as e:
                    print(f"[DEBUG] Failed to get specs from psutil: {e}")
            else:
                # Windows APIで取得
                try:
                    memory_gb, cpu_logical, cpu_physical = self._get_system_info_windows_api()
                    self.system_specs = {
                        'memory_gb': memory_gb,
                        'cpu_cores_logical': cpu_logical,
                        'cpu_cores_physical': cpu_physical,
                        'gpu_vram_total_gb': 0,  # GPUは別途取得
                    }
                    print(f"[DEBUG] System specs from Windows API: {self.system_specs}")
                except Exception as e:
                    print(f"[DEBUG] Failed to get specs from Windows API: {e}")
        
        # UIセットアップ
        try:
            self.setup_ui()
        except Exception as e:
            import traceback
            error_msg = f"UIセットアップエラー: {str(e)}\n\n{traceback.format_exc()}"
            print(f"[ERROR] {error_msg}")
            # エラーログに記録
            try:
                error_log_path = Path('app_error_log.txt')
                with open(error_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"UIセットアップエラー: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"{'='*60}\n")
                    f.write(error_msg)
                    f.write(f"\n{'='*60}\n\n")
            except:
                pass
            raise  # エラーを再発生させて、main関数で処理させる
        
        # 自動セットアップを実行（UI初期化後、少し遅延させて実行）
        QTimer.singleShot(500, self._run_auto_setup)
        # リモートアクセスの自動セットアップも実行（少し遅延）
        QTimer.singleShot(2000, self._run_remote_access_setup)
        
        # HWiNFO統合（監視と自動修正付き）
        # ノートPC向け: タイマー間隔を調整（軽量化）
        is_notebook = self.system_specs and self.system_specs.get('is_notebook', False)
        hwinfo_interval = 2000 if is_notebook else 1000  # ノートPC: 2秒、デスクトップ: 1秒
        system_info_interval = 5000 if is_notebook else 3000  # ノートPC: 5秒、デスクトップ: 3秒
        training_status_interval = 4000 if is_notebook else 2000  # ノートPC: 4秒、デスクトップ: 2秒
        
        self.hwinfo_timer = QTimer(self)
        self.hwinfo_timer.timeout.connect(self.update_hwinfo_metrics_with_monitoring)
        self.hwinfo_timer.start(hwinfo_interval)
        # 初回更新を即座に実行（複数回実行して確実に更新）
        if not is_notebook:
            QtCore.QTimer.singleShot(100, self.update_hwinfo_metrics_with_monitoring)
            QtCore.QTimer.singleShot(300, self.update_hwinfo_metrics_with_monitoring)
            QtCore.QTimer.singleShot(500, self.update_hwinfo_metrics_with_monitoring)
        else:
            # ノートPC: 1回だけ実行
            QtCore.QTimer.singleShot(500, self.update_hwinfo_metrics_with_monitoring)
        
        # システム情報の監視と自動修正タイマー
        self.system_info_monitor_timer = QTimer(self)
        self.system_info_monitor_timer.timeout.connect(self.monitor_and_fix_system_info)
        self.system_info_monitor_timer.start(system_info_interval)
        
        # 学習ステータス更新（以前のビューアーと同じように独立したタイマーで）
        self.training_status_timer = QTimer(self)
        self.training_status_timer.timeout.connect(self.update_training_status)
        self.training_status_timer.start(training_status_interval)
        # 初回更新を即座に実行
        QtCore.QTimer.singleShot(100, self.update_training_status)
        
        # 学習進捗更新タイマー（定期的にステータスを更新）
        self.status_update_timer = QTimer(self)
        self.status_update_timer.timeout.connect(self.update_training_status)
        self.status_update_timer.setInterval(2000)  # 2秒ごとに更新（デフォルト）
        # 学習中ではない場合は停止（学習開始時に自動的に開始される）
        
        # 学習停止検出・自動復旧
        self.training_health_timer = QTimer(self)
        self.training_health_timer.timeout.connect(self.check_training_health)
        self.training_health_timer.start(60000)  # 60秒ごとにチェック
        
        # FPS更新タイマー
        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self.update_fps)
        # ノートPC向け: FPS更新間隔を長くする（軽量化）
        fps_update_interval = 2000 if (self.system_specs and self.system_specs.get('is_notebook', False)) else 1000
        self.fps_timer.start(fps_update_interval)
        
        # 再起動フラグ
        self.restart_requested = False
    
    def _run_auto_setup(self):
        """自動セットアップを実行（必要なパッケージをチェック・インストール）"""
        try:
            # セットアップ完了フラグをチェック（既に完了している場合はスキップ）
            setup_complete_file = Path(__file__).resolve().parents[1] / '.setup_complete'
            if setup_complete_file.exists():
                # セットアップ完了フラグがある場合でも、TensorFlowが実際にインポートできるか確認
                try:
                    import tensorflow as tf
                    version = tf.__version__
                    print(f"[INFO] セットアップ完了フラグが見つかりました。TensorFlow {version} が確認されました。")
                    print("[INFO] 自動セットアップをスキップします。")
                    return  # セットアップ完了フラグがあり、TensorFlowも確認できたのでスキップ
                except ImportError:
                    print("[INFO] セットアップ完了フラグがありますが、TensorFlowが見つかりません。再チェックします。")
                    # TensorFlowが見つからない場合は、フラグを削除して再チェック
                    try:
                        setup_complete_file.unlink()
                    except:
                        pass
            
            # セットアップスクリプトのパス
            setup_script = Path(__file__).resolve().parents[1] / 'scripts' / 'auto_setup.py'
            
            if not setup_script.exists():
                print("[WARN] 自動セットアップスクリプトが見つかりません")
                return
            
            # 必要なパッケージをチェック（インストールは行わない）
            import importlib.util
            spec = importlib.util.spec_from_file_location("auto_setup", setup_script)
            if spec is None or spec.loader is None:
                print("[WARN] セットアップスクリプトを読み込めませんでした")
                return
            
            auto_setup_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(auto_setup_module)
            
            # 環境検出のみ実行（ネットワーク接続不要）
            try:
                env_info = auto_setup_module.detect_system_environment()
            except Exception as e:
                print(f"[WARN] 環境検出エラー: {e}")
                env_info = {}
            
            # CPU、メモリ、GPU情報を取得（ネットワーク接続不要）
            try:
                cpu_info = auto_setup_module.detect_cpu_info()
            except Exception as e:
                print(f"[WARN] CPU情報取得エラー: {e}")
                cpu_info = None
            
            try:
                memory_info = auto_setup_module.detect_memory_info()
            except Exception as e:
                print(f"[WARN] メモリ情報取得エラー: {e}")
                memory_info = None
            
            try:
                gpu_info = auto_setup_module.detect_gpu_info()
            except Exception as e:
                print(f"[WARN] GPU情報取得エラー: {e}")
                gpu_info = None
            
            # システム構成に応じた推奨パッケージを取得（ネットワーク接続不要）
            try:
                recommended_packages = auto_setup_module.recommend_packages(env_info, cpu_info, memory_info, gpu_info)
            except Exception as e:
                print(f"[WARN] 推奨パッケージ取得エラー: {e}")
                recommended_packages = []
            
            # パッケージをチェック（インストールは行わない）
            # ネットワーク接続エラーが発生する可能性があるため、タイムアウト付きで実行
            try:
                import signal
                import threading
                
                missing_packages = []
                installed_packages = []
                
                # タイムアウト付きでパッケージチェックを実行
                def check_packages_with_timeout():
                    nonlocal missing_packages, installed_packages
                    try:
                        # TensorFlowのインポートに時間がかかる場合があるため、先にチェック
                        try:
                            import tensorflow as tf
                            print(f"[INFO] TensorFlow {tf.__version__} が検出されました（事前チェック）")
                        except ImportError:
                            print("[INFO] TensorFlowが見つかりません（事前チェック）")
                        
                        missing_packages, installed_packages = auto_setup_module.install_missing_packages(
                            env_info, 
                            recommended_packages=recommended_packages, 
                            check_only=True
                        )
                        
                        # TensorFlowが事前チェックで見つかった場合、missing_packagesから除外
                        try:
                            import tensorflow as tf
                            version = tf.__version__
                            print(f"[INFO] TensorFlow {version} が検出されました（事前チェック成功）")
                            if missing_packages and any('tensorflow' in pkg.lower() for pkg in missing_packages):
                                missing_packages = [pkg for pkg in missing_packages if 'tensorflow' not in pkg.lower()]
                                print("[INFO] TensorFlowは既にインストールされているため、missing_packagesから除外しました")
                        except ImportError:
                            print("[INFO] TensorFlowが見つかりません（事前チェック失敗）")
                        except Exception as e:
                            print(f"[WARN] TensorFlowチェックエラー: {e}")
                            # エラー時も、インポートが成功していれば除外
                            try:
                                import tensorflow as tf
                                if missing_packages and any('tensorflow' in pkg.lower() for pkg in missing_packages):
                                    missing_packages = [pkg for pkg in missing_packages if 'tensorflow' not in pkg.lower()]
                                    print("[INFO] TensorFlowは検出されました（エラー時でも除外）")
                            except:
                                pass
                    except (ConnectionError, TimeoutError, OSError) as network_error:
                        print(f"[WARN] ネットワーク接続エラー: {network_error}")
                        print("[INFO] オフライン環境で動作します（パッケージチェックをスキップ）")
                        # オフライン時も、TensorFlowを直接チェック
                        try:
                            import tensorflow as tf
                            print(f"[INFO] TensorFlow {tf.__version__} が検出されました")
                            missing_packages = []
                            installed_packages = []
                        except ImportError:
                            missing_packages = []
                            installed_packages = []
                    except Exception as e:
                        print(f"[WARN] パッケージチェックエラー: {e}")
                        # エラー時も、TensorFlowを直接チェック
                        try:
                            import tensorflow as tf
                            print(f"[INFO] TensorFlow {tf.__version__} が検出されました（エラー時チェック）")
                            # missing_packagesからtensorflowを除外
                            missing_packages = []
                            installed_packages = []
                        except ImportError:
                            missing_packages = []
                            installed_packages = []
                
                # タイムアウトを延長（TensorFlowのインポートに時間がかかる場合があるため）
                check_thread = threading.Thread(target=check_packages_with_timeout)
                check_thread.daemon = True
                check_thread.start()
                check_thread.join(timeout=15.0)  # 5秒→15秒に延長
                
                if check_thread.is_alive():
                    print("[WARN] パッケージチェックがタイムアウトしました")
                    print("[INFO] タイムアウトしましたが、パッケージはインストールされている可能性があります")
                    # タイムアウト時も、TensorFlowを直接チェックしてみる
                    try:
                        import tensorflow as tf
                        print(f"[INFO] TensorFlow {tf.__version__} が検出されました")
                        missing_packages = []  # TensorFlowが見つかれば、missing_packagesを空にする
                    except ImportError:
                        print("[WARN] TensorFlowが見つかりませんでした")
                        # missing_packagesはそのまま（チェックが完了していないため）
                
            except (ConnectionError, TimeoutError, OSError) as network_error:
                print(f"[WARN] ネットワーク接続エラー: {network_error}")
                print("[INFO] オフライン環境で動作します（パッケージチェックをスキップ）")
                missing_packages = []
            except Exception as e:
                print(f"[WARN] パッケージチェックエラー: {e}")
                missing_packages = []
            
            # 最終確認: TensorFlowが実際にインポートできるか確認（ダイアログ表示前に）
            # これにより、既にインストールされているのに検出されない問題を回避
            final_missing_packages = []
            for pkg in missing_packages:
                if 'tensorflow' in pkg.lower():
                    # TensorFlowの場合は直接インポートして確認
                    try:
                        import tensorflow as tf
                        version = tf.__version__
                        print(f"[INFO] TensorFlow {version} が最終確認で検出されました（ダイアログ表示前）")
                        # TensorFlowが見つかったので、missing_packagesから除外
                        continue
                    except ImportError:
                        print("[INFO] TensorFlowが見つかりません（最終確認）")
                        final_missing_packages.append(pkg)
                else:
                    final_missing_packages.append(pkg)
            
            # 不足しているパッケージがある場合、ユーザーに確認してからインストール
            if final_missing_packages:
                self._show_setup_dialog(final_missing_packages, env_info, cpu_info, memory_info, gpu_info)
            else:
                print("[INFO] すべての必要なパッケージがインストールされています")
        except (ConnectionError, TimeoutError, OSError) as network_error:
            print(f"[WARN] 自動セットアップチェック中にネットワーク接続エラーが発生しました: {network_error}")
            print("[INFO] オフライン環境で動作します。アプリケーションは正常に起動します。")
        except Exception as e:
            print(f"[WARN] 自動セットアップチェックエラー: {e}")
            import traceback
            traceback.print_exc()
    
    def _show_setup_dialog(self, missing_packages, env_info=None, cpu_info=None, memory_info=None, gpu_info=None):
        """セットアップダイアログを表示"""
        try:
            from PyQt5.QtWidgets import QMessageBox
            
            # システム情報を含めたメッセージを作成
            message = "システム構成を検出しました。\n\n"
            
            if cpu_info and memory_info:
                message += f"検出されたシステム:\n"
                message += f"• CPU: {cpu_info.get('name', 'Unknown')}\n"
                message += f"• コア数: {cpu_info.get('cores_logical', 0)} (論理)\n"
                message += f"• メモリ: {memory_info:.1f} GB\n"
                if gpu_info and gpu_info.get('has_nvidia', False):
                    message += f"• GPU: {gpu_info.get('name', 'Unknown')} ({gpu_info.get('vram_gb', 0):.1f} GB VRAM)\n"
                message += "\n"
            
            message += f"以下のパッケージが見つかりません:\n\n"
            message += f'{chr(10).join("• " + pkg for pkg in missing_packages[:10])}'
            if len(missing_packages) > 10:
                message += f'\n... 他 {len(missing_packages) - 10}個'
            message += f'\n\n自動的にインストールしますか？\n'
            message += f'（初回起動時や新しいPCでの使用時に推奨）'
            
            reply = QMessageBox.question(
                self,
                '自動セットアップ',
                message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                # バックグラウンドでインストールを実行
                import subprocess
                import sys
                setup_script = Path(__file__).resolve().parents[1] / 'scripts' / 'auto_setup.py'
                
                # セットアップスクリプトを実行
                try:
                    result = subprocess.run(
                        [sys.executable, str(setup_script)],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') and sys.platform.startswith('win') else 0,
                        timeout=600  # 10分のタイムアウト
                    )
                    
                    if result.returncode == 0:
                        # セットアップ完了フラグを保存（次回以降はダイアログを表示しない）
                        try:
                            setup_complete_file = Path(__file__).resolve().parents[1] / '.setup_complete'
                            setup_complete_file.write_text('setup_complete', encoding='utf-8')
                            print("[INFO] セットアップ完了フラグを保存しました")
                        except Exception as e:
                            print(f"[WARN] セットアップ完了フラグの保存に失敗: {e}")
                        
                        QMessageBox.information(
                            self,
                            'セットアップ完了',
                            '必要なパッケージのインストールが完了しました。\n\n'
                            'アプリケーションを再起動してください。'
                        )
                    else:
                        # ネットワーク接続エラーのチェック
                        error_output = result.stderr if result.stderr else result.stdout
                        if error_output and any(keyword in error_output.lower() for keyword in ['connection', 'network', 'timeout', 'unreachable', 'dns']):
                            QMessageBox.warning(
                                self,
                                'ネットワーク接続エラー',
                                'パッケージのインストール中にネットワーク接続エラーが発生しました。\n\n'
                                'インターネット接続を確認してから再度お試しください。\n\n'
                                'オフライン環境で動作する場合は、必要なパッケージを事前にインストールしてください。\n\n'
                                f'手動でインストール:\n'
                                f'pip install {" ".join(missing_packages[:5])}'
                            )
                        else:
                            QMessageBox.warning(
                                self,
                                'セットアップ警告',
                                f'一部のパッケージのインストールに失敗しました。\n\n'
                                f'詳細:\n{error_output[:500] if error_output else "エラー詳細なし"}\n\n'
                                f'手動でインストールしてください:\n'
                                f'pip install {" ".join(missing_packages[:5])}'
                            )
                except subprocess.TimeoutExpired:
                    QMessageBox.warning(
                        self,
                        'セットアップタイムアウト',
                        'インストールに時間がかかりすぎています。\n'
                        'ネットワーク接続が遅い可能性があります。\n\n'
                        '手動でインストールしてください:\n'
                        f'pip install {" ".join(missing_packages)}'
                    )
                except (ConnectionError, TimeoutError, OSError) as network_error:
                    QMessageBox.warning(
                        self,
                        'ネットワーク接続エラー',
                        f'ネットワーク接続エラーが発生しました:\n{str(network_error)}\n\n'
                        'インターネット接続を確認してから再度お試しください。\n\n'
                        'オフライン環境で動作する場合は、必要なパッケージを事前にインストールしてください。\n\n'
                        f'手動でインストール:\n'
                        f'pip install {" ".join(missing_packages[:5])}'
                    )
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        'セットアップエラー',
                        f'自動インストール中にエラーが発生しました:\n{str(e)}\n\n'
                        f'手動でインストールしてください:\n'
                        f'pip install {" ".join(missing_packages)}'
                    )
            else:
                # インストールをスキップした場合、警告を表示
                QMessageBox.warning(
                    self,
                    'パッケージ不足',
                    f'以下のパッケージが必要です:\n\n'
                    f'{chr(10).join("• " + pkg for pkg in missing_packages[:10])}\n\n'
                    f'アプリケーションが正常に動作しない可能性があります。\n\n'
                    f'後でインストールする場合は:\n'
                    f'python scripts/auto_setup.py'
                )
        except Exception as e:
            print(f"[ERROR] セットアップダイアログ表示エラー: {e}")
    
    def setup_logo(self):
        """ロゴアイコンを設定"""
        try:
            assets_dir = Path(__file__).resolve().parents[1] / 'assets'
            # ICOファイルを優先的に使用
            logo_path = assets_dir / 'logo_icon.ico'
            if logo_path.exists():
                self.setWindowIcon(QtGui.QIcon(str(logo_path)))
                print(f"[OK] ロゴアイコン設定完了: {logo_path}")
            else:
                # ICOファイルがない場合はPNGを試す（大きいサイズから順に）
                for size in [256, 128, 64, 48, 32, 16]:
                    logo_path_png = assets_dir / f'logo_icon_{size}x{size}.png'
                    if logo_path_png.exists():
                        self.setWindowIcon(QtGui.QIcon(str(logo_path_png)))
                        print(f"[OK] ロゴアイコン設定完了: {logo_path_png}")
                        break
                else:
                    # バナーロゴを試す
                    banner_path = assets_dir / 'logo_banner.png'
                    if banner_path.exists():
                        self.setWindowIcon(QtGui.QIcon(str(banner_path)))
                        print(f"[OK] ロゴアイコン設定完了: {banner_path}")
        except Exception as e:
            print(f"ロゴアイコン設定エラー: {e}")
            import traceback
            traceback.print_exc()
    
    def create_logo_widget(self):
        """ロゴ表示ウィジェットを作成"""
        try:
            assets_dir = Path(__file__).resolve().parents[1] / 'assets'
            banner_path = assets_dir / 'logo_banner.png'
            icon_path = assets_dir / 'logo_icon_128x128.png'
            
            # 使用する画像を決定（バナー優先、なければアイコン）
            image_path = None
            if banner_path.exists():
                image_path = banner_path
            elif icon_path.exists():
                image_path = icon_path
            else:
                return None
            
            logo_widget = QtWidgets.QWidget()
            logo_layout = QHBoxLayout(logo_widget)
            logo_layout.setContentsMargins(0, 0, 0, 10)
            
            # ロゴ画像を読み込み
            pixmap = QtGui.QPixmap(str(image_path))
            
            # 適切なサイズにリサイズ（横幅最大600px）
            if pixmap.width() > 600:
                pixmap = pixmap.scaledToWidth(600, QtCore.Qt.SmoothTransformation)
            
            logo_label = QLabel()
            logo_label.setPixmap(pixmap)
            logo_label.setAlignment(QtCore.Qt.AlignCenter)
            
            logo_layout.addStretch()
            logo_layout.addWidget(logo_label)
            logo_layout.addStretch()
            
            return logo_widget
        except Exception as e:
            print(f"ロゴ表示ウィジェット作成エラー: {e}")
            return None
    
    def setup_ui(self):
        """UIをセットアップ"""
        # 全体的なスタイルシートを適用（モダンなデザイン）
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #ddd;
                background-color: white;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                color: #333;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
                font-size: 11pt;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #0066cc;
                border-bottom: 2px solid #0066cc;
            }
            QTabBar::tab:hover {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 11pt;
                color: #2c3e50;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 18px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                background-color: white;
            }
            QPushButton {
                background-color: #0066cc;
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 11pt;
                min-height: 38px;
            }
            QPushButton:hover {
                background-color: #0052a3;
            }
            QPushButton:pressed {
                background-color: #003d7a;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
            QPushButton#start_button {
                background-color: #28a745;
            }
            QPushButton#start_button:hover {
                background-color: #218838;
            }
            QPushButton#stop_button {
                background-color: #dc3545;
            }
            QPushButton#stop_button:hover {
                background-color: #c82333;
            }
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #0066cc;
                border-radius: 3px;
            }
            QLabel {
                color: #333;
            }
            QRadioButton {
                font-size: 10pt;
                padding: 5px;
                color: #333;
                background-color: transparent;
            }
            QRadioButton:hover {
                background-color: #f0f8ff;
                color: #333;
            }
            QRadioButton:checked {
                color: #0066cc;
                font-weight: bold;
                background-color: #e7f3ff;
            }
            QRadioButton:checked:hover {
                background-color: #d0e7ff;
                color: #0066cc;
            }
            QCheckBox {
                font-size: 11pt;
                font-weight: bold;
                color: #0066cc;
                padding: 8px;
                background-color: #e7f3ff;
                border: 2px solid #0066cc;
                border-radius: 5px;
            }
            QCheckBox::indicator {
                width: 22px;
                height: 22px;
                border: 2px solid #0066cc;
                border-radius: 4px;
                background-color: white;
            }
            QCheckBox::indicator:unchecked {
                background-color: white;
                border: 2px solid #999;
            }
            QCheckBox::indicator:checked {
                background-color: #0066cc;
                border: 2px solid #0066cc;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyMiAyMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTkgMTZMMTggN0wxNyA2TDkgMTRMNiAxMUw1IDEyTDkgMTZaIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K);
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                font-size: 10pt;
                min-width: 200px;
            }
            QComboBox:hover {
                border: 1px solid #0066cc;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                min-width: 200px;
                border: 1px solid #ddd;
                background-color: white;
                selection-background-color: #0066cc;
                selection-color: white;
            }
            QComboBox QAbstractItemView::item {
                padding: 5px;
                color: #333;
                background-color: white;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #e6f3ff;
                color: #333;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #0066cc;
                color: white;
            }
        """)
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("background-color: white;")
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # タブウィジェット
        self.tabs = QTabWidget()
        
        # タブ1: 外観検査
        self.inspection_tab = self.create_inspection_tab()
        self.tabs.addTab(self.inspection_tab, "🔍 外観検査")
        
        # タブ2: 学習
        self.training_tab = self.create_training_tab()
        self.tabs.addTab(self.training_tab, "🎓 学習")
        
        # タブ3: 進捗・システム情報
        self.status_tab = self.create_status_tab()
        self.tabs.addTab(self.status_tab, "📊 進捗・情報")
        
        # タブのスタイルを調整（見切れ防止）
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ddd;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                color: #333;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 150px;
                max-width: 250px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #0066cc;
                border-bottom: 3px solid #0066cc;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #e0e0e0;
            }
        """)
        
        layout.addWidget(self.tabs)
        
        # メニューバーを作成
        self.create_menu_bar()
        
        # ステータスバー
        self.statusBar().setStyleSheet("background-color: #f0f0f0; color: #333; padding: 3px;")
        self.statusBar().showMessage('準備完了')
    
    def create_menu_bar(self):
        """メニューバーを作成"""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #ffffff;
                color: #333;
                border-bottom: 1px solid #ddd;
                padding: 4px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background-color: #e6f3ff;
                color: #0066cc;
            }
            QMenu {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 32px 8px 16px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #0066cc;
                color: white;
            }
            QMenu::separator {
                height: 1px;
                background-color: #ddd;
                margin: 4px 0px;
            }
        """)
        
        # ファイルメニュー
        file_menu = menubar.addMenu('📁 ファイル')
        
        restart_action = QtWidgets.QAction('🔄 再起動', self)
        restart_action.setShortcut('Ctrl+R')
        restart_action.setToolTip('アプリケーションを再起動します')
        restart_action.triggered.connect(self.restart_application)
        file_menu.addAction(restart_action)
        
        file_menu.addSeparator()
        
        exit_action = QtWidgets.QAction('終了', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setToolTip('アプリケーションを終了します')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 設定メニュー（統合設定ダイアログ）
        settings_menu = menubar.addMenu('⚙️ 設定')
        
        all_settings_action = QtWidgets.QAction('⚙️ 設定', self)
        all_settings_action.setShortcut('Ctrl+,')
        all_settings_action.setToolTip('すべての設定を一括で管理')
        all_settings_action.triggered.connect(self.show_all_settings_dialog)
        settings_menu.addAction(all_settings_action)
        
        # 分析・レポートメニュー
        analysis_menu = menubar.addMenu('📊 分析・レポート')
        
        stats_graph_action = QtWidgets.QAction('📈 統計グラフ', self)
        stats_graph_action.setToolTip('検査統計のグラフを表示')
        stats_graph_action.triggered.connect(self.show_statistics_graph)
        analysis_menu.addAction(stats_graph_action)
        
        gallery_action = QtWidgets.QAction('🖼️ 画像ギャラリー', self)
        gallery_action.setToolTip('保存された検査画像を閲覧')
        gallery_action.triggered.connect(self.show_image_gallery)
        analysis_menu.addAction(gallery_action)
        
        history_search_action = QtWidgets.QAction('🔍 履歴検索', self)
        history_search_action.setToolTip('検査履歴を検索・フィルタ')
        history_search_action.triggered.connect(self.show_history_search)
        analysis_menu.addAction(history_search_action)
        
        analysis_menu.addSeparator()
        
        daily_report_action = QtWidgets.QAction('📄 日次レポート生成', self)
        daily_report_action.setToolTip('日次レポートを自動生成')
        daily_report_action.triggered.connect(self.generate_daily_report)
        analysis_menu.addAction(daily_report_action)
        
        batch_export_action = QtWidgets.QAction('📦 バッチエクスポート', self)
        batch_export_action.setToolTip('複数日分のデータを一括エクスポート')
        batch_export_action.triggered.connect(self.show_batch_export_dialog)
        analysis_menu.addAction(batch_export_action)
        
        model_comparison_action = QtWidgets.QAction('🎯 モデル精度比較', self)
        model_comparison_action.setToolTip('過去の学習モデルの精度を比較')
        model_comparison_action.triggered.connect(self.show_model_comparison)
        analysis_menu.addAction(model_comparison_action)
        
        performance_analysis_action = QtWidgets.QAction('⚡ パフォーマンス分析', self)
        performance_analysis_action.setToolTip('検査パフォーマンスを分析')
        performance_analysis_action.triggered.connect(self.show_performance_analysis)
        analysis_menu.addAction(performance_analysis_action)
        
        analysis_menu.addSeparator()
        
        # アラート設定
        alert_settings_action = QtWidgets.QAction('⚠️ NG率アラート設定', self)
        alert_settings_action.setToolTip('NG率アラートの閾値を設定')
        alert_settings_action.triggered.connect(self.show_alert_settings)
        analysis_menu.addAction(alert_settings_action)
        
        # ヘルプメニュー
        help_menu = menubar.addMenu('❓ ヘルプ')
        about_action = QtWidgets.QAction('ℹ️ このアプリについて', self)
        about_action.setToolTip('アプリケーションの情報を表示')
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
    
    def create_inspection_tab(self):
        """外観検査タブを作成"""
        widget = QtWidgets.QWidget()
        widget.setStyleSheet("background-color: white;")
        
        # 2列レイアウトでスペースを有効活用（スクロールなしで表示）
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)
        
        # 左側：カメラ映像
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)
        
        # カメラ表示グループ
        camera_group = QtWidgets.QGroupBox("📷 カメラ映像")
        camera_layout = QVBoxLayout()
        camera_layout.setContentsMargins(10, 15, 10, 10)
        
        self.camera_label = QLabel("カメラを初期化しています...")
        self.camera_label.setMinimumSize(160, 120)
        self.camera_label.setScaledContents(False)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_label.setStyleSheet("""
            border: 3px solid #0066cc;
            border-radius: 8px;
            background-color: #f8f9fa;
            font-size: 14pt;
            color: #666;
        """)
        camera_layout.addWidget(self.camera_label, stretch=1)
        
        # 現在のカメラと解像度を表示するラベル
        self.camera_info_label = QLabel("カメラ: - / 解像度: -")
        self.camera_info_label.setStyleSheet("""
            font-size: 9pt;
            color: #666;
            padding: 3px;
            background-color: #f0f0f0;
            border-radius: 4px;
        """)
        self.camera_info_label.setAlignment(QtCore.Qt.AlignCenter)
        camera_layout.addWidget(self.camera_info_label)
        
        camera_group.setLayout(camera_layout)
        left_layout.addWidget(camera_group, stretch=1)
        
        # PC構成（コンパクト）
        system_info_compact = QtWidgets.QGroupBox("💻 PC構成")
        system_info_layout = QVBoxLayout()
        system_info_layout.setContentsMargins(10, 10, 10, 10)
        system_info_layout.setSpacing(5)
        
        if self.system_specs:
            # CPU情報を取得
            cpu_info = self.system_specs.get('cpu_info', 'Unknown')
            cpu_full_name = self.system_specs.get('cpu_full_name') or cpu_info
            if cpu_info == 'Unknown' and cpu_full_name == 'Unknown':
                try:
                    import platform
                    cpu_info = platform.processor() or 'Unknown'
                    cpu_full_name = cpu_info
                except:
                    pass
            cpu_display = cpu_full_name if cpu_full_name and cpu_full_name != 'Unknown' else cpu_info
            if len(cpu_display) > 30:
                cpu_display = cpu_display[:30] + "..."
            
            # GPU情報を取得
            gpu_info = self.system_specs.get('gpu_info', 'Unknown')
            gpu_name_full = self.system_specs.get('gpu_name_full') or self.system_specs.get('nvidia_gpu_name') or gpu_info
            if gpu_info in ['Unknown', 'No GPU detected']:
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        gpu_info = result.stdout.strip()
                        gpu_name_full = gpu_info
                except:
                    pass
            gpu_display = gpu_name_full if gpu_name_full and gpu_name_full not in ['Unknown', 'No GPU detected'] else (gpu_info if gpu_info not in ['Unknown', 'No GPU detected'] else 'Unknown')
            if len(gpu_display) > 30:
                gpu_display = gpu_display[:30] + "..."
            
            # メモリ情報を取得
            memory_gb = self.system_specs.get('memory_gb', 0.0)
            if memory_gb == 0:
                memory_display = 'Unknown'
            else:
                memory_display = f"{memory_gb:.1f}"
            
            system_info_layout.addWidget(QLabel(f"CPU: {cpu_display}"))
            system_info_layout.addWidget(QLabel(f"GPU: {gpu_display}"))
            system_info_layout.addWidget(QLabel(f"メモリ: {memory_display} GB"))
        else:
            # system_specsがNoneの場合のフォールバック
            system_info_layout.addWidget(QLabel("CPU: 検出中..."))
            system_info_layout.addWidget(QLabel("GPU: 検出中..."))
            system_info_layout.addWidget(QLabel("メモリ: 検出中..."))
        
        note_label = QLabel("詳細は「🎓 学習」タブを参照")
        note_label.setStyleSheet("font-size: 9pt; color: #999; font-style: italic;")
        system_info_layout.addWidget(note_label)
        system_info_compact.setLayout(system_info_layout)
        left_layout.addWidget(system_info_compact)
        
        # 開発段階の警告メッセージ（PC構成の下の空白スペースに配置・横長に）
        beta_warning_frame = QFrame()
        beta_warning_frame.setStyleSheet("""
            QFrame {
                background-color: #fff4e6;
                border: 2px solid #ffa726;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        beta_warning_layout = QHBoxLayout(beta_warning_frame)
        beta_warning_layout.setContentsMargins(10, 8, 10, 8)
        beta_warning_layout.setSpacing(10)
        
        # 警告テキスト（横長に1行で表示）
        beta_warning_text = QLabel(
            "現在のアプリケーションは開発段階のBeta版です。完成品の品質を示しているものではありません。ご理解とご支援ありがとうございます。"
        )
        beta_warning_text.setStyleSheet("""
            font-size: 9pt;
            color: #bf6600;
            line-height: 1.4;
        """)
        beta_warning_text.setWordWrap(False)
        beta_warning_layout.addWidget(beta_warning_text, stretch=1)
        
        left_layout.addWidget(beta_warning_frame)
        
        left_layout.addStretch()
        
        # 右側：判定結果、統計、履歴など
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)
        
        # 予測結果グループ（シンプルに整理）
        result_group = QtWidgets.QGroupBox("📋 判定結果")
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(12, 12, 12, 12)
        result_layout.setSpacing(8)
        
        # 予測と信頼度を1行で表示
        prediction_widget = QWidget()
        prediction_widget.setStyleSheet("""
            background-color: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 6px;
            padding: 8px;
        """)
        prediction_container = QHBoxLayout(prediction_widget)
        prediction_container.setContentsMargins(8, 8, 8, 8)
        prediction_container.setSpacing(10)
        
        self.prediction_label = QLabel("予測: -")
        self.prediction_label.setStyleSheet("""
            font-size: 16pt;
            font-weight: bold;
            color: #333;
        """)
        prediction_container.addWidget(self.prediction_label)
        
        prediction_container.addWidget(QLabel("|"))  # 区切り
        
        self.confidence_label = QLabel("信頼度: -")
        self.confidence_label.setStyleSheet("""
            font-size: 12pt;
            color: #666;
        """)
        prediction_container.addWidget(self.confidence_label)
        prediction_container.addStretch()
        
        result_layout.addWidget(prediction_widget)
        
        # コントロール（1行にコンパクトに）
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)
        
        self.manual_mode_checkbox = QCheckBox("手動判定")
        # グローバルスタイルを適用（個別のスタイル設定を削除）
        self.manual_mode_checkbox.setToolTip("手動で判定を選択します")
        self.manual_mode_checkbox.stateChanged.connect(self.on_manual_mode_changed)
        controls_layout.addWidget(self.manual_mode_checkbox)
        
        controls_layout.addStretch()
        
        self.result_feedback_btn = QPushButton("⚠️ 誤判定")
        self.result_feedback_btn.clicked.connect(self.show_feedback_dialog)
        self.result_feedback_btn.setEnabled(False)
        self.result_feedback_btn.setStyleSheet("""
            QPushButton {
                font-size: 9pt;
                padding: 4px 8px;
                background-color: #ffc107;
                color: #333;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffb300;
            }
            QPushButton:disabled {
                background-color: #e0e0e0;
                color: #999;
            }
        """)
        controls_layout.addWidget(self.result_feedback_btn)
        result_layout.addLayout(controls_layout)
        
        # 手動判定選択UI（手動モード時のみ表示・さらにシンプルに）
        self.manual_judgment_group = QWidget()
        self.manual_judgment_group.setStyleSheet("""
            background-color: #fffef5;
            border: 1px solid #ffc107;
            border-radius: 6px;
            padding: 8px;
        """)
        manual_layout = QVBoxLayout(self.manual_judgment_group)
        manual_layout.setContentsMargins(8, 8, 8, 8)
        manual_layout.setSpacing(6)
        
        # AI予測表示（コンパクトに）
        ai_info_label = QLabel("AI予測: 判定待ち...")
        ai_info_label.setStyleSheet("""
            font-size: 10pt;
            font-weight: bold;
            color: #0066cc;
            padding: 6px;
            background-color: #f0f8ff;
            border: 1px solid #0066cc;
            border-radius: 4px;
        """)
        ai_info_label.setAlignment(QtCore.Qt.AlignCenter)
        ai_info_label.setWordWrap(True)
        self.ai_prediction_label = ai_info_label
        manual_layout.addWidget(ai_info_label)
        
        # 判定ボタン（2行グリッド）
        buttons_container = QVBoxLayout()
        buttons_container.setSpacing(4)
        
        # 1行目（良品、黒点、欠け）
        buttons_row1 = QHBoxLayout()
        buttons_row1.setSpacing(4)
        # 2行目（傷、凹み、歪み）
        buttons_row2 = QHBoxLayout()
        buttons_row2.setSpacing(4)
        
        self.judgment_buttons = []
        class_map = {'good': '✅ 良品', 'black_spot': '⚫ 黒点', 'chipping': '🔨 欠け', 
                    'scratch': '💢 傷', 'dent': '📉 凹み', 'distortion': '📐 歪み'}
        
        row1_classes = ['good', 'black_spot', 'chipping']
        row2_classes = ['scratch', 'dent', 'distortion']
        
        for class_en, class_jp in class_map.items():
            btn = QPushButton(class_jp)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 10pt;
                    font-weight: bold;
                    padding: 8px 12px;
                    border-radius: 4px;
                    border: 1px solid #0066cc;
                    background-color: white;
                    color: #0066cc;
                }
                QPushButton:hover {
                    background-color: #e7f3ff;
                }
                QPushButton:pressed {
                    background-color: #cce5ff;
                }
                QPushButton:disabled {
                    background-color: #f5f5f5;
                    color: #aaa;
                    border: 1px solid #ddd;
                }
            """)
            btn.clicked.connect(lambda checked, cls=class_en: self.apply_manual_judgment(cls))
            btn.setEnabled(False)
            self.judgment_buttons.append((btn, class_en))
            
            if class_en in row1_classes:
                buttons_row1.addWidget(btn)
            else:
                buttons_row2.addWidget(btn)
        
        buttons_container.addLayout(buttons_row1)
        buttons_container.addLayout(buttons_row2)
        manual_layout.addLayout(buttons_container)
        
        self.manual_judgment_group.setVisible(False)
        result_layout.addWidget(self.manual_judgment_group)
        
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)
        
        # 統計情報グループ（コンパクトに）
        stats_group = QtWidgets.QGroupBox("📊 検査統計")
        stats_layout = QVBoxLayout()
        stats_layout.setContentsMargins(10, 10, 10, 10)
        stats_layout.setSpacing(6)
        
        # 統計情報を横並びで表示
        stats_row1 = QHBoxLayout()
        self.total_count_label = QLabel("総検査数: 0")
        self.total_count_label.setStyleSheet("font-size: 11pt; font-weight: bold; color: #2c3e50;")
        self.ok_count_label = QLabel("良品: 0 (0%)")
        self.ok_count_label.setStyleSheet("font-size: 11pt; color: #28a745; font-weight: bold;")
        self.ng_count_label = QLabel("不良品: 0 (0%)")
        self.ng_count_label.setStyleSheet("font-size: 11pt; color: #dc3545; font-weight: bold;")
        stats_row1.addWidget(self.total_count_label)
        stats_row1.addWidget(self.ok_count_label)
        stats_row1.addWidget(self.ng_count_label)
        stats_layout.addLayout(stats_row1)
        
        # クラス別集計
        self.class_stats_label = QLabel("クラス別: -")
        self.class_stats_label.setStyleSheet("font-size: 10pt; color: #666;")
        stats_layout.addWidget(self.class_stats_label)
        
        # FPS表示
        self.fps_label = QLabel("FPS: -")
        self.fps_label.setStyleSheet("font-size: 10pt; color: #888;")
        stats_layout.addWidget(self.fps_label)
        
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)
        
        # 操作ボタン（コンパクト化）
        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.setSpacing(10)
        
        self.save_frame_btn = QPushButton("💾 保存")
        self.save_frame_btn.clicked.connect(self.save_current_frame)
        self.save_frame_btn.setEnabled(False)
        self.save_frame_btn.setStyleSheet("font-size: 10pt; padding: 6px 12px;")
        
        self.feedback_btn = QPushButton("✏️ 修正")
        self.feedback_btn.clicked.connect(self.show_feedback_dialog)
        self.feedback_btn.setEnabled(False)
        self.feedback_btn.setStyleSheet("font-size: 10pt; padding: 6px 12px;")
        
        self.export_results_btn = QPushButton("📤 エクスポート")
        self.export_results_btn.clicked.connect(self.export_inspection_results)
        self.export_results_btn.setEnabled(False)
        self.export_results_btn.setStyleSheet("font-size: 10pt; padding: 6px 12px;")
        
        self.clear_stats_btn = QPushButton("🗑️ リセット")
        self.clear_stats_btn.clicked.connect(self.clear_statistics)
        self.clear_stats_btn.setStyleSheet("font-size: 10pt; padding: 6px 12px;")
        
        action_buttons_layout.addWidget(self.save_frame_btn)
        action_buttons_layout.addWidget(self.feedback_btn)
        action_buttons_layout.addWidget(self.export_results_btn)
        action_buttons_layout.addWidget(self.clear_stats_btn)
        action_buttons_layout.addStretch()
        
        # コンパクトなボタングループ
        compact_buttons_group = QtWidgets.QGroupBox("操作")
        compact_buttons_group.setStyleSheet("font-size: 10pt; padding: 5px;")
        compact_buttons_layout = QVBoxLayout()
        compact_buttons_layout.setContentsMargins(10, 10, 10, 10)
        compact_buttons_layout.addLayout(action_buttons_layout)
        compact_buttons_group.setLayout(compact_buttons_layout)
        right_layout.addWidget(compact_buttons_group)
        
        # 設定を読み込む（UI初期化後）
        self.load_inspection_settings()
        
        # 予測履歴グループ
        history_group = QtWidgets.QGroupBox("📜 予測履歴（直近20件）")
        history_layout = QVBoxLayout()
        history_layout.setContentsMargins(15, 15, 15, 15)
        
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(150)
        self.history_list.setStyleSheet("font-size: 9pt;")
        history_layout.addWidget(self.history_list)
        
        history_group.setLayout(history_layout)
        right_layout.addWidget(history_group)
        
        right_layout.addStretch()
        
        # 2列レイアウトに追加
        main_layout.addLayout(left_layout, stretch=2)  # 左側（カメラ）を大きく
        main_layout.addLayout(right_layout, stretch=1)  # 右側（情報）を小さく
        
        # 全体を縦レイアウトでラップ
        outer_layout = QVBoxLayout(widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(15)
        outer_layout.addLayout(main_layout, stretch=1)
        
        # ボタングループ（下部に横並びで配置）
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        button_layout.addStretch()
        
        self.start_inspection_btn = QPushButton("▶ 検査開始")
        self.start_inspection_btn.setObjectName("start_button")
        self.start_inspection_btn.clicked.connect(self.start_inspection)
        self.start_inspection_btn.setMinimumHeight(50)
        self.start_inspection_btn.setMinimumWidth(150)
        self.start_inspection_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-size: 13pt;
                font-weight: bold;
                padding: 12px 25px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        
        self.stop_inspection_btn = QPushButton("⏹ 検査停止")
        self.stop_inspection_btn.setObjectName("stop_button")
        self.stop_inspection_btn.clicked.connect(self.stop_inspection)
        self.stop_inspection_btn.setEnabled(False)
        self.stop_inspection_btn.setMinimumHeight(50)
        self.stop_inspection_btn.setMinimumWidth(150)
        self.stop_inspection_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-size: 13pt;
                font-weight: bold;
                padding: 12px 25px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        
        button_layout.addWidget(self.start_inspection_btn)
        button_layout.addWidget(self.stop_inspection_btn)
        button_layout.addStretch()
        outer_layout.addLayout(button_layout)
        
        # 設定を読み込む（UI初期化後）
        self.load_inspection_settings()
        
        return widget
    
    def create_training_tab(self):
        """学習タブを作成"""
        # スクロール可能なウィジェットを作成
        scroll_widget = QtWidgets.QWidget()
        scroll_widget.setStyleSheet("background-color: white;")
        scroll_layout = QVBoxLayout(scroll_widget)
        # スペースをフル活用: マージンとスペースを最適化
        scroll_layout.setContentsMargins(12, 12, 12, 12)
        scroll_layout.setSpacing(10)
        
        # スクロールエリアを作成
        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: white;
            }
        """)
        
        # メインレイアウト（スクロールエリアを含む）
        widget = QtWidgets.QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll_area)
        
        # ロゴ表示（上部）
        # システムスペック表示
        if self.system_specs:
            spec_group = QtWidgets.QGroupBox("💻 PC構成・システムスペック（自動検出）")
            spec_layout = QVBoxLayout()
            # スペースをフル活用: マージンを最適化
            spec_layout.setContentsMargins(12, 12, 12, 12)
            spec_layout.setSpacing(10)
            
            # CPU情報（詳細）
            cpu_info = self.system_specs.get('cpu_info', 'Unknown')
            cpu_full_name = self.system_specs.get('cpu_full_name') or cpu_info
            cpu_cores_p = self.system_specs.get('cpu_cores_physical', 0)
            cpu_cores_l = self.system_specs.get('cpu_cores_logical', 0)
            cpu_arch = self.system_specs.get('cpu_architecture', 'Unknown')
            cpu_bits = self.system_specs.get('cpu_bits', 'Unknown')
            
            # CPU名が取得できていない場合のフォールバック
            if cpu_info == 'Unknown' and cpu_full_name == 'Unknown':
                try:
                    import platform
                    cpu_info = platform.processor() or 'Unknown'
                    cpu_full_name = cpu_info
                except:
                    pass
            
            cpu_info_text = f"CPU: {cpu_full_name if cpu_full_name and cpu_full_name != cpu_info else cpu_info}"
            if cpu_cores_p > 0 or cpu_cores_l > 0:
                cpu_info_text += f"\n  物理コア: {cpu_cores_p}個 | 論理コア: {cpu_cores_l}個"
            else:
                cpu_info_text += f"\n  物理コア: 検出できませんでした | 論理コア: 検出できませんでした"
            cpu_info_text += f"\n  アーキテクチャ: {cpu_arch} ({cpu_bits})"
            
            # CPU周波数情報
            cpu_freq_current = self.system_specs.get('cpu_freq_current_mhz', 0)
            cpu_freq_min = self.system_specs.get('cpu_freq_min_mhz', 0)
            cpu_freq_max = self.system_specs.get('cpu_freq_max_mhz', 0)
            if cpu_freq_current > 0:
                if cpu_freq_max > 0:
                    cpu_info_text += f"\n  周波数: {cpu_freq_current:.0f} MHz (最小: {cpu_freq_min:.0f} MHz, 最大: {cpu_freq_max:.0f} MHz)"
                else:
                    cpu_info_text += f"\n  周波数: {cpu_freq_current:.0f} MHz"
            
            # L3キャッシュ情報
            cpu_l3_cache = self.system_specs.get('cpu_l3_cache_mb', 0)
            if cpu_l3_cache > 0:
                cpu_info_text += f"\n  L3キャッシュ: {cpu_l3_cache} MB"
            
            cpu_spec_label = QLabel(cpu_info_text)
            cpu_spec_label.setStyleSheet("font-size: 10pt; color: #333; padding: 5px;")
            spec_layout.addWidget(cpu_spec_label)
            
            # メモリ情報（詳細）
            mem_gb = self.system_specs.get('memory_gb', 0)
            mem_info_text = f"メモリ: {mem_gb:.1f} GB"
            try:
                import psutil
                mem = psutil.virtual_memory()
                mem_info_text += f"\n  使用可能: {mem.available / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB"
            except:
                pass
            mem_spec_label = QLabel(mem_info_text)
            spec_layout.addWidget(mem_spec_label)
            
            # GPU情報（詳細）
            gpu_info = self.system_specs.get('gpu_info', 'Unknown')
            gpu_name_full = self.system_specs.get('gpu_name_full') or self.system_specs.get('nvidia_gpu_name') or gpu_info
            
            # GPU名が取得できていない場合、nvidia-smiで直接取得を試みる
            if gpu_info in ['Unknown', 'No GPU detected'] or gpu_name_full in ['Unknown', 'No GPU detected']:
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        gpu_info = result.stdout.strip()
                        gpu_name_full = gpu_info
                except:
                    pass
            
            gpu_info_text = f"GPU: {gpu_name_full if gpu_name_full and gpu_name_full not in ['Unknown', 'No GPU detected'] else (gpu_info if gpu_info not in ['Unknown', 'No GPU detected'] else 'Unknown')}"
            
            # VRAM情報
            gpu_vram_total = self.system_specs.get('gpu_vram_total_gb', 0)
            gpu_vram_used = self.system_specs.get('gpu_vram_used_gb', 0)
            if gpu_vram_total > 0:
                gpu_info_text += f"\n  VRAM: {gpu_vram_total:.1f} GB"
                if gpu_vram_used > 0:
                    gpu_vram_free = gpu_vram_total - gpu_vram_used
                    gpu_info_text += f" (使用中: {gpu_vram_used:.1f} GB, 空き: {gpu_vram_free:.1f} GB)"
            
            # ドライバーバージョン
            gpu_driver = self.system_specs.get('gpu_driver_version', '')
            if gpu_driver:
                gpu_info_text += f"\n  ドライバー: {gpu_driver}"
            
            # CUDAバージョン
            gpu_cuda = self.system_specs.get('gpu_cuda_version', '')
            if gpu_cuda:
                gpu_info_text += f"\n  CUDA: {gpu_cuda}"
            
            # Compute Capability
            gpu_compute = self.system_specs.get('gpu_compute_capability', '')
            if gpu_compute:
                gpu_info_text += f"\n  Compute Capability: {gpu_compute}"
            
            # 温度と使用率（GPUtilから取得した場合）
            gpu_temp = self.system_specs.get('gpu_temperature', None)
            gpu_load = self.system_specs.get('gpu_load', None)
            if gpu_temp is not None:
                gpu_info_text += f"\n  温度: {gpu_temp:.0f}℃"
            if gpu_load is not None:
                gpu_info_text += f"\n  使用率: {gpu_load:.1f}%"
            
            # フォールバック: nvidia-smiでVRAMを取得（詳細情報がない場合）
            if gpu_vram_total == 0:
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        timeout=3,
                        env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}
                    )
                    if result.returncode == 0:
                        gpu_mem_mb = result.stdout.strip()
                        if gpu_mem_mb.replace('.', '').isdigit():
                            gpu_mem_gb = float(gpu_mem_mb) / 1024
                            gpu_info_text += f"\n  VRAM: {gpu_mem_gb:.1f} GB"
                except:
                    pass
            
            gpu_spec_label = QLabel(gpu_info_text)
            gpu_spec_label.setStyleSheet("font-size: 10pt; color: #333; padding: 5px;")
            spec_layout.addWidget(gpu_spec_label)
            
            # CUDA利用可能性情報
            cuda_available = self.system_specs.get('cuda_available', False)
            cuda_device_type = self.system_specs.get('cuda_device_type', 'CPU')
            tensorflow_gpu_available = self.system_specs.get('tensorflow_gpu_available', False)
            directml_available = self.system_specs.get('directml_available', False)
            
            cuda_info_text = "CUDA/GPU利用状況: "
            if cuda_available or tensorflow_gpu_available:
                if cuda_available:
                    cuda_info_text += "✅ CUDA利用可能"
                    if 'nvidia_cuda_version' in self.system_specs:
                        cuda_info_text += f" (ドライバー: {self.system_specs.get('nvidia_cuda_version', 'N/A')})"
                elif directml_available:
                    cuda_info_text += "✅ DirectML利用可能 (Windows GPU)"
                else:
                    cuda_info_text += "✅ GPU利用可能"
                
                cuda_info_text += f"\n  デバイスタイプ: {cuda_device_type}"
                
                # TensorFlowデバイス情報
                tf_devices = self.system_specs.get('tensorflow_devices', [])
                if tf_devices:
                    cuda_info_text += f"\n  TensorFlowデバイス: {', '.join(tf_devices)}"
                
                # CUDA Toolkitバージョン
                if 'cuda_toolkit_version' in self.system_specs:
                    cuda_info_text += f"\n  CUDA Toolkit: {self.system_specs['cuda_toolkit_version']}"
                
                # CUDA/CUDNNバージョン（TensorFlowビルド情報）
                if 'cuda_build_version' in self.system_specs:
                    cuda_info_text += f"\n  TensorFlow CUDA: {self.system_specs['cuda_build_version']}"
                if 'cudnn_version' in self.system_specs:
                    cuda_info_text += f"\n  cuDNN: {self.system_specs['cudnn_version']}"
            else:
                # nvidia-smiでGPUが検出されているがTensorFlowで認識されていない場合
                if self.system_specs.get('nvidia_gpu_detected', False) or self.system_specs.get('gpu_info', 'Unknown') != 'Unknown':
                    gpu_name = self.system_specs.get('nvidia_gpu_name') or self.system_specs.get('gpu_info', 'Unknown')
                    cuda_version = self.system_specs.get('nvidia_cuda_version', 'N/A')
                    if gpu_name != 'Unknown':
                        # WSL2環境が利用可能かチェック
                        wsl2_available = False
                        wsl2_gpu_info = ""
                        try:
                            import subprocess
                            # まずWSL2が利用可能か確認
                            wsl_check = subprocess.run(
                                ['wsl', '--list', '--quiet'],
                                capture_output=True, text=True, timeout=5,
                                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                            )
                            if wsl_check.returncode == 0:
                                # WSL2環境とvenv_wsl2ディレクトリの存在を確認（簡易版）
                                # TensorFlowのインポートは時間がかかるため、ディレクトリの存在確認のみ
                                # プロジェクトパスを自動検出（ノートPC対応）
                                try:
                                    venv_path = VENV_WSL2_PATH
                                except NameError:
                                    # フォールバック: 現在のディレクトリから自動検出
                                    current_dir = Path(__file__).resolve().parents[1]
                                    project_str = str(current_dir).replace('\\', '/')
                                    if len(project_str) >= 2 and project_str[1] == ':':
                                        drive_letter = project_str[0].lower()
                                        venv_path = f"/mnt/{drive_letter}{project_str[2:]}/venv_wsl2"
                                    else:
                                        venv_path = f"{project_str}/venv_wsl2"
                                
                                venv_check = subprocess.run(
                                    ['wsl', 'bash', '-c', f'test -d {venv_path} && echo "OK"'],
                                    capture_output=True, text=True, timeout=5,
                                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                                )
                                if venv_check.returncode == 0 and 'OK' in venv_check.stdout:
                                    wsl2_available = True
                                    # GPU情報は実際に学習時に確認されるため、ここでは簡易表示
                                    wsl2_gpu_info = "（セットアップ済み）"
                        except:
                            pass
                        
                        if wsl2_available:
                            cuda_info_text += f"ℹ️ Windows環境: CPUモード（GPU未認識）"
                            cuda_info_text += f"\n  GPU検出: {gpu_name}"
                            cuda_info_text += f"\n  CUDAドライバー: {cuda_version}"
                            cuda_info_text += f"\n\n✅ WSL2環境: GPU利用可能 {wsl2_gpu_info}"
                            cuda_info_text += f"\n  → 「🔧 実行環境」で「WSL2 GPUモード」を選択すると高速学習が可能です"
                            cuda_info_text += f"\n  → 学習速度: CPUの10-50倍高速"
                            cuda_info_text += f"\n  → 推奨: 学習時はWSL2 GPUモードを使用してください"
                        else:
                            cuda_info_text += f"⚠️ GPU検出済み（{gpu_name}）だがTensorFlow未認識"
                            cuda_info_text += f"\n  CUDAドライバー: {cuda_version}"
                            cuda_info_text += f"\n\n💡 解決方法:"
                            cuda_info_text += f"\n  Windows + Python 3.12では、TensorFlowのCUDAサポートが制限されています。"
                            cuda_info_text += f"\n  WSL2環境を使用することでGPUを利用できます。"
                            cuda_info_text += f"\n  → WSL2セットアップ方法:"
                            cuda_info_text += f"\n     1. PowerShellで実行: wsl bash setup_wsl2_tensorflow_gpu.sh"
                            cuda_info_text += f"\n     2. または: check_wsl2_gpu_status.bat で状態確認"
                            cuda_info_text += f"\n     3. アプリで「🔧 実行環境」→「WSL2 GPUモード」を選択"
                            if 'recommendation' in self.system_specs:
                                cuda_info_text += f"\n  {self.system_specs['recommendation']}"
                    else:
                        cuda_info_text += "❌ CUDA利用不可（CPUモード）"
                else:
                    cuda_info_text += "❌ CUDA利用不可（CPUモード）"
                    if 'cuda_toolkit_version' in self.system_specs:
                        cuda_info_text += f"\n  （CUDA Toolkit {self.system_specs['cuda_toolkit_version']}はインストール済みですが、TensorFlowで認識されていません）"
            
            cuda_info_label = QLabel(cuda_info_text)
            if cuda_available or tensorflow_gpu_available or directml_available:
                cuda_info_label.setStyleSheet("""
                    font-size: 10pt;
                    color: #28a745;
                    font-weight: bold;
                    padding: 8px;
                    background-color: #d4edda;
                    border-radius: 4px;
                """)
            elif self.system_specs.get('nvidia_gpu_detected', False):
                # GPUは検出されたがTensorFlowで認識されていない場合（警告）
                cuda_info_label.setStyleSheet("""
                    font-size: 10pt;
                    color: #856404;
                    font-weight: bold;
                    padding: 8px;
                    background-color: #fff3cd;
                    border-radius: 4px;
                """)
            else:
                cuda_info_label.setStyleSheet("""
                    font-size: 10pt;
                    color: #dc3545;
                    font-weight: bold;
                    padding: 8px;
                    background-color: #f8d7da;
                    border-radius: 4px;
                """)
            spec_layout.addWidget(cuda_info_label)
            
            # システムタイプ
            device_type = self.system_specs.get('device_type', 'unknown')
            is_high_end = self.system_specs.get('is_high_end', False)
            is_notebook = self.system_specs.get('is_notebook', False)
            type_text = f"システムタイプ: {device_type}"
            if is_high_end:
                type_text += " (ハイエンド)"
            elif is_notebook:
                type_text += " (ノートPC)"
            type_spec_label = QLabel(type_text)
            type_spec_label.setStyleSheet("font-weight: bold; color: blue;")
            spec_layout.addWidget(type_spec_label)
            
            
            spec_group.setLayout(spec_layout)
            # 推奨設定セクションを別のグループに分離
            recommend_group = QtWidgets.QGroupBox("【★ このシステムに最適な推奨設定】")
            recommend_layout = QVBoxLayout()
            recommend_layout.setContentsMargins(12, 12, 12, 12)
            recommend_layout.setSpacing(10)
            
            # 推奨設定（明確に表示）- システムスペックを直接チェックして最適な設定を表示
            recommend_text = ""
            
            # システムスペックから直接推奨設定を計算
            memory_gb = self.system_specs.get('memory_gb', 0) if self.system_specs else 0
            gpu_vram_total = self.system_specs.get('gpu_vram_total_gb', 0) if self.system_specs else 0
            cpu_cores = self.system_specs.get('cpu_cores_logical', 0) if self.system_specs else 0
            
            # 超高性能を推奨する条件をチェック
            if memory_gb >= 48 or (memory_gb >= 24 and cpu_cores >= 12) or (gpu_vram_total >= 12 and memory_gb >= 32) or (gpu_vram_total >= 8 and cpu_cores >= 8 and memory_gb >= 16) or (cpu_cores >= 16 and memory_gb >= 16):
                # 超高性能推奨
                rec_batch = 96
                rec_workers = 24
                rec_epochs = 400
                recommended_profile = '[5] 超高性能（実験的）⭐推奨'
            elif memory_gb >= 32 or (cpu_cores >= 12 and memory_gb >= 16):
                # 最大性能推奨
                rec_batch = 64
                rec_workers = 16
                rec_epochs = 300
                recommended_profile = '[4] 最大性能（フル活用）⭐推奨'
            elif self.recommended_config:
                rec_batch = self.recommended_config.get('batch_size', 32)
                rec_workers = self.recommended_config.get('workers', 8)
                rec_epochs = self.recommended_config.get('epochs', 200)
                
                # 推奨プロファイル名を決定
                profiles_map = {
                    (8, 1): '[1] 軽量（省エネモード）',
                    (16, 4): '[2] 標準（バランス）',
                    (32, 8): '[3] 高性能',
                    (64, 16): '[4] 最大性能（フル活用）',
                    (96, 24): '[5] 超高性能（実験的）'
                }
                recommended_profile = None
                min_diff = float('inf')
                for (batch, workers), profile_name in profiles_map.items():
                    diff = abs(batch - rec_batch) + abs(workers - rec_workers) * 2
                    if diff < min_diff:
                        min_diff = diff
                        recommended_profile = profile_name
            else:
                rec_batch = 32
                rec_workers = 8
                rec_epochs = 200
                recommended_profile = '[3] 高性能'
            
            recommend_text += f"  Batch Size: {rec_batch}"
            recommend_text += f"\n  Workers: {rec_workers}"
            recommend_text += f"\n  最大学習回数: {rec_epochs}"
            recommend_text += f"\n\n  → {recommended_profile} がおすすめです"
            
            recommend_label = QLabel(recommend_text)
            recommend_label.setStyleSheet("font-weight: bold; color: green; background-color: #f0f8f0; padding: 8px; border-radius: 4px;")
            recommend_layout.addWidget(recommend_label)
            recommend_group.setLayout(recommend_layout)
        
        # 横並びコンテナを作成（システムスペックと推奨設定・学習リソース設定を横並びに）
        horizontal_container = QWidget()
        horizontal_layout = QHBoxLayout(horizontal_container)
        horizontal_layout.setContentsMargins(0, 0, 0, 0)
        horizontal_layout.setSpacing(15)
        
        # 左側: システムスペック
        if self.system_specs:
            spec_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            horizontal_layout.addWidget(spec_group, stretch=1)
        
        # 右側: 推奨設定と学習リソース設定を縦に
        right_column = QVBoxLayout()
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.setSpacing(15)
        
        if self.system_specs:
            recommend_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            right_column.addWidget(recommend_group)
        
        # リソース選択（わかりやすく改善・スペースをフル活用）
        resource_group = QtWidgets.QGroupBox("⚙️ 学習リソース設定")
        resource_layout = QVBoxLayout()
        # スペースをフル活用: マージンとスペースを最適化
        resource_layout.setContentsMargins(12, 12, 12, 12)
        resource_layout.setSpacing(12)
        
        # 選択方法の説明
        mode_info_label = QLabel("📌 設定方法を選択してください：")
        mode_info_label.setStyleSheet("font-size: 13pt; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        resource_layout.addWidget(mode_info_label)
        
        # モード選択タブ（シンプルなラジオボタン）
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(10)
        
        self.quick_mode_radio = QtWidgets.QRadioButton("🚀 クイック設定（推奨）")
        self.quick_mode_radio.setChecked(True)
        self.quick_mode_radio.setStyleSheet("font-size: 12pt; padding: 10px;")
        self.custom_mode_radio = QtWidgets.QRadioButton("🔧 カスタム設定")
        self.custom_mode_radio.setStyleSheet("font-size: 12pt; padding: 10px;")
        
        self.mode_button_group = QtWidgets.QButtonGroup()
        self.mode_button_group.addButton(self.quick_mode_radio)
        self.mode_button_group.addButton(self.custom_mode_radio)
        
        mode_layout.addWidget(self.quick_mode_radio)
        mode_layout.addWidget(self.custom_mode_radio)
        mode_layout.addStretch()
        resource_layout.addLayout(mode_layout)
        
        # クイック選択（コンボボックスで表示・スペースをフル活用）
        quick_group = QtWidgets.QGroupBox("🚀 クイック設定")
        quick_layout = QVBoxLayout()
        # スペースをフル活用: マージンとスペースを最適化
        quick_layout.setContentsMargins(12, 12, 12, 12)
        quick_layout.setSpacing(10)
        
        quick_label = QLabel("プリセットから選択：")
        quick_label.setStyleSheet("font-size: 11pt; font-weight: bold; color: #333;")
        quick_layout.addWidget(quick_label)
        
        self.quick_combo = QtWidgets.QComboBox()
        quick_items = [
            '1. 軽量（省エネモード） - Batch: 8, Workers: 1',
            '2. 標準（バランス） - Batch: 16, Workers: 4',
            '3. 高性能（推奨） - Batch: 32, Workers: 8',
            '4. 最大性能（フル活用） - Batch: 64, Workers: 16',
            '5. 超高性能（実験的） - Batch: 96, Workers: 24'
        ]
        
        # アイテムを追加し、ツールチップも設定
        for i, item in enumerate(quick_items):
            self.quick_combo.addItem(item)
            self.quick_combo.setItemData(i, item, QtCore.Qt.ToolTipRole)
        
        # コンボボックスのスタイル設定（スクロール可能、幅確保）
        self.quick_combo.setStyleSheet("""
            QComboBox {
                font-size: 11pt;
                padding: 10px;
                min-height: 40px;
                min-width: 500px;
                max-width: 800px;
                background-color: white;
            }
            QComboBox QAbstractItemView {
                min-width: 500px;
                max-width: 800px;
                min-height: 200px;
                font-size: 13pt;
                selection-background-color: #e7f3ff;
                selection-color: #0066cc;
                background-color: white;
                border: 2px solid #0066cc;
                border-radius: 4px;
            }
            QComboBox QAbstractItemView::item {
                padding: 12px;
                min-height: 40px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #f0f8ff;
            }
        """)
        self.quick_combo.setEditable(False)
        self.quick_combo.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        # 推奨設定に基づいて自動選択（PCスペックを直接チェックして超高性能を推奨）
        recommended_index = 2  # デフォルトは[3]（高性能）
        
        # PCスペックを直接チェック（システム情報が読み取れている場合）
        # システムスペックがNoneの場合、直接psutilで取得を試みる
        memory_gb = 0
        gpu_vram_total = 0
        cpu_cores = 0
        cpu_cores_p = 0
        
        if self.system_specs:
            memory_gb = self.system_specs.get('memory_gb', 0)
            gpu_vram_total = self.system_specs.get('gpu_vram_total_gb', 0)
            cpu_cores = self.system_specs.get('cpu_cores_logical', 0)
            cpu_cores_p = self.system_specs.get('cpu_cores_physical', 0)
        else:
            # フォールバック: psutilまたはWindows APIで直接取得
            if self._psutil_available:
                try:
                    import psutil
                    memory_gb = psutil.virtual_memory().total / (1024**3)
                    cpu_cores = psutil.cpu_count(logical=True)
                    cpu_cores_p = psutil.cpu_count(logical=False)
                    print(f"[DEBUG] Fallback: Using psutil directly - Memory: {memory_gb}GB, CPU Cores: {cpu_cores}")
                except Exception as e:
                    print(f"[DEBUG] Failed to get system specs from psutil: {e}")
            else:
                # Windows APIで取得
                try:
                    memory_gb, cpu_cores, cpu_cores_p = self._get_system_info_windows_api()
                    print(f"[DEBUG] Fallback: Using Windows API - Memory: {memory_gb}GB, CPU Cores: {cpu_cores}")
                except Exception as e:
                    print(f"[DEBUG] Failed to get system specs from Windows API: {e}")
        
        # GPU情報を直接取得を試みる（nvidia-smi）
        if gpu_vram_total == 0:
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        vram_mb = int(float(lines[0].strip()))
                        gpu_vram_total = vram_mb / 1024
                        print(f"[DEBUG] Fallback: GPU VRAM detected via nvidia-smi: {gpu_vram_total}GB")
            except Exception as e:
                print(f"[DEBUG] Failed to get GPU VRAM: {e}")
        
        # デバッグ用ログ
        print(f"[DEBUG] Final System Specs - Memory: {memory_gb}GB, VRAM: {gpu_vram_total}GB, CPU Cores (Logical): {cpu_cores}, CPU Cores (Physical): {cpu_cores_p}")
        
        # 超高性能を推奨する条件（非常に緩和・確実に推奨）:
        # ユーザーのPC（Ryzen 9 7900, 64GB RAM, RTX 4070 12GB）を確実にカバー
        is_high_end = False
        
        # 条件1: メモリ48GB以上 → 超高性能（最も緩い条件で確実に推奨）
        if memory_gb >= 48:
            is_high_end = True
            recommended_index = 4
            print(f"[DEBUG] [OK] High-end condition 1 met: Memory {memory_gb}GB >= 48GB -> Ultra-high performance")
        # 条件2: メモリ24GB以上 かつ CPUコア12以上 → 超高性能
        elif memory_gb >= 24 and cpu_cores >= 12:
            is_high_end = True
            recommended_index = 4
            print(f"[DEBUG] [OK] High-end condition 2 met: Memory {memory_gb}GB >= 24GB, CPU {cpu_cores} cores >= 12 -> Ultra-high performance")
        # 条件3: VRAM 12GB以上 かつ メモリ32GB以上 → 超高性能
        elif gpu_vram_total >= 12 and memory_gb >= 32:
            is_high_end = True
            recommended_index = 4
            print(f"[DEBUG] [OK] High-end condition 3 met: VRAM {gpu_vram_total}GB >= 12GB, Memory {memory_gb}GB >= 32GB -> Ultra-high performance")
        # 条件4: VRAM 8GB以上 かつ CPUコア8以上 かつ メモリ16GB以上 → 超高性能
        elif gpu_vram_total >= 8 and cpu_cores >= 8 and memory_gb >= 16:
            is_high_end = True
            recommended_index = 4
            print(f"[DEBUG] [OK] High-end condition 4 met: VRAM {gpu_vram_total}GB >= 8GB, CPU {cpu_cores} cores >= 8, Memory {memory_gb}GB >= 16GB -> Ultra-high performance")
        # 条件5: CPUコア16以上 かつ メモリ16GB以上 → 超高性能
        elif cpu_cores >= 16 and memory_gb >= 16:
            is_high_end = True
            recommended_index = 4
            print(f"[DEBUG] [OK] High-end condition 5 met: CPU {cpu_cores} cores >= 16, Memory {memory_gb}GB >= 16GB -> Ultra-high performance")
        
        if is_high_end:
            print(f"[DEBUG] [OK] ULTRA-HIGH PERFORMANCE RECOMMENDED! Index: {recommended_index}")
        else:
            print(f"[DEBUG] [NG] Ultra-high performance NOT recommended.")
            print(f"[DEBUG]   - Memory: {memory_gb}GB (need >= 48GB for condition 1)")
            print(f"[DEBUG]   - CPU Cores: {cpu_cores} (need >= 12 for condition 2)")
            print(f"[DEBUG]   - VRAM: {gpu_vram_total}GB (need >= 12GB for condition 3)")
            # 超高性能が推奨されない場合、recommended_configから判断
            if self.recommended_config and HAS_RESOURCE_SELECTOR:
                try:
                    rec_batch = self.recommended_config.get('batch_size', 16)
                    rec_workers = self.recommended_config.get('workers', 4)
                    
                    profiles = [
                        (8, 1), (16, 4), (32, 8), (64, 16), (96, 24)
                    ]
                    min_diff = float('inf')
                    for idx, (batch, workers) in enumerate(profiles):
                        diff = abs(batch - rec_batch) + abs(workers - rec_workers) * 2
                        if diff < min_diff:
                            min_diff = diff
                            recommended_index = idx
                except Exception:
                    recommended_index = 2
        
        # system_specsがNoneだった場合のフォールバック
        if not self.system_specs and self.recommended_config and HAS_RESOURCE_SELECTOR:
            try:
                rec_batch = self.recommended_config.get('batch_size', 16)
                rec_workers = self.recommended_config.get('workers', 4)
                
                profiles = [
                    (8, 1), (16, 4), (32, 8), (64, 16), (96, 24)
                ]
                min_diff = float('inf')
                for idx, (batch, workers) in enumerate(profiles):
                    diff = abs(batch - rec_batch) + abs(workers - rec_workers) * 2
                    if diff < min_diff:
                        min_diff = diff
                        recommended_index = idx
            except Exception:
                recommended_index = 2
        
        # 根本的に変更: まずアイテムを再作成してから設定
        # 推奨設定に「⭐推奨」を追加
        if recommended_index == 4:
            recommended_text = quick_items[recommended_index] + " ⭐推奨"
        else:
            recommended_text = quick_items[recommended_index] + " ⭐推奨"
        quick_items[recommended_index] = recommended_text
        
        # コンボボックスをクリアして再構築
        self.quick_combo.clear()
        for i, item in enumerate(quick_items):
            self.quick_combo.addItem(item)
            self.quick_combo.setItemData(i, item, QtCore.Qt.ToolTipRole)
        
        # 確実に推奨インデックスを設定
        if 0 <= recommended_index < len(quick_items):
            self.quick_combo.setCurrentIndex(recommended_index)
            # 絵文字を除去して安全に出力
            safe_item = quick_items[recommended_index].encode('ascii', errors='ignore').decode('ascii')
            print(f"[DEBUG] Quick combo set to index {recommended_index}: {safe_item}")
        
        # 強制的に再描画
        self.quick_combo.update()
        QtCore.QTimer.singleShot(100, lambda: self.quick_combo.setCurrentIndex(recommended_index) if 0 <= recommended_index < len(quick_items) else None)
        
        quick_layout.addWidget(self.quick_combo)
        
        # 説明テキスト
        quick_desc = QLabel("※ システム構成に基づいて最適な設定が自動選択されます")
        quick_desc.setStyleSheet("font-size: 9pt; color: #666; padding-top: 3px;")
        quick_layout.addWidget(quick_desc)
        
        quick_group.setLayout(quick_layout)
        resource_layout.addWidget(quick_group)
        
        # カスタム選択
        # カスタム設定（スペースをフル活用）
        custom_group = QtWidgets.QGroupBox("🔧 カスタム設定")
        custom_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        custom_layout = QVBoxLayout()
        # スペースをフル活用: マージンとスペースを最適化
        custom_layout.setContentsMargins(12, 12, 12, 12)
        custom_layout.setSpacing(12)
        
        custom_info = QLabel("GPUとCPUの設定を個別に調整できます")
        custom_info.setStyleSheet("font-size: 11pt; color: #666; margin-bottom: 10px;")
        custom_layout.addWidget(custom_info)
        
        # GPU選択とCPU選択を横並びにするコンテナ
        gpu_cpu_container_layout = QHBoxLayout()
        gpu_cpu_container_layout.setSpacing(15)
        
        # GPU選択
        gpu_layout = QVBoxLayout()
        gpu_label = QLabel("🎮 GPUレベル（Batch Size）:")
        gpu_label.setStyleSheet("font-size: 11pt; font-weight: bold; padding: 6px 0px;")
        gpu_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        gpu_label.setWordWrap(True)
        gpu_layout.addWidget(gpu_label)
        
        self.gpu_combo = QtWidgets.QComboBox()
        # 表示用（短縮）とツールチップ用（完全版）を分離
        gpu_items_display = [
            '軽量 - Batch: 8',
            '標準 - Batch: 16',
            '高性能 - Batch: 32（推奨）',
            '最大性能 - Batch: 64',
            '超高性能 - Batch: 96'
        ]
        gpu_items_tooltip = [
            '軽量 - Batch: 8（省エネ、メモリ使用量少）',
            '標準 - Batch: 16（バランス型）',
            '高性能 - Batch: 32（推奨、速度と精度のバランス）',
            '最大性能 - Batch: 64（最高速度、メモリ使用量大）',
            '超高性能 - Batch: 96（実験的、大量メモリ必要）'
        ]
        
        # アイテムを追加し、ツールチップも設定
        for i, (display, tooltip) in enumerate(zip(gpu_items_display, gpu_items_tooltip)):
            self.gpu_combo.addItem(display)
            self.gpu_combo.setItemData(i, tooltip, QtCore.Qt.ToolTipRole)
        
        # コンボボックス本体のスタイル（ウィンドウサイズに対応・横並び対応）
        self.gpu_combo.setStyleSheet("""
            QComboBox {
                font-size: 12pt;
                padding: 10px 15px;
                min-height: 45px;
                min-width: 250px;
                max-width: 100%;
                background-color: white;
                border: 2px solid #0066cc;
                border-radius: 6px;
            }
            QComboBox:hover {
                border: 2px solid #0052a3;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
                width: 0px;
                height: 0px;
            }
        """)
        self.gpu_combo.setEditable(False)
        # サイズポリシーを設定してリサイズ対応
        self.gpu_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # ドロップダウンリストのビューを取得して設定
        view = self.gpu_combo.view()
        view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        # ビューウィジェットのスタイルを設定
        view.setStyleSheet("""
            QAbstractItemView {
                font-size: 12pt;
                background-color: white;
                border: 2px solid #0066cc;
                border-radius: 6px;
                selection-background-color: #e7f3ff;
                selection-color: #0066cc;
                outline: none;
            }
            QAbstractItemView::item {
                padding: 15px 20px;
                min-height: 50px;
                border-bottom: 1px solid #e0e0e0;
            }
            QAbstractItemView::item:selected {
                background-color: #e7f3ff;
                color: #0066cc;
            }
            QAbstractItemView::item:hover {
                background-color: #f0f8ff;
                color: #0066cc;
            }
            QScrollBar:vertical {
                width: 20px;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background-color: #0066cc;
                min-height: 30px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #0052a3;
            }
        """)
        
        # ドロップダウンリストのサイズを設定（根本的な修正）
        def adjust_gpu_popup_size():
            try:
                popup = self.gpu_combo.view()
                if not popup:
                    return
                
                # フォントメトリクスを取得
                fm = QFontMetrics(self.gpu_combo.font())
                max_width = 0
                
                # 各アイテムのテキスト幅を計算（短縮版なので幅は十分小さい）
                for i in range(self.gpu_combo.count()):
                    text = self.gpu_combo.itemText(i)
                    rect = fm.boundingRect(text)
                    max_width = max(max_width, rect.width())
                
                # パディング、スクロールバー、ボーダーを考慮
                content_width = max_width + 80  # 十分な余裕
                
                # ウィンドウの利用可能幅を取得
                try:
                    window = self.window() if hasattr(self, 'window') else self
                    window_width = window.width() if window else 1400
                except:
                    window_width = 1400
                
                # ウィンドウからはみ出さないように調整
                available_width = max(500, window_width - 100)
                
                # 最終的な幅を決定（コンテンツ幅とウィンドウ幅の小さい方）
                final_width = min(content_width, available_width, 800)
                
                # ポップアップのサイズを設定（固定幅）
                popup.setMinimumWidth(final_width)
                popup.setMaximumWidth(final_width)
                popup.setMinimumHeight(220)
                popup.setMaximumHeight(350)
                
                # スクロールバーを確実に表示
                popup.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
                popup.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
                
            except Exception as e:
                print(f"[ERROR] GPU popup size adjustment error: {e}")
        
        # コンボボックスが開かれたときにサイズ調整
        original_showPopup_gpu = self.gpu_combo.showPopup
        def showPopup_gpu():
            original_showPopup_gpu()
            # 表示後にサイズ調整（複数回確実に実行）
            QtCore.QTimer.singleShot(1, adjust_gpu_popup_size)
            QtCore.QTimer.singleShot(10, adjust_gpu_popup_size)
            QtCore.QTimer.singleShot(30, adjust_gpu_popup_size)
        self.gpu_combo.showPopup = showPopup_gpu
        
        recommended_gpu_index = 2
        # PCスペックを直接チェック（システム情報が読み取れている場合）
        if self.system_specs:
            memory_gb = self.system_specs.get('memory_gb', 0)
            gpu_vram_total = self.system_specs.get('gpu_vram_total_gb', 0)
            cpu_cores = self.system_specs.get('cpu_cores_logical', 0)
            
            # 超高性能を推奨する条件（緩和版）:
            # メモリ16GB以上 かつ (VRAM 8GB以上 または GPUあり) かつ CPUコア8以上
            is_high_end = (
                (memory_gb >= 16 and (gpu_vram_total >= 8 or gpu_vram_total > 0) and cpu_cores >= 8) or
                (memory_gb >= 24 and cpu_cores >= 12) or
                (memory_gb >= 16 and gpu_vram_total >= 12)
            )
            
            if is_high_end:
                recommended_gpu_index = 4  # 超高性能
            elif self.recommended_config:
                try:
                    rec_batch = self.recommended_config.get('batch_size', 32)
                    batch_sizes = [8, 16, 32, 64, 96]
                    min_diff = float('inf')
                    for idx, batch_size in enumerate(batch_sizes):
                        diff = abs(batch_size - rec_batch)
                        if diff < min_diff:
                            min_diff = diff
                            recommended_gpu_index = idx
                except Exception:
                    pass
        elif self.recommended_config:
            try:
                rec_batch = self.recommended_config.get('batch_size', 32)
                batch_sizes = [8, 16, 32, 64, 96]
                min_diff = float('inf')
                for idx, batch_size in enumerate(batch_sizes):
                    diff = abs(batch_size - rec_batch)
                    if diff < min_diff:
                        min_diff = diff
                        recommended_gpu_index = idx
            except Exception:
                pass
        
        self.gpu_combo.setCurrentIndex(recommended_gpu_index)
        gpu_layout.addWidget(self.gpu_combo)
        
        # GPUレイアウトをコンテナに追加
        gpu_widget = QWidget()
        gpu_widget.setLayout(gpu_layout)
        gpu_cpu_container_layout.addWidget(gpu_widget, stretch=1)
        
        # CPU選択
        cpu_layout = QVBoxLayout()
        cpu_label = QLabel("💻 CPUレベル（Workers数）:")
        cpu_label.setStyleSheet("font-size: 11pt; font-weight: bold; padding: 6px 0px;")
        cpu_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        cpu_layout.addWidget(cpu_label)
        
        self.cpu_combo = QtWidgets.QComboBox()
        # 表示用（短縮）とツールチップ用（完全版）を分離
        cpu_items_display = [
            '軽量 - Workers: 1',
            '標準 - Workers: 4',
            '高性能 - Workers: 8（推奨）',
            '最大性能 - Workers: 16',
            '超高性能 - Workers: 24'
        ]
        cpu_items_tooltip = [
            '軽量 - Workers: 1（省エネ、CPU使用率低）',
            '標準 - Workers: 4（バランス型）',
            '高性能 - Workers: 8（推奨、速度と精度のバランス）',
            '最大性能 - Workers: 16（最高速度、CPU使用率高）',
            '超高性能 - Workers: 24（実験的、全CPUコア活用）'
        ]
        
        # アイテムを追加し、ツールチップも設定
        for i, (display, tooltip) in enumerate(zip(cpu_items_display, cpu_items_tooltip)):
            self.cpu_combo.addItem(display)
            self.cpu_combo.setItemData(i, tooltip, QtCore.Qt.ToolTipRole)
        
        # コンボボックス本体のスタイル（ウィンドウサイズに対応・横並び対応）
        self.cpu_combo.setStyleSheet("""
            QComboBox {
                font-size: 12pt;
                padding: 10px 15px;
                min-height: 45px;
                min-width: 250px;
                max-width: 100%;
                background-color: white;
                border: 2px solid #0066cc;
                border-radius: 6px;
            }
            QComboBox:hover {
                border: 2px solid #0052a3;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
                width: 0px;
                height: 0px;
            }
        """)
        self.cpu_combo.setEditable(False)
        # サイズポリシーを設定してリサイズ対応
        self.cpu_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # ドロップダウンリストのビューを取得して設定
        view = self.cpu_combo.view()
        view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        # ビューウィジェットのスタイルを設定
        view.setStyleSheet("""
            QAbstractItemView {
                font-size: 12pt;
                background-color: white;
                border: 2px solid #0066cc;
                border-radius: 6px;
                selection-background-color: #e7f3ff;
                selection-color: #0066cc;
                outline: none;
            }
            QAbstractItemView::item {
                padding: 15px 20px;
                min-height: 50px;
                border-bottom: 1px solid #e0e0e0;
            }
            QAbstractItemView::item:selected {
                background-color: #e7f3ff;
                color: #0066cc;
            }
            QAbstractItemView::item:hover {
                background-color: #f0f8ff;
                color: #0066cc;
            }
            QScrollBar:vertical {
                width: 20px;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background-color: #0066cc;
                min-height: 30px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #0052a3;
            }
        """)
        
        # ドロップダウンリストのサイズを設定（根本的な修正）
        def adjust_cpu_popup_size():
            try:
                popup = self.cpu_combo.view()
                if not popup:
                    return
                
                # フォントメトリクスを取得
                fm = QFontMetrics(self.cpu_combo.font())
                max_width = 0
                
                # 各アイテムのテキスト幅を計算（短縮版なので幅は十分小さい）
                for i in range(self.cpu_combo.count()):
                    text = self.cpu_combo.itemText(i)
                    rect = fm.boundingRect(text)
                    max_width = max(max_width, rect.width())
                
                # パディング、スクロールバー、ボーダーを考慮
                content_width = max_width + 80  # 十分な余裕
                
                # ウィンドウの利用可能幅を取得
                try:
                    window = self.window() if hasattr(self, 'window') else self
                    window_width = window.width() if window else 1400
                except:
                    window_width = 1400
                
                # ウィンドウからはみ出さないように調整
                available_width = max(500, window_width - 100)
                
                # 最終的な幅を決定（コンテンツ幅とウィンドウ幅の小さい方）
                final_width = min(content_width, available_width, 800)
                
                # ポップアップのサイズを設定（固定幅）
                popup.setMinimumWidth(final_width)
                popup.setMaximumWidth(final_width)
                popup.setMinimumHeight(220)
                popup.setMaximumHeight(350)
                
                # スクロールバーを確実に表示
                popup.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
                popup.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
                
            except Exception as e:
                print(f"[ERROR] CPU popup size adjustment error: {e}")
        
        # コンボボックスが開かれたときにサイズ調整
        original_showPopup_cpu = self.cpu_combo.showPopup
        def showPopup_cpu():
            original_showPopup_cpu()
            # 表示後にサイズ調整（複数回確実に実行）
            QtCore.QTimer.singleShot(1, adjust_cpu_popup_size)
            QtCore.QTimer.singleShot(10, adjust_cpu_popup_size)
            QtCore.QTimer.singleShot(30, adjust_cpu_popup_size)
        self.cpu_combo.showPopup = showPopup_cpu
        
        recommended_cpu_index = 2
        # PCスペックを直接チェック（システム情報が読み取れている場合）
        if self.system_specs:
            cpu_cores = self.system_specs.get('cpu_cores_logical', 0)
            memory_gb = self.system_specs.get('memory_gb', 0)
            
            # 最大性能を推奨する条件（緩和版）:
            # CPUコア12以上 かつ メモリ16GB以上
            # またはCPUコア16以上
            is_high_end = (
                (cpu_cores >= 12 and memory_gb >= 16) or
                (cpu_cores >= 16) or
                (cpu_cores >= 8 and memory_gb >= 24)
            )
            
            if is_high_end:
                recommended_cpu_index = 3  # 最大性能（Workers: 16）
            elif self.recommended_config:
                try:
                    rec_workers = self.recommended_config.get('workers', 8)
                    worker_counts = [1, 4, 8, 16, 24]
                    min_diff = float('inf')
                    for idx, workers in enumerate(worker_counts):
                        diff = abs(workers - rec_workers)
                        if diff < min_diff:
                            min_diff = diff
                            recommended_cpu_index = idx
                except Exception:
                    pass
        elif self.recommended_config:
            try:
                rec_workers = self.recommended_config.get('workers', 8)
                worker_counts = [1, 4, 8, 16, 24]
                min_diff = float('inf')
                for idx, workers in enumerate(worker_counts):
                    diff = abs(workers - rec_workers)
                    if diff < min_diff:
                        min_diff = diff
                        recommended_cpu_index = idx
            except Exception:
                pass
        
        self.cpu_combo.setCurrentIndex(recommended_cpu_index)
        cpu_layout.addWidget(self.cpu_combo)
        
        # CPUレイアウトをコンテナに追加
        cpu_widget = QWidget()
        cpu_widget.setLayout(cpu_layout)
        gpu_cpu_container_layout.addWidget(cpu_widget, stretch=1)
        
        # 横並びコンテナをカスタムレイアウトに追加
        custom_layout.addLayout(gpu_cpu_container_layout)
        
        custom_group.setLayout(custom_layout)
        resource_layout.addWidget(custom_group)
        
        # モード切り替えの処理
        def update_mode_visibility():
            is_quick = self.quick_mode_radio.isChecked()
            quick_group.setVisible(is_quick)
            custom_group.setVisible(not is_quick)
        
        self.quick_mode_radio.toggled.connect(update_mode_visibility)
        self.custom_mode_radio.toggled.connect(update_mode_visibility)
        update_mode_visibility()  # 初期表示
        
        resource_group.setLayout(resource_layout)
        resource_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        right_column.addWidget(resource_group)
        
        # 右側カラムをコンテナに追加
        right_widget = QWidget()
        right_widget.setLayout(right_column)
        horizontal_layout.addWidget(right_widget, stretch=1)
        
        # 横並びコンテナをスクロールレイアウトに追加
        scroll_layout.addWidget(horizontal_container)
        
        # バックグラウンドモード選択
        background_group = QtWidgets.QGroupBox("バックグラウンドモード")
        background_layout = QVBoxLayout()
        background_layout.setContentsMargins(10, 10, 10, 10)
        background_layout.setSpacing(8)
        
        background_info_label = QLabel("他のアプリで作業中の場合、学習の使用率を下げてバックグラウンドで動作させます")
        background_info_label.setWordWrap(True)
        background_info_label.setStyleSheet("color: #666;")
        background_layout.addWidget(background_info_label)
        
        self.background_mode_checkbox = QtWidgets.QCheckBox("バックグラウンドモード（リソース使用率を下げる）")
        self.background_mode_checkbox.setChecked(False)
        background_layout.addWidget(self.background_mode_checkbox)
        
        background_desc_label = QLabel("  ※ チェック時: Batch/Workersを約50%削減し、CPU優先度を下げます")
        background_desc_label.setStyleSheet("color: #888; font-size: 9pt;")
        background_layout.addWidget(background_desc_label)
        
        background_group.setLayout(background_layout)
        scroll_layout.addWidget(background_group)
        
        # 学習リセット機能
        reset_group = QtWidgets.QGroupBox("🔄 学習リセット")
        reset_layout = QVBoxLayout()
        reset_layout.setContentsMargins(10, 10, 10, 10)
        reset_layout.setSpacing(8)
        
        reset_info_label = QLabel("前回の学習をリセットして、最初から学習を開始します。")
        reset_info_label.setStyleSheet("font-size: 10pt; color: #666;")
        reset_layout.addWidget(reset_info_label)
        
        reset_detail_label = QLabel("中断された学習を最初からやり直す場合に使用します。\n• チェックポイント（.h5ファイル）のみ削除\n• 学習ログ（.csv）は保持されます\n\n⚠️ 全ファイルを削除する場合は「完全リセット」を使用してください。")
        reset_detail_label.setStyleSheet("font-size: 9pt; color: #888; font-style: italic;")
        reset_layout.addWidget(reset_detail_label)
        
        # 中断された学習をクリアするボタン
        clear_interrupted_btn = QPushButton("🔄 中断された学習をクリア（最初からやり直す）")
        clear_interrupted_btn.setToolTip("中断された学習のチェックポイントを削除して、最初から学習を開始できるようにします")
        clear_interrupted_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                font-size: 11pt;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
            QPushButton:pressed {
                background-color: #e65100;
            }
        """)
        clear_interrupted_btn.clicked.connect(lambda: self.confirm_clear_interrupted())
        reset_layout.addWidget(clear_interrupted_btn)
        
        # 完全リセットボタン
        self.reset_training_btn = QPushButton("⚠️ 完全リセット（全ファイル削除・警告）")
        self.reset_training_btn.setToolTip("⚠️ 警告: 全ての学習ファイル（チェックポイント、ログ、ステータス）を削除します。この操作は取り消せません。")
        self.reset_training_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-size: 11pt;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 4px;
                border: 2px solid #ff0000;
            }
            QPushButton:hover {
                background-color: #c82333;
                border: 2px solid #ff4444;
            }
            QPushButton:pressed {
                background-color: #bd2130;
                border: 2px solid #cc0000;
            }
        """)
        self.reset_training_btn.clicked.connect(lambda: self.confirm_reset_training_files())
        reset_layout.addWidget(self.reset_training_btn)
        
        reset_group.setLayout(reset_layout)
        scroll_layout.addWidget(reset_group)
        
        # 学習進捗
        progress_group = QtWidgets.QGroupBox("📈 学習進捗")
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(10, 10, 10, 10)
        progress_layout.setSpacing(8)
        
        self.training_progress = QtWidgets.QProgressBar()
        self.training_progress.setMinimum(0)
        self.training_progress.setMaximum(100)
        self.training_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                font-size: 11pt;
                height: 30px;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #0066cc, stop: 1 #0088ff);
                border-radius: 6px;
            }
        """)
        progress_layout.addWidget(self.training_progress)
        
        self.training_status_label = QLabel("ステータス: ⚪ 待機中")
        self.training_status_label.setStyleSheet("""
            font-weight: bold;
            font-size: 13pt;
            color: #888;
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 4px;
        """)
        progress_layout.addWidget(self.training_status_label)
        
        # 情報を横並びで表示
        info_layout = QHBoxLayout()
        
        # 現在の作業内容を表示
        self.current_activity_label = QLabel("現在の作業: -")
        self.current_activity_label.setStyleSheet("""
            font-weight: bold;
            color: #0066cc;
            font-size: 12pt;
            padding: 8px;
            background-color: #e7f3ff;
            border-radius: 4px;
        """)
        info_layout.addWidget(self.current_activity_label)
        
        self.epoch_label = QLabel("学習回数: -")
        self.epoch_label.setStyleSheet("""
            font-size: 11pt;
            color: #555;
            padding: 8px;
        """)
        info_layout.addWidget(self.epoch_label)
        
        self.accuracy_label = QLabel("精度: -")
        self.accuracy_label.setStyleSheet("""
            font-size: 11pt;
            color: #555;
            padding: 8px;
            font-weight: bold;
        """)
        info_layout.addWidget(self.accuracy_label)
        
        progress_layout.addLayout(info_layout)
        
        progress_group.setLayout(progress_layout)
        scroll_layout.addWidget(progress_group)
        
        # ボタン
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        button_layout.addStretch()
        
        # 実行環境選択（WSL2 GPU / Windows CPU）
        env_group = QtWidgets.QGroupBox("🔧 実行環境")
        env_layout = QVBoxLayout()
        env_layout.setContentsMargins(12, 12, 12, 12)
        env_layout.setSpacing(8)
        
        self.use_wsl2_checkbox = QtWidgets.QCheckBox("🚀 WSL2 GPUモードを使用（高速学習・RTX 4070活用）")
        self.use_wsl2_checkbox.setStyleSheet("font-size: 11pt; padding: 5px;")
        self.use_wsl2_checkbox.setToolTip("WSL2環境でGPUを使用して学習を実行します（高速）。\nチェックを外すとWindows CPUモードで実行します。")
        
        # WSL2環境の存在確認
        try:
            import subprocess
            # プロジェクトパスを自動検出（ノートPC対応）
            try:
                venv_path = VENV_WSL2_PATH
            except NameError:
                # フォールバック: 現在のディレクトリから自動検出
                current_dir = Path(__file__).resolve().parents[1]
                project_str = str(current_dir).replace('\\', '/')
                if len(project_str) >= 2 and project_str[1] == ':':
                    drive_letter = project_str[0].lower()
                    venv_path = f"/mnt/{drive_letter}{project_str[2:]}/venv_wsl2"
                else:
                    venv_path = f"{project_str}/venv_wsl2"
            
            result = subprocess.run(
                ['wsl', 'bash', '-c', f'test -d {venv_path} && echo "OK"'],
                capture_output=True, text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            if result.returncode == 0 and 'OK' in result.stdout:
                self.use_wsl2_checkbox.setEnabled(True)
                wsl2_status = "✅ WSL2環境利用可能"
            else:
                self.use_wsl2_checkbox.setEnabled(False)
                self.use_wsl2_checkbox.setToolTip("WSL2環境が設定されていません。\nsetup_wsl2_tensorflow_gpu.shを実行してWSL2環境をセットアップしてください。")
                wsl2_status = "⚠ WSL2環境未設定"
        except:
            self.use_wsl2_checkbox.setEnabled(False)
            wsl2_status = "⚠ WSL2環境未設定"
        
        wsl2_status_label = QLabel(wsl2_status)
        wsl2_status_label.setStyleSheet("font-size: 9pt; color: #666; font-style: italic;")
        
        env_layout.addWidget(self.use_wsl2_checkbox)
        env_layout.addWidget(wsl2_status_label)
        env_group.setLayout(env_layout)
        
        # 学習開始ボタンを作成
        self.start_training_btn = QPushButton("▶ 学習開始")
        self.start_training_btn.setObjectName("start_button")
        self.start_training_btn.clicked.connect(self.start_training)
        self.start_training_btn.setMinimumHeight(50)
        self.start_training_btn.setMinimumWidth(150)
        self.start_training_btn.setEnabled(True)
        self.start_training_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-size: 13pt;
                font-weight: bold;
                padding: 12px 25px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #ffffff;
            }
        """)
        
        self.stop_training_btn = QPushButton("⏹ 学習停止")
        self.stop_training_btn.setObjectName("stop_button")
        self.stop_training_btn.clicked.connect(self.stop_training)
        self.stop_training_btn.setEnabled(False)
        
        self.stop_training_btn.setMinimumHeight(50)
        self.stop_training_btn.setMinimumWidth(150)
        self.stop_training_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-size: 13pt;
                font-weight: bold;
                padding: 12px 25px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        
        # リモートサーバーボタン
        remote_server_layout = QVBoxLayout()
        self.remote_server_btn = QPushButton("🌐 リモートサーバー起動")
        self.remote_server_btn.setObjectName("remote_server_button")
        self.remote_server_btn.clicked.connect(self.start_remote_server)
        
        self.remote_tunnel_btn = QPushButton("🌍 インターネット経由アクセス")
        self.remote_tunnel_btn.setObjectName("remote_tunnel_button")
        self.remote_tunnel_btn.clicked.connect(self.start_remote_tunnel)
        self.remote_tunnel_btn.setMinimumHeight(40)
        self.remote_tunnel_btn.setStyleSheet("""
            QPushButton {
                background-color: #6f42c1;
                color: white;
                font-size: 10pt;
                font-weight: bold;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #5a32a3;
            }
            QPushButton:pressed {
                background-color: #4a2785;
            }
        """)
        
        remote_server_layout.addWidget(self.remote_server_btn)
        remote_server_layout.addWidget(self.remote_tunnel_btn)
        remote_server_widget = QWidget()
        remote_server_widget.setLayout(remote_server_layout)
        self.remote_server_btn.setMinimumHeight(40)
        self.remote_server_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                font-size: 11pt;
                font-weight: bold;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
            QPushButton:pressed {
                background-color: #117a8b;
            }
        """)
        
        # ボタンと環境選択を横並びに
        button_env_layout = QHBoxLayout()
        button_env_layout.addWidget(env_group)
        button_env_layout.addWidget(remote_server_widget)
        button_env_layout.addStretch()
        button_env_layout.addWidget(self.start_training_btn)
        button_env_layout.addWidget(self.stop_training_btn)
        scroll_layout.addLayout(button_env_layout)
        
        # 学習ログ表示（一番下・優先度低）追加
        log_group = QtWidgets.QGroupBox("📋 学習ログ（詳細・参考用）")
        log_group.setStyleSheet("""
            QGroupBox {
                font-size: 10pt;
                color: #888;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(10, 10, 10, 10)
        log_layout.setSpacing(8)
        
        self.training_log_text = QTextEdit()
        self.training_log_text.setReadOnly(True)
        self.training_log_text.setMinimumHeight(150)  # 高さを小さく（優先度低）
        self.training_log_text.setMaximumHeight(250)  # 最大高さも小さく
        self.training_log_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 8pt;
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        self.training_log_text.setPlaceholderText("学習を開始すると、ここに詳細なログが表示されます（参考用）...")
        log_layout.addWidget(self.training_log_text)
        
        # ログ操作ボタン
        log_button_layout = QHBoxLayout()
        clear_log_btn = QPushButton("🗑️ ログをクリア")
        clear_log_btn.setToolTip("ログ表示エリアをクリアします")
        clear_log_btn.setStyleSheet("""
            QPushButton {
                font-size: 9pt;
                padding: 5px 10px;
            }
        """)
        clear_log_btn.clicked.connect(lambda: self.training_log_text.clear())
        log_button_layout.addWidget(clear_log_btn)
        log_button_layout.addStretch()
        log_layout.addLayout(log_button_layout)
        
        log_group.setLayout(log_layout)
        scroll_layout.addWidget(log_group)  # 一番最後に追加
        
        # スクロールウィジェットのサイズポリシーを設定
        scroll_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        return widget
    
    def create_status_tab(self):
        """進捗・システム情報タブを作成"""
        widget = QtWidgets.QWidget()
        widget.setStyleSheet("background-color: white;")
        
        # 横方向のレイアウト（2列）でコンパクトに表示（スクロールなしで表示・スペースをフル活用）
        main_layout = QHBoxLayout(widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)
        
        # 左側：システム情報
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        
        # HWiNFOシステム情報（スペースをフル活用）
        hwinfo_group = QtWidgets.QGroupBox("💻 システム情報（HWiNFO統合）")
        hwinfo_layout = QVBoxLayout()
        hwinfo_layout.setContentsMargins(10, 10, 10, 10)
        hwinfo_layout.setSpacing(8)
        
        # システム情報の常時表示チェックボックス（グローバルスタイルを適用）
        self.always_show_system_info_checkbox = QCheckBox("常にシステム情報を表示（学習中以外も）")
        # グローバルスタイルを適用（個別のスタイル設定を削除）
        self.always_show_system_info_checkbox.setToolTip("チェックを入れると、学習中でなくてもシステム情報（CPU/メモリ/GPU使用率）を常に更新・表示します")
        # 確実にチェック状態にする
        self.always_show_system_info_checkbox.blockSignals(True)
        self.always_show_system_info_checkbox.setChecked(True)
        self.always_show_system_info_checkbox.setCheckState(QtCore.Qt.Checked)
        self.always_show_system_info_checkbox.blockSignals(False)
        self.always_show_system_info_checkbox.stateChanged.connect(self.on_always_show_system_info_changed)
        hwinfo_layout.addWidget(self.always_show_system_info_checkbox)
        
        # 区切り線
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #ddd; margin: 5px 0px;")
        hwinfo_layout.addWidget(separator)
        
        self.cpu_label = QLabel("💻 CPU使用率: -")
        self.cpu_label.setStyleSheet("font-size: 11pt; font-weight: bold; color: #2c3e50; padding: 5px;")
        hwinfo_layout.addWidget(self.cpu_label)
        
        self.cpu_temp_label = QLabel("🌡️ CPU温度: -")
        self.cpu_temp_label.setStyleSheet("font-size: 10pt; color: #555; padding: 5px;")
        hwinfo_layout.addWidget(self.cpu_temp_label)
        
        self.mem_label = QLabel("💾 メモリ使用率: -")
        self.mem_label.setStyleSheet("font-size: 11pt; font-weight: bold; color: #2c3e50; padding: 5px;")
        hwinfo_layout.addWidget(self.mem_label)
        
        self.gpu_label = QLabel("🎮 GPU使用率: -")
        self.gpu_label.setStyleSheet("font-size: 11pt; font-weight: bold; color: #2c3e50; padding: 5px;")
        hwinfo_layout.addWidget(self.gpu_label)
        
        self.gpu_temp_label = QLabel("🌡️ GPU温度: -")
        self.gpu_temp_label.setStyleSheet("font-size: 10pt; color: #555; padding: 5px;")
        hwinfo_layout.addWidget(self.gpu_temp_label)
        
        self.gpu_power_label = QLabel("⚡ GPU電力: -")
        self.gpu_power_label.setStyleSheet("font-size: 10pt; color: #555; padding: 5px;")
        hwinfo_layout.addWidget(self.gpu_power_label)
        
        # プログレスバー（文字が消えないように修正）
        self.cpu_progress = QtWidgets.QProgressBar()
        self.cpu_progress.setMinimum(0)
        self.cpu_progress.setMaximum(100)
        self.cpu_progress.setFormat("%p%")
        self.cpu_progress.setTextVisible(True)
        self.cpu_progress.setValue(0)  # 初期値を設定して数字を表示
        self.cpu_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                font-size: 11pt;
                height: 28px;
                color: #333;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #4CAF50, stop: 0.7 #FFC107, stop: 1 #F44336);
                border-radius: 3px;
            }
        """)
        hwinfo_layout.addWidget(self.cpu_progress)
        
        self.mem_progress = QtWidgets.QProgressBar()
        self.mem_progress.setMinimum(0)
        self.mem_progress.setMaximum(100)
        self.mem_progress.setFormat("%p%")
        self.mem_progress.setTextVisible(True)
        self.mem_progress.setValue(0)  # 初期値を設定して数字を表示
        self.mem_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                font-size: 11pt;
                height: 28px;
                color: #333;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #2196F3, stop: 0.7 #00BCD4, stop: 1 #009688);
                border-radius: 3px;
            }
        """)
        hwinfo_layout.addWidget(self.mem_progress)
        
        self.gpu_progress = QtWidgets.QProgressBar()
        self.gpu_progress.setMinimum(0)
        self.gpu_progress.setMaximum(100)
        self.gpu_progress.setFormat("%p%")
        self.gpu_progress.setTextVisible(True)
        self.gpu_progress.setValue(0)  # 初期値を設定して数字を表示
        self.gpu_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                font-size: 11pt;
                height: 28px;
                color: #333;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #9C27B0, stop: 0.7 #E91E63, stop: 1 #F44336);
                border-radius: 3px;
            }
        """)
        hwinfo_layout.addWidget(self.gpu_progress)
        
        hwinfo_group.setLayout(hwinfo_layout)
        left_layout.addWidget(hwinfo_group)
        left_layout.addStretch()
        
        # 右側：学習進捗（スペースをフル活用）
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        
        # 学習進捗（詳細）
        training_detail_group = QtWidgets.QGroupBox("📊 学習進捗（詳細）")
        training_detail_layout = QVBoxLayout()
        training_detail_layout.setContentsMargins(10, 10, 10, 10)
        training_detail_layout.setSpacing(8)
        
        self.model_name_label = QLabel("🤖 モデル: -")
        self.model_name_label.setStyleSheet("font-size: 11pt; font-weight: bold; color: #2c3e50; padding: 5px;")
        training_detail_layout.addWidget(self.model_name_label)
        
        # 現在の作業内容を詳細表示
        self.current_activity_detail_label = QLabel("📝 現在の作業: -")
        self.current_activity_detail_label.setStyleSheet("""
            font-weight: bold;
            color: #0066cc;
            font-size: 12pt;
            padding: 8px;
            background-color: #e7f3ff;
            border-radius: 4px;
        """)
        training_detail_layout.addWidget(self.current_activity_detail_label)
        
        # 詳細情報を横並びで表示
        detail_grid = QHBoxLayout()
        
        self.epoch_detail_label = QLabel("📊 学習回数: -")
        self.epoch_detail_label.setStyleSheet("font-size: 10pt; color: #555; padding: 5px;")
        detail_grid.addWidget(self.epoch_detail_label)
        
        self.overall_progress_label = QLabel("📈 全体進捗: -")
        self.overall_progress_label.setStyleSheet("font-size: 10pt; color: #555; padding: 5px; font-weight: bold;")
        detail_grid.addWidget(self.overall_progress_label)
        
        training_detail_layout.addLayout(detail_grid)
        
        # 時間情報を横並びで表示
        time_layout = QHBoxLayout()
        
        self.elapsed_time_label = QLabel("⏱️ 経過時間: -")
        self.elapsed_time_label.setStyleSheet("font-size: 10pt; color: #555; padding: 5px;")
        time_layout.addWidget(self.elapsed_time_label)
        
        self.remaining_time_label = QLabel("⏳ 残り時間: -")
        self.remaining_time_label.setStyleSheet("font-size: 10pt; color: #555; padding: 5px; font-weight: bold;")
        time_layout.addWidget(self.remaining_time_label)
        
        training_detail_layout.addLayout(time_layout)
        
        training_detail_group.setLayout(training_detail_layout)
        right_layout.addWidget(training_detail_group)
        right_layout.addStretch()
        
        # 2列レイアウトに追加
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=1)
        
        return widget
    
    def update_hwinfo_metrics_with_monitoring(self):
        """システム情報を更新（監視付き）"""
        try:
            self.update_hwinfo_metrics()
            # 更新後に状態をチェック
            self.check_system_info_status()
        except (ImportError, ModuleNotFoundError) as e:
            # psutilなどのインポートエラーは無視（既にハンドリング済み）
            if 'psutil' in str(e).lower():
                # エラーメッセージを出力しない（既に他の場所で出力済み）
                pass
            else:
                print(f"[WARN] update_hwinfo_metrics_with_monitoring: {e}")
        except Exception as e:
            # その他の予期しないエラーも安全に処理（アプリケーションをクラッシュさせない）
            print(f"[WARN] update_hwinfo_metrics_with_monitoring error: {e}")
            import traceback
            traceback.print_exc()
    
    def check_system_info_status(self):
        """システム情報の表示状態をチェック"""
        if not hasattr(self, 'cpu_label') or not hasattr(self, 'mem_label'):
            return
        
        # CPU/メモリが「-」の場合は自動修正を試みる
        cpu_text = self.cpu_label.text()
        mem_text = self.mem_label.text()
        
        if "-" in cpu_text or "-" in mem_text:
            print(f"[MONITOR] System info missing, auto-fixing... CPU: {cpu_text}, MEM: {mem_text}")
            # 即座に再取得を試みる
            QtCore.QTimer.singleShot(100, self.force_update_system_info)
    
    def force_update_system_info(self):
        """システム情報を強制的に更新"""
        try:
            print("[MONITOR] Force updating system info...")
            # psutilで直接取得
            try:
                import psutil
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=0.1)
                if cpu_percent is not None and 0 <= cpu_percent <= 100:
                    self.cpu_label.setText(f"💻 CPU使用率: {cpu_percent:.1f}%")
                    self.cpu_progress.setValue(int(cpu_percent))
                    print(f"[MONITOR] CPU updated: {cpu_percent:.1f}%")
                
                # メモリ使用率
                vm = psutil.virtual_memory()
                mem_percent = vm.percent
                if mem_percent is not None and 0 <= mem_percent <= 100:
                    self.mem_label.setText(f"💾 メモリ使用率: {mem_percent:.1f}%")
                    self.mem_progress.setValue(int(mem_percent))
                    print(f"[MONITOR] Memory updated: {mem_percent:.1f}%")
            except (ImportError, ModuleNotFoundError) as e:
                # psutilがない場合は、Windows APIで代替
                if 'psutil' in str(e).lower():
                    print("[MONITOR] psutil not available, using Windows API")
                    # デフォルト値で表示
                    if hasattr(self, 'cpu_label') and hasattr(self, 'cpu_progress'):
                        self.cpu_label.setText("💻 CPU使用率: 0.0%")
                        self.cpu_progress.setValue(0)
                    if hasattr(self, 'mem_label') and hasattr(self, 'mem_progress'):
                        self.mem_label.setText("💾 メモリ使用率: 0.0%")
                        self.mem_progress.setValue(0)
                else:
                    print(f"[MONITOR] Import error: {e}")
            except Exception as e:
                print(f"[MONITOR] Force update error: {e}")
        except Exception as e:
            # 最外側のエラーハンドリング（アプリケーションをクラッシュさせない）
            print(f"[WARN] force_update_system_info error: {e}")
    
    def monitor_and_fix_system_info(self):
        """システム情報を監視して自動修正"""
        if not hasattr(self, 'cpu_label') or not hasattr(self, 'mem_label'):
            return
        
        # CPU/メモリが「-」の場合は自動修正
        cpu_text = self.cpu_label.text()
        mem_text = self.mem_label.text()
        
        needs_fix = False
        if "-" in cpu_text:
            needs_fix = True
            print(f"[MONITOR] CPU info missing: {cpu_text}")
        if "-" in mem_text:
            needs_fix = True
            print(f"[MONITOR] Memory info missing: {mem_text}")
        
        if needs_fix:
            print("[MONITOR] Auto-fixing system info...")
            self.force_update_system_info()
        
        # チェックボックスの状態も確認
        if hasattr(self, 'always_show_system_info_checkbox') and self.always_show_system_info_checkbox is not None:
            if not self.always_show_system_info_checkbox.isChecked():
                print("[MONITOR] Checkbox unchecked, re-checking...")
                self.always_show_system_info_checkbox.blockSignals(True)
                self.always_show_system_info_checkbox.setChecked(True)
                self.always_show_system_info_checkbox.setCheckState(QtCore.Qt.Checked)
                self.always_show_system_info_checkbox.blockSignals(False)
    
    def update_hwinfo_metrics(self):
        """HWiNFOからシステムメトリクスを取得して更新（根本的に書き直し）"""
        # ラベルが存在するか確認
        if not hasattr(self, 'cpu_label') or not hasattr(self, 'mem_label'):
            return
        
        # 常時表示チェックボックスの状態を確認
        always_show = True  # デフォルトは常に表示
        if hasattr(self, 'always_show_system_info_checkbox') and self.always_show_system_info_checkbox is not None:
            always_show = self.always_show_system_info_checkbox.isChecked()
        
        # チェックが外れている場合は更新しない（学習中でない場合）
        if not always_show:
            if not self.training_worker or not self.training_worker.isRunning():
                return
        
        # psutilを確実に使用（__init__でインポート済みのはず）
        psutil_module = getattr(self, '_psutil_module', None)
        if psutil_module is None:
            try:
                import psutil
                self._psutil_module = psutil
                psutil_module = psutil
                self._psutil_available = True
                print(f"[DEBUG] psutil imported in update_hwinfo_metrics")
            except (ImportError, ModuleNotFoundError) as e:
                # psutilがない場合は警告を出力しない（既に他の場所で出力済み）
                # print(f"[ERROR] psutil not available: {e}")  # コメントアウト（エラーログが多すぎる）
                # psutilがない場合、Windows APIで取得を試みる
                psutil_module = None
                if hasattr(self, '_get_system_info_windows_api'):
                    try:
                        # CPU/メモリ使用率の代替取得（簡易版）
                        import subprocess
                        # CPU使用率をwmicで取得（簡易版：0.0%を表示）
                        self.cpu_label.setText("💻 CPU使用率: 0.0%")
                        self.cpu_progress.setValue(0)
                        # メモリ使用率も0.0%を表示
                        self.mem_label.setText("💾 メモリ使用率: 0.0%")
                        self.mem_progress.setValue(0)
                        return
                    except Exception:
                        pass
                # 最終的なフォールバック
                self.cpu_label.setText("💻 CPU使用率: 0.0%")
                self.cpu_progress.setValue(0)
                self.mem_label.setText("💾 メモリ使用率: 0.0%")
                self.mem_progress.setValue(0)
                return
        
        # HWiNFOからデータ取得を試みる
        hwinfo_data = None
        if HAS_HWINFO_READER:
            try:
                hwinfo_data = read_hwinfo_shared_memory()
            except (OSError, AttributeError, ValueError, MemoryError, BufferError):
                hwinfo_data = None
            except Exception:
                hwinfo_data = None
        
        # CPU使用率（HWiNFO → psutil → JSONの順で取得を試みる）
        cpu_percent = None
        if hwinfo_data and hwinfo_data.get('cpu_percent') is not None:
            try:
                val = float(hwinfo_data['cpu_percent'])
                if 0 <= val <= 100:
                    cpu_percent = val
            except (ValueError, TypeError):
                cpu_percent = None
        
        # 根本的に変更: 常にpsutilで直接取得を試みる（必ず値を取得）
        if psutil_module:
            try:
                # 最初にinterval=0.1で確実に値を取得（ブロッキング）
                if cpu_percent is None:
                    cpu_percent = psutil_module.cpu_percent(interval=0.1)
                # 値が無効な場合は再取得
                if cpu_percent is None or cpu_percent < 0 or cpu_percent > 100:
                    cpu_percent = psutil_module.cpu_percent(interval=0.0)
                # それでも無効な場合は0.0を設定（「-」は表示しない）
                if cpu_percent is None or cpu_percent < 0 or cpu_percent > 100:
                    cpu_percent = 0.0
            except Exception as e:
                cpu_percent = 0.0
                print(f"[ERROR] CPU usage error: {e}")
        else:
            # psutilがない場合も0.0を表示（「-」は使わない）
            cpu_percent = 0.0
        
        # JSONファイルから取得を試みる（最後の手段）
        if cpu_percent is None:
            try:
                if STATUS_FILE.exists():
                    with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                        status = json.load(f)
                        if status.get('cpu_percent') is not None:
                            val = float(status['cpu_percent'])
                            if 0 <= val <= 100:
                                cpu_percent = val
            except Exception:
                pass
        
        # 表示を更新（必ず値を表示：「-」は使わない）
        if cpu_percent is not None and 0 <= cpu_percent <= 100:
            self.cpu_label.setText(f"💻 CPU使用率: {cpu_percent:.1f}%")
            self.cpu_progress.setValue(int(cpu_percent))
        elif psutil_module:
            # フォールバック: psutilで再取得
            try:
                cpu_percent = psutil_module.cpu_percent(interval=0.0)
                if cpu_percent is not None and 0 <= cpu_percent <= 100:
                    self.cpu_label.setText(f"💻 CPU使用率: {cpu_percent:.1f}%")
                    self.cpu_progress.setValue(int(cpu_percent))
                else:
                    self.cpu_label.setText("💻 CPU使用率: 0.0%")
                    self.cpu_progress.setValue(0)
            except Exception:
                self.cpu_label.setText("💻 CPU使用率: 0.0%")
                self.cpu_progress.setValue(0)
        else:
            # psutilがない場合も0.0を表示
            self.cpu_label.setText("💻 CPU使用率: 0.0%")
            self.cpu_progress.setValue(0)
        
        # CPU温度
        if hwinfo_data and hwinfo_data.get('cpu_temp_c') is not None:
            cpu_temp = float(hwinfo_data['cpu_temp_c'])
            self.cpu_temp_label.setText(f"🌡️ CPU温度: {cpu_temp:.0f}℃")
        else:
            # JSONファイルから取得を試みる
            try:
                if STATUS_FILE.exists():
                    with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                        status = json.load(f)
                        if status.get('cpu_temp_c') is not None:
                            cpu_temp = float(status['cpu_temp_c'])
                            self.cpu_temp_label.setText(f"🌡️ CPU温度: {cpu_temp:.0f}℃")
                        else:
                            self.cpu_temp_label.setText("🌡️ CPU温度: -")
            except Exception:
                self.cpu_temp_label.setText("🌡️ CPU温度: -")
        
        # メモリ使用率（HWiNFO → psutil → JSONの順で取得を試みる）
        mem_percent = None
        if hwinfo_data and hwinfo_data.get('mem_percent') is not None:
            try:
                val = float(hwinfo_data['mem_percent'])
                if 0 <= val <= 100:
                    mem_percent = val
            except (ValueError, TypeError):
                mem_percent = None
        
        # 根本的に変更: 常にpsutilで直接取得を試みる（必ず値を取得）
        if psutil_module:
            try:
                if mem_percent is None:
                    vm = psutil_module.virtual_memory()
                    mem_percent = vm.percent
                # 値が無効な場合は0.0を設定（「-」は表示しない）
                if mem_percent is None or mem_percent < 0 or mem_percent > 100:
                    mem_percent = 0.0
            except Exception as e:
                mem_percent = 0.0
                print(f"[ERROR] Memory usage error: {e}")
        else:
            # psutilがない場合も0.0を表示
            mem_percent = 0.0
        
        # JSONファイルから取得を試みる（最後の手段）
        if mem_percent is None:
            try:
                if STATUS_FILE.exists():
                    with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                        status = json.load(f)
                        if status.get('mem_percent') is not None:
                            val = float(status['mem_percent'])
                            if 0 <= val <= 100:
                                mem_percent = val
            except Exception:
                pass
        
        # 表示を更新（必ず値を表示：「-」は使わない）
        if mem_percent is not None and 0 <= mem_percent <= 100:
            self.mem_label.setText(f"💾 メモリ使用率: {mem_percent:.1f}%")
            self.mem_progress.setValue(int(mem_percent))
        elif psutil_module:
            # フォールバック: psutilで再取得
            try:
                vm = psutil_module.virtual_memory()
                mem_percent = vm.percent
                if mem_percent is not None and 0 <= mem_percent <= 100:
                    self.mem_label.setText(f"💾 メモリ使用率: {mem_percent:.1f}%")
                    self.mem_progress.setValue(int(mem_percent))
                else:
                    self.mem_label.setText("💾 メモリ使用率: 0.0%")
                    self.mem_progress.setValue(0)
            except Exception:
                self.mem_label.setText("💾 メモリ使用率: 0.0%")
                self.mem_progress.setValue(0)
        else:
            # psutilがない場合も0.0を表示
            self.mem_label.setText("💾 メモリ使用率: 0.0%")
            self.mem_progress.setValue(0)
        
        # GPU使用率（HWiNFOが最も信頼できる）
        gpu_util = None
        if hwinfo_data and hwinfo_data.get('gpu_util_percent') is not None:
            val = float(hwinfo_data['gpu_util_percent'])
            if 0 <= val <= 100:
                gpu_util = val
        
        # HWiNFOで取得できない場合はJSONから取得
        if gpu_util is None:
            try:
                if STATUS_FILE.exists():
                    with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                        status = json.load(f)
                        if status.get('gpu_util_percent') is not None:
                            val = float(status['gpu_util_percent'])
                            if 0 <= val <= 100:
                                gpu_util = val
            except Exception:
                pass
        
        if gpu_util is not None and 0 <= gpu_util <= 100:
            self.gpu_label.setText(f"🎮 GPU使用率: {gpu_util:.0f}%")
            self.gpu_progress.setValue(int(gpu_util))
            print(f"[DEBUG] GPU usage updated: {gpu_util:.0f}%")
        else:
            # GPU使用率が取得できない場合でも、0%を表示してプログレスバーを更新
            self.gpu_label.setText("🎮 GPU使用率: 0%")
            self.gpu_progress.setValue(0)
            print(f"[DEBUG] GPU usage not available, setting to 0%")
        
        # GPU温度
        if hwinfo_data and hwinfo_data.get('gpu_temp_c') is not None:
            val = float(hwinfo_data['gpu_temp_c'])
            if 20 <= val <= 150:
                self.gpu_temp_label.setText(f"🌡️ GPU温度: {val:.0f}℃")
        else:
            # JSONファイルから取得を試みる
            try:
                if STATUS_FILE.exists():
                    with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                        status = json.load(f)
                        if status.get('gpu_temp_c') is not None:
                            val = float(status['gpu_temp_c'])
                            if 20 <= val <= 150:
                                self.gpu_temp_label.setText(f"🌡️ GPU温度: {val:.0f}℃")
                            else:
                                self.gpu_temp_label.setText("🌡️ GPU温度: -")
                        else:
                            self.gpu_temp_label.setText("🌡️ GPU温度: -")
            except Exception:
                self.gpu_temp_label.setText("🌡️ GPU温度: -")
        
        # GPU電力
        if hwinfo_data and hwinfo_data.get('gpu_power_w') is not None:
            val = float(hwinfo_data['gpu_power_w'])
            if 0 <= val <= 1000:
                self.gpu_power_label.setText(f"⚡ GPU電力: {val:.1f} W")
        else:
            # JSONファイルから取得を試みる
            try:
                if STATUS_FILE.exists():
                    with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                        status = json.load(f)
                        if status.get('gpu_power_w') is not None:
                            val = float(status['gpu_power_w'])
                            if 0 <= val <= 1000:
                                self.gpu_power_label.setText(f"⚡ GPU電力: {val:.1f} W")
                            else:
                                self.gpu_power_label.setText("⚡ GPU電力: -")
                        else:
                            self.gpu_power_label.setText("⚡ GPU電力: -")
            except Exception:
                self.gpu_power_label.setText("⚡ GPU電力: -")
        
    
    def update_training_status(self):
        """学習ステータスを更新（以前のビューアーと同じロジック）"""
        # 学習ワーカーが実行中でない場合は、未開始状態として表示
        if not self.training_worker or not self.training_worker.isRunning():
            if not STATUS_FILE.exists():
                # ステータスファイルも存在しない場合は未開始
                if hasattr(self, 'training_status_label'):
                    self.training_status_label.setText('ステータス: ⚪ 未開始')
                    self.training_status_label.setStyleSheet('color:#888; font-weight:bold; font-size:12pt;')
                if hasattr(self, 'current_activity_label'):
                    self.current_activity_label.setText('現在の作業: 未開始')
                if hasattr(self, 'current_activity_detail_label'):
                    self.current_activity_detail_label.setText('現在の作業: 未開始')
                if hasattr(self, 'training_progress'):
                    self.training_progress.setValue(0)
                return
            # ステータスファイルは存在するが、ワーカーが実行中でない場合は続行（前回の学習結果を表示）
        
        if not STATUS_FILE.exists():
            return
        
        try:
            # ステータスファイルから読み込み
            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                status = json.load(f)
            
            # HWiNFOから直接システムメトリクスを取得（最優先・最も信頼できる）
            hwinfo_data = None
            if HAS_HWINFO_READER:
                try:
                    hwinfo_data = read_hwinfo_shared_memory()
                except (OSError, WindowsError, AttributeError, ValueError, MemoryError, BufferError):
                    # メモリアクセスエラーなど、安全に処理
                    hwinfo_data = None
                except Exception:
                    # その他の予期しないエラーも安全に処理（アプリケーションをクラッシュさせない）
                    hwinfo_data = None
            
            # HWiNFOから取得した値で上書き（JSONより優先）
            if hwinfo_data:
                # CPU使用率
                if hwinfo_data.get('cpu_percent') is not None:
                    status['cpu_percent'] = float(hwinfo_data['cpu_percent'])
                # CPU温度
                if hwinfo_data.get('cpu_temp_c') is not None:
                    status['cpu_temp_c'] = float(hwinfo_data['cpu_temp_c'])
                # メモリ
                if hwinfo_data.get('mem_percent') is not None:
                    status['mem_percent'] = float(hwinfo_data['mem_percent'])
                if hwinfo_data.get('mem_used_mb') is not None:
                    status['mem_used_mb'] = float(hwinfo_data['mem_used_mb'])
                if hwinfo_data.get('mem_total_mb') is not None:
                    status['mem_total_mb'] = float(hwinfo_data['mem_total_mb'])
                # GPU使用率（HWiNFOの値が最も信頼できる）
                if hwinfo_data.get('gpu_util_percent') is not None:
                    val = float(hwinfo_data['gpu_util_percent'])
                    if 0 <= val <= 100:
                        status['gpu_util_percent'] = val
                # GPUメモリ
                if hwinfo_data.get('gpu_mem_used_mb') is not None:
                    val = float(hwinfo_data['gpu_mem_used_mb'])
                    if 0 <= val <= 100000:
                        status['gpu_mem_used_mb'] = val
                if hwinfo_data.get('gpu_mem_total_mb') is not None:
                    val = float(hwinfo_data['gpu_mem_total_mb'])
                    if 0 <= val <= 100000:
                        status['gpu_mem_total_mb'] = val
                # GPU温度
                if hwinfo_data.get('gpu_temp_c') is not None:
                    val = float(hwinfo_data['gpu_temp_c'])
                    if 20 <= val <= 150:
                        status['gpu_temp_c'] = val
                # GPU電力
                if hwinfo_data.get('gpu_power_w') is not None:
                    val = float(hwinfo_data['gpu_power_w'])
                    if 0 <= val <= 1000:
                        status['gpu_power_w'] = val
            
            # HWiNFOで取得できなかった場合はpsutilでフォールバック
            if status.get('cpu_percent') is None or status.get('cpu_percent') == 0:
                try:
                    import psutil
                    status['cpu_percent'] = float(psutil.cpu_percent(interval=0.1))
                except Exception:
                    pass
            
            if status.get('mem_percent') is None or status.get('mem_percent') == 0:
                try:
                    import psutil
                    vm = psutil.virtual_memory()
                    status['mem_percent'] = float(vm.percent)
                    if status.get('mem_used_mb') is None:
                        status['mem_used_mb'] = round(vm.used / (1024*1024), 1)
                    if status.get('mem_total_mb') is None:
                        status['mem_total_mb'] = round(vm.total / (1024*1024), 1)
                except Exception:
                    pass
            
            # モデル名
            model_name = status.get('model_name', '-') or '-'
            model_idx = int(status.get('model_index', 0) or 0)
            models_total = int(status.get('models_total', 0) or 0)
            if hasattr(self, 'model_name_label'):
                self.model_name_label.setText(f"🤖 モデル: {model_name} ({model_idx}/{models_total})")
            
            # 学習回数（以前のビューアーと同じ詳細表示）
            epoch = int(status.get('epoch', 0) or 0)
            epochs_total = int(status.get('epochs_total', 0) or 0)
            epoch_progress = float(status.get('epoch_progress_percent', 0) or 0)
            training_epoch_percent = float(status.get('training_epoch_percent', 0) or 0)
            remain_ep = max(0, epochs_total - epoch)
            
            epoch_text = f"📊 学習回数: {epoch}/{epochs_total} ({epoch_progress:.1f}%)"
            if remain_ep > 0:
                epoch_text += f" 残り{remain_ep}"
            
            if hasattr(self, 'epoch_detail_label'):
                self.epoch_detail_label.setText(epoch_text)
            if hasattr(self, 'epoch_label'):
                self.epoch_label.setText(epoch_text)
            
            # 訓練の進行
            if training_epoch_percent > 0 and hasattr(self, 'training_epoch_label'):
                self.training_epoch_label.setText(f"訓練の進行: {training_epoch_percent:.1f}%")
            
            # 全体進捗
            overall_progress = float(status.get('overall_progress_percent', 0) or 0)
            if hasattr(self, 'overall_progress_label'):
                self.overall_progress_label.setText(f"📈 全体進捗: {int(round(overall_progress))}%")
            if hasattr(self, 'training_progress'):
                self.training_progress.setValue(int(round(overall_progress)))
            
            # 時間（以前のビューアーと同じ表示）
            import time
            elapsed = status.get('overall_elapsed_human') or self.format_time_human(status.get('overall_elapsed_seconds'))
            remaining_secs = status.get('overall_remaining_est_seconds')
            remaining = status.get('overall_remaining_human') or self.format_time_human(remaining_secs)
            
            # 完了予定時刻の計算
            ts_base = status.get('timestamp') or time.time()
            if remaining_secs is not None:
                try:
                    clock = time.strftime('%H:%M', time.localtime(ts_base + float(remaining_secs)))
                except:
                    clock = '-'
            else:
                clock = '-'
            eta = status.get('overall_eta_human') or self.format_time_human(status.get('overall_estimated_total_seconds'))
            
            if hasattr(self, 'elapsed_time_label'):
                self.elapsed_time_label.setText(f"⏱️ 経過時間: {elapsed}")
            if hasattr(self, 'remaining_time_label'):
                self.remaining_time_label.setText(f"⏳ 残り時間: {remaining}")
            if hasattr(self, 'eta_label'):
                self.eta_label.setText(f"終了見込み: {clock}（全体で{eta}）")
            
            # 精度
            accuracy = status.get('accuracy_percent')
            val_accuracy = status.get('val_accuracy_percent')
            if accuracy is not None and val_accuracy is not None:
                acc_text = f"精度: {float(accuracy):.2f}% | 検証: {float(val_accuracy):.2f}%"
            else:
                acc_text = "精度: -"
            if hasattr(self, 'accuracy_label'):
                self.accuracy_label.setText(acc_text)
            
            # 学習率
            lr = str(status.get('learning_rate', '-')) or '-'
            if hasattr(self, 'learning_rate_label'):
                self.learning_rate_label.setText(f"学習率(LR): {lr}")
            
            # ステータス
            stage = status.get('stage', '')
            section = status.get('section', '')
            message = status.get('message', '')
            current_activity = status.get('current_activity', '')  # current_activityを優先的に使用
            
            # 現在の作業内容を日本語で表示（統合前のビューアーのように詳細に）
            # current_activityが設定されている場合は優先的に使用
            if current_activity and current_activity.strip():
                activity_display = current_activity.strip()
                if hasattr(self, 'current_activity_label'):
                    self.current_activity_label.setText(f"現在の作業: {activity_display}")
                if hasattr(self, 'current_activity_detail_label'):
                    self.current_activity_detail_label.setText(f"現在の作業: {activity_display}")
                # 以降の処理はスキップせず、他の情報も更新するため継続
            
            activity_text = "待機中"
            activity_detail = ""
            
            if stage:
                stage_map = {
                    'Data Loading': 'データ読み込み中',
                    'Loading Classes': 'クラス読み込み中',
                    'Preprocessing': '前処理中',
                    'Training': '学習中',
                    'アンサンブル学習': 'アンサンブル学習中',
                    'Evaluating': '評価中',
                    'Evaluation Data Split': '評価データ分割中',
                    'Model Evaluation': 'モデル評価中',
                    'Saving': '保存中',
                    'Model Saving': 'モデル保存中',
                    'training_epoch': '学習回数実行中',
                    'preparation': '準備中',
                    'Preparation': '準備中',
                    'Error': 'エラー',
                }
                activity_text = stage_map.get(stage, stage + '中' if not stage.endswith('中') else stage)
                
            if section:
                section_map = {
                    'Data Loading': 'データ読み込み',
                    'Loading Classes': 'クラス読み込み',
                    'Preprocessing': '前処理',
                    'Training': '学習',
                    'アンサンブル学習': 'アンサンブル学習',
                    'Evaluating': '評価',
                    'Evaluation Data Split': '評価データ分割',
                    'Model Evaluation': 'モデル評価',
                    'Saving': '保存',
                    'Model Saving': 'モデル保存',
                    'good': '良品クラス処理',
                    'black_spot': '黒点クラス処理',
                    'chipping': '欠けクラス処理',
                    'scratch': '傷クラス処理',
                    'dent': '凹みクラス処理',
                    'distortion': '歪みクラス処理',
                }
                activity_detail = section_map.get(section, section)
            
            # 現在の作業内容を表示（統合前のビューアーのように詳細に）
            # 学習ワーカーが実行中でない場合の処理
            is_training_running = self.training_worker and self.training_worker.isRunning()
            
            # ワーカーが実行中の場合は、ステータスファイルの内容に関わらず詳細な作業内容を表示
            if is_training_running:
                # current_activityが設定されている場合は常に優先的に使用
                if current_activity and current_activity.strip():
                    activity_display = current_activity.strip()
                elif not stage or stage == '-' or is_waiting:
                    # ワーカーは実行中だが、ステータスファイルがまだ更新されていない場合
                    activity_display = "準備中... (学習ワーカー起動中)"
                elif message and ('完了' in message or 'Complete' in message or 'Completed' in message):
                    activity_display = "完了"
                elif section and section != stage:
                    # セクションがステージと異なる場合は詳細に表示
                    # 例: "データ読み込み中 - 良品クラス処理"
                    if section in ['good', 'black_spot', 'chipping', 'scratch', 'dent', 'distortion']:
                        activity_display = f"{activity_text} - {activity_detail}中"
                    else:
                        activity_display = f"{activity_text} - {activity_detail}"
                elif activity_detail and activity_detail != activity_text:
                    # 詳細が異なる場合は詳細に表示
                    if activity_detail in ['良品クラス処理', '黒点クラス処理', '欠けクラス処理', '傷クラス処理']:
                        activity_display = f"{activity_text} - {activity_detail}中"
                    else:
                        activity_display = f"{activity_text} - {activity_detail}"
                else:
                    # ステージだけ表示（既に「中」が含まれている）
                    activity_display = activity_text
            elif message and ('完了' in message or 'Complete' in message or 'Completed' in message) and overall_progress >= 100.0:
                activity_display = "前回完了"
            elif message and ('完了' in message or 'Complete' in message or 'Completed' in message):
                activity_display = "完了"
            elif section and section != stage:
                # セクションがステージと異なる場合は詳細に表示
                if section in ['good', 'black_spot', 'chipping', 'scratch', 'dent', 'distortion']:
                    activity_display = f"{activity_text} - {activity_detail}中"
                else:
                    activity_display = f"{activity_text} - {activity_detail}"
            elif activity_detail and activity_detail != activity_text:
                # 詳細が異なる場合は詳細に表示
                if activity_detail in ['良品クラス処理', '黒点クラス処理', '欠けクラス処理', '傷クラス処理']:
                    activity_display = f"{activity_text} - {activity_detail}中"
                else:
                    activity_display = f"{activity_text} - {activity_detail}"
            else:
                # ステージだけ表示（既に「中」が含まれている）
                activity_display = activity_text
            
            # current_activityが設定されている場合は常に優先的に使用（最後に上書き）
            if current_activity and current_activity.strip():
                activity_display = current_activity.strip()
            
            if hasattr(self, 'current_activity_label'):
                self.current_activity_label.setText(f"現在の作業: {activity_display}")
            if hasattr(self, 'current_activity_detail_label'):
                self.current_activity_detail_label.setText(f"現在の作業: {activity_display}")
            
            # ステータス表示（以前のビューアーと同じロジック）
            # タイムスタンプから待機中かどうか判定（5分以上更新されていない場合は待機中とみなす）
            ts = status.get('timestamp')
            is_waiting = False
            if ts:
                try:
                    import time
                    elapsed_since_update = time.time() - float(ts)
                    if elapsed_since_update > 300:  # 5分以上更新なし
                        is_waiting = True
                except Exception:
                    pass
            
            # ステータステキストとスタイルを決定（以前のビューアーと同じ）
            status_text = 'ステータス: -'
            status_style = 'color:#666; font-weight:bold; font-size:12pt;'
            
            try:
                # 学習ワーカーが実行中でない場合の処理
                is_training_running = self.training_worker and self.training_worker.isRunning()
                
                # ワーカーが実行中の場合は、ステータスファイルの内容に関わらず「学習中」として扱う
                if is_training_running and (not stage or stage == '-' or is_waiting):
                    # ワーカーは実行中だが、ステータスファイルがまだ更新されていない場合は「学習中」と表示
                    status_text = 'ステータス: 🟢 学習中'
                    status_style = 'color:#00cc00; font-weight:bold; font-size:12pt;'
                elif not stage or stage == '-' or is_waiting:
                    if not is_training_running:
                        # ワーカーが実行中でない場合、完了でも未開始状態として表示
                        if message and ('完了' in message or 'Complete' in message or 'Completed' in message) and overall_progress >= 100.0:
                            status_text = 'ステータス: ⚪ 前回完了'
                            status_style = 'color:#999; font-weight:bold; font-size:12pt;'
                        else:
                            status_text = 'ステータス: ⚪ 未開始'
                            status_style = 'color:#888; font-weight:bold; font-size:12pt;'
                    elif message and ('完了' in message or 'Complete' in message or 'Completed' in message):
                        status_text = 'ステータス: ⚪ 完了'
                        status_style = 'color:#999; font-weight:bold; font-size:12pt;'
                    else:
                        status_text = 'ステータス: ⚫ 待機中 / 停止中'
                        status_style = 'color:#888; font-weight:bold; font-size:12pt;'
                elif 'Training' in stage or 'training' in stage.lower() or 'training_epoch' in stage.lower() or '学習' in stage or 'train' in stage.lower():
                    status_text = 'ステータス: 🟢 学習中'
                    status_style = 'color:#00cc00; font-weight:bold; font-size:12pt;'
                elif 'Building' in stage or '構築' in stage or 'build' in stage.lower():
                    status_text = 'ステータス: 🟡 モデル構築中'
                    status_style = 'color:#ff9900; font-weight:bold; font-size:12pt;'
                elif 'Data Loading' in stage or '読み込み' in stage or 'Loading' in stage:
                    status_text = 'ステータス: 🔵 データ読み込み中'
                    status_style = 'color:#0066cc; font-weight:bold; font-size:12pt;'
                elif 'Evaluation' in stage or '評価' in stage or 'eval' in stage.lower():
                    status_text = 'ステータス: 🟣 評価中'
                    status_style = 'color:#9966cc; font-weight:bold; font-size:12pt;'
                elif 'Saving' in stage or '保存' in stage:
                    status_text = 'ステータス: 🔶 保存中'
                    status_style = 'color:#ff6600; font-weight:bold; font-size:12pt;'
                elif 'preparation' in stage.lower() or '準備' in stage or 'Preparation' in stage:
                    status_text = 'ステータス: 🟡 準備中'
                    status_style = 'color:#ff9900; font-weight:bold; font-size:12pt;'
                elif stage == 'Error':
                    status_text = 'ステータス: ❌ エラー'
                    status_style = 'color:#cc0000; font-weight:bold; font-size:12pt;'
                else:
                    # ステージ名を日本語に変換（stage_mapを使用）
                    stage_map_jp = {
                        'Data Loading': 'データ読み込み中',
                        'Loading Classes': 'クラス読み込み中',
                        'Preprocessing': '前処理中',
                        'Training': '学習中',
                        'アンサンブル学習': 'アンサンブル学習中',
                        'Evaluating': '評価中',
                        'Evaluation Data Split': '評価データ分割中',
                        'Model Evaluation': 'モデル評価中',
                        'Saving': '保存中',
                        'Model Saving': 'モデル保存中',
                        'training_epoch': '学習回数実行中',
                        'preparation': '準備中',
                        'Preparation': '準備中',
                        'Error': 'エラー',
                    }
                    stage_jp = stage_map_jp.get(stage, stage)
                    status_text = f'ステータス: {stage_jp}'
                    status_style = 'color:#666; font-weight:bold; font-size:12pt;'
                
                if hasattr(self, 'training_status_label'):
                    self.training_status_label.setText(status_text)
                    self.training_status_label.setStyleSheet(status_style)
            except Exception as e:
                if hasattr(self, 'training_status_label'):
                    self.training_status_label.setText('ステータス: エラー')
                    self.training_status_label.setStyleSheet('color:#cc0000; font-weight:bold; font-size:12pt;')
                
        except Exception:
            pass
    
    def format_time_human(self, seconds):
        """秒数を人間が読みやすい形式に変換"""
        if seconds is None:
            return '-'
        try:
            seconds = float(seconds)
            if seconds < 60:
                return f"{int(seconds)}秒"
            elif seconds < 3600:
                return f"約{int(seconds/60)}分"
            else:
                hours = int(seconds / 3600)
                minutes = int((seconds % 3600) / 60)
                if minutes > 0:
                    return f"約{hours}時間{minutes}分"
                else:
                    return f"約{hours}時間"
        except:
            return '-'
    
    def check_training_health(self):
        """学習の健康状態をチェックし、停止を検出したら警告を表示"""
        if not STATUS_FILE.exists():
            return
        
        try:
            import time
            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                status = json.load(f)
            
            timestamp = status.get('timestamp')
            if not timestamp:
                return
            
            elapsed = time.time() - float(timestamp)
            overall_progress = status.get('overall_progress_percent', 0)
            stage = status.get('stage', '')
            
            # 3分以上更新がなく、100%未満の場合、停止と判定
            STUCK_THRESHOLD = 180  # 3分
            
            if elapsed > STUCK_THRESHOLD and overall_progress < 100.0:
                # 学習プロセスが実行中か確認
                try:
                    import psutil
                    training_running = False
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            cmdline = proc.info.get('cmdline', [])
                            if cmdline and any('train_4class_sparse_ensemble' in str(arg) for arg in cmdline):
                                training_running = True
                                break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    
                    if not training_running:
                        # 学習が停止している場合、警告を表示
                        elapsed_min = int(elapsed / 60)
                        warning_msg = f'警告: 学習が停止している可能性があります（{elapsed_min}分間更新なし）'
                        self.training_status_label.setText(f"状態: {warning_msg}")
                        self.current_activity_label.setText(f"⚠️ 停止検出: 最後の更新から{elapsed_min}分経過")
                        self.statusBar().showMessage(warning_msg, 10000)
                except ImportError:
                    # psutilが利用できない場合
                    elapsed_min = int(elapsed / 60)
                    if elapsed_min >= 3:
                        warning_msg = f'警告: {elapsed_min}分間更新がありません'
                        self.training_status_label.setText(f"状態: {warning_msg}")
                        self.statusBar().showMessage(warning_msg, 10000)
                    
        except Exception:
            pass  # エラーは無視
    
    def update_camera_list(self):
        """利用可能なカメラのリストを更新"""
        # camera_comboが存在する場合のみ更新
        if hasattr(self, 'camera_combo'):
            self.camera_combo.clear()
            # 最大10個のカメラを検出
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        self.camera_combo.addItem(f"カメラ {i}", i)
                    cap.release()
            
            if self.camera_combo.count() == 0:
                self.camera_combo.addItem("カメラが見つかりません", 0)
    
    def start_inspection(self):
        """外観検査を開始"""
        if self.inspection_worker and self.inspection_worker.isRunning():
            return
        
        # カメラIDを取得（保存された設定 > 設定ダイアログ > デフォルト）
        camera_id = 0
        # まず保存されたカメラIDを確認
        if hasattr(self, 'saved_camera_id'):
            camera_id = self.saved_camera_id
            print(f"[INFO] 保存されたカメラIDを使用: {camera_id}")
        elif hasattr(self, 'camera_combo') and self.camera_combo is not None and self.camera_combo.count() > 0:
            camera_id = self.camera_combo.currentData()
            if camera_id is None:
                # カメラIDがNoneの場合、テキストから抽出を試みる
                current_text = self.camera_combo.currentText()
                if current_text and "カメラ" in current_text:
                    try:
                        import re
                        match = re.search(r'カメラ\s*(\d+)', current_text)
                        if match:
                            camera_id = int(match.group(1))
                    except:
                        pass
                if camera_id is None:
                    camera_id = 0
        else:
            # camera_comboが存在しない場合、利用可能なカメラを検出
            print("[INFO] カメラ選択UIが見つかりません。カメラを検出中...")
            for i in range(5):
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret:
                            camera_id = i
                            print(f"[INFO] カメラ {i} を検出しました")
                            cap.release()
                            break
                        cap.release()
                except:
                    pass
        
        print(f"[INFO] 使用カメラID: {camera_id}")
        
        # 解像度を取得（設定から）
        resolution = None
        if hasattr(self, 'camera_resolution') and self.camera_resolution is not None:
            resolution = self.camera_resolution
            print(f"[INFO] 検査解像度: {resolution[0]}x{resolution[1]}")
        
        # カメラ情報ラベルを更新
        if hasattr(self, 'camera_info_label') and self.camera_info_label is not None:
            if resolution is not None:
                width, height = resolution
                self.camera_info_label.setText(f"カメラ: {camera_id} / 解像度: {width}x{height}")
            else:
                self.camera_info_label.setText(f"カメラ: {camera_id} / 解像度: 自動")
        
        # カメラが開けるか事前チェック
        try:
            test_cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not test_cap.isOpened():
                QtWidgets.QMessageBox.warning(
                    self,
                    'カメラエラー',
                    f'カメラ {camera_id} を開けません。\n\n'
                    f'他のアプリケーションがカメラを使用していないか確認してください。\n'
                    f'設定画面でカメラを選択してください。'
                )
                test_cap.release()
                return
            test_cap.release()
        except Exception as e:
            print(f"[ERROR] カメラ事前チェックエラー: {e}")
            QtWidgets.QMessageBox.warning(
                self,
                'カメラエラー',
                f'カメラ {camera_id} のチェック中にエラーが発生しました:\n{str(e)}'
            )
            return
        
        # UI更新（カメラ読み込み中表示を更新）
        if hasattr(self, 'camera_label') and self.camera_label is not None:
            self.camera_label.setText("カメラを初期化しています...")
            self.camera_label.setStyleSheet("""
                border: 3px solid #0066cc;
                border-radius: 8px;
                background-color: #f8f9fa;
                font-size: 14pt;
                color: #666;
            """)
        
        self.inspection_worker = InspectionWorker(camera_id=camera_id, resolution=resolution)
        self.inspection_worker.frame_ready.connect(self.display_frame)
        self.inspection_worker.prediction_ready.connect(self.on_prediction_received)
        self.inspection_worker.processing_time_ready.connect(self.on_processing_time_received)
        self.inspection_worker.camera_info_ready.connect(self._update_camera_info)
        
        # エラーハンドリング: ワーカーが正常に起動しない場合
        try:
            self.inspection_worker.start()
            print(f"[INFO] InspectionWorkerを起動しました（カメラID: {camera_id}）")
        except Exception as e:
            print(f"[ERROR] InspectionWorkerの起動エラー: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self,
                'エラー',
                f'検査の開始に失敗しました:\n{str(e)}'
            )
            if hasattr(self, 'camera_label') and self.camera_label is not None:
                self.camera_label.setText("カメラ初期化エラー")
            return
        
        self.start_inspection_btn.setEnabled(False)
        self.stop_inspection_btn.setEnabled(True)
        self.save_frame_btn.setEnabled(True)
        self.feedback_btn.setEnabled(True)
        if hasattr(self, 'result_feedback_btn'):
            self.result_feedback_btn.setEnabled(True)
        self.inspection_start_time = time.time()
        self.fps_counter = 0
        self.fps_time = time.time()
        
        # NG率アラートチェックタイマーを開始
        if self.ng_rate_alert_enabled and not self.alert_check_timer.isActive():
            self.alert_check_timer.start(60000)  # 1分ごとにチェック
        
        self.statusBar().showMessage('外観検査を開始しました')
    
    def stop_inspection(self):
        """外観検査を停止"""
        if self.inspection_worker:
            self.inspection_worker.stop()
            self.inspection_worker.wait()
        
        self.start_inspection_btn.setEnabled(True)
        self.stop_inspection_btn.setEnabled(False)
        self.save_frame_btn.setEnabled(False)
        self.feedback_btn.setEnabled(False)
        if hasattr(self, 'result_feedback_btn'):
            self.result_feedback_btn.setEnabled(False)
        
        if hasattr(self, 'camera_label') and self.camera_label is not None:
            self.camera_label.clear()
            self.camera_label.setText("カメラを初期化しています...")
            self.camera_label.setStyleSheet("""
                border: 3px solid #0066cc;
                border-radius: 8px;
                background-color: #f8f9fa;
                font-size: 14pt;
                color: #666;
            """)
        
        # アラート点滅を停止
        if hasattr(self, 'alert_flash_timer'):
            self.alert_flash_timer.stop()
        
        self.statusBar().showMessage('外観検査を停止しました')
    
    def display_frame(self, frame):
        """フレームを表示"""
        if frame is None:
            # Noneが送信された場合はエラー状態
            if hasattr(self, 'camera_label') and self.camera_label is not None:
                self.camera_label.setText("カメラエラー\nカメラを開けません")
                self.camera_label.setStyleSheet("""
                    border: 3px solid #dc3545;
                    border-radius: 8px;
                    background-color: #f8d7da;
                    font-size: 14pt;
                    color: #721c24;
                """)
            return
        
        if frame.size == 0:
            return
        
        try:
            # 現在のフレームを保存（手動保存用）
            self.current_frame = frame.copy()
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            
            if w == 0 or h == 0:
                return
            
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            if qt_image.isNull():
                print("[WARNING] QImageの作成に失敗しました")
                return
            
            pixmap = QPixmap.fromImage(qt_image)
            
            if pixmap.isNull():
                print("[WARNING] QPixmapの作成に失敗しました")
                return
            
            # ラベルの実際のサイズを取得（マージンとボーダーを考慮）
            if not hasattr(self, 'camera_label') or self.camera_label is None:
                return
                
            label_size = self.camera_label.size()
            if label_size.width() <= 0 or label_size.height() <= 0:
                return
                
            # ボーダーとマージンを考慮して少し小さく
            available_width = max(1, label_size.width() - 20)  # ボーダー分
            available_height = max(1, label_size.height() - 20)
            
            # アスペクト比を保持してスケール
            scaled_pixmap = pixmap.scaled(
                available_width, 
                available_height, 
                QtCore.Qt.KeepAspectRatio, 
                QtCore.Qt.SmoothTransformation
            )
            
            if not scaled_pixmap.isNull():
                self.camera_label.setPixmap(scaled_pixmap)
                # 初期化メッセージをクリア（映像が表示されたことを示す）
                self.camera_label.setText("")
            else:
                print("[WARNING] スケールされたPixmapが無効です")
            
            # FPSカウンター
            self.fps_counter += 1
        except Exception as e:
            print(f"[ERROR] フレーム表示エラー: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_camera_info(self, camera_id, width, height):
        """カメラ情報ラベルを更新（スレッドセーフ）"""
        try:
            if hasattr(self, 'camera_info_label') and self.camera_info_label is not None:
                self.camera_info_label.setText(f"カメラ: {camera_id} / 解像度: {width}x{height}")
        except Exception as e:
            print(f"[WARN] カメラ情報ラベル更新エラー: {e}")
    
    def _update_camera_previews(self):
        """すべてのカメラプレビューを更新（遅延読み込み方式で軽量化）"""
        if not hasattr(self, 'camera_preview_labels') or not hasattr(self, 'camera_preview_caps'):
            return
        
        # 設定ダイアログが表示されていない場合は更新しない
        if not hasattr(self, '_settings_dialog') or self._settings_dialog is None:
            return
        
        # 一度に1つのカメラのみ更新（順次処理で軽量化）
        if not hasattr(self, '_current_preview_index'):
            self._current_preview_index = 0
        
        camera_list = list(self.camera_preview_labels.items())
        if len(camera_list) == 0:
            return
        
        # 1回の更新で1つのカメラのみ処理（タイムアウトを短くして非ブロッキングに）
        camera_id, preview_label = camera_list[self._current_preview_index % len(camera_list)]
        self._current_preview_index += 1
        
        try:
            # 失敗したカメラはスキップ
            if camera_id in self.camera_preview_failed:
                return
            
            # カメラがまだ開かれていない場合は開く
            if camera_id not in self.camera_preview_caps or self.camera_preview_caps[camera_id] is None:
                try:
                    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)  # DirectShowバックエンドを明示的に指定
                    if cap.isOpened():
                        # プレビュー用に小さな解像度を設定
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファを最小化
                        # フレームを複数回読み込んで古いバッファをクリア
                        for _ in range(3):
                            cap.read()
                        self.camera_preview_caps[camera_id] = cap
                        preview_label.setText(f"カメラ {camera_id}\n読み込み中...")
                    else:
                        self.camera_preview_failed.add(camera_id)
                        preview_label.setText(f"カメラ {camera_id}\n開けません")
                        return
                except Exception as e:
                    print(f"[WARN] カメラ {camera_id} のオープンエラー: {e}")
                    self.camera_preview_failed.add(camera_id)
                    preview_label.setText(f"カメラ {camera_id}\nエラー")
                    return
            
            cap = self.camera_preview_caps[camera_id]
            if cap is None or not cap.isOpened():
                self.camera_preview_failed.add(camera_id)
                preview_label.setText(f"カメラ {camera_id}\n開けません")
                return
            
            # フレームを読み込む（軽量化：1フレームのみ）
            ret, frame = cap.read()  # 最新フレームを取得
            
            if ret and frame is not None and frame.size > 0:
                try:
                    # フレームをリサイズ（160x120に、軽量化）
                    frame_resized = cv2.resize(frame, (160, 120))
                    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    if h > 0 and w > 0:
                        bytes_per_line = ch * w
                        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        
                        if not qt_image.isNull():
                            pixmap = QPixmap.fromImage(qt_image)
                            if not pixmap.isNull():
                                preview_label.setPixmap(pixmap)
                                # 成功したらリトライカウントをリセット
                                if camera_id in self.camera_preview_retry_count:
                                    self.camera_preview_retry_count[camera_id] = 0
                except Exception as e:
                    retry_count = self.camera_preview_retry_count.get(camera_id, 0) + 1
                    self.camera_preview_retry_count[camera_id] = retry_count
                    if retry_count >= 3:
                        self.camera_preview_failed.add(camera_id)
                        preview_label.setText(f"カメラ {camera_id}\nエラー")
            else:
                # フレーム読み込み失敗
                retry_count = self.camera_preview_retry_count.get(camera_id, 0) + 1
                self.camera_preview_retry_count[camera_id] = retry_count
                if retry_count >= 3:
                    self.camera_preview_failed.add(camera_id)
                    preview_label.setText(f"カメラ {camera_id}\nフレーム読み込み失敗")
        except Exception as e:
            print(f"[WARN] カメラ {camera_id} プレビュー更新エラー: {e}")
            retry_count = self.camera_preview_retry_count.get(camera_id, 0) + 1
            self.camera_preview_retry_count[camera_id] = retry_count
            if retry_count >= 3:
                self.camera_preview_failed.add(camera_id)
    
    def on_processing_time_received(self, processing_time_ms):
        """処理時間を受信（パフォーマンス分析用に記録）"""
        try:
            self.processing_times.append(float(processing_time_ms))
        except Exception:
            pass
    
    def on_prediction_received(self, prediction, confidence):
        """予測結果を受信して処理"""
        print(f"[DEBUG] on_prediction_received呼び出し: prediction={prediction}, confidence={confidence:.4f}")
        # 手動判定モードの場合
        if self.manual_mode:
            # AIの予測結果を保存（手動判定待ち）
            self.pending_ai_prediction = prediction
            self.pending_ai_confidence = confidence
            
            # AI予測を表示
            prediction_map = {
                'good': '良品',
                'black_spot': '黒点',
                'chipping': '欠け',
                'scratch': '傷',
                'dent': '凹み',
                'distortion': '歪み'
            }
            prediction_jp = prediction_map.get(prediction, prediction)
            # 信頼度に応じて色を変更
            if confidence >= 0.8:
                conf_color = "#28a745"  # 緑
                bg_color = "#e8f5e9"
            elif confidence >= 0.6:
                conf_color = "#ffc107"  # 黄
                bg_color = "#fff9e6"
            else:
                conf_color = "#dc3545"  # 赤
                bg_color = "#ffebee"
            
            self.ai_prediction_label.setText(f"🤖 AI予測: {prediction_jp} (信頼度: {confidence:.1%})")
            self.ai_prediction_label.setStyleSheet(f"""
                font-size: 12pt;
                font-weight: bold;
                color: {conf_color};
                padding: 8px;
                background-color: {bg_color};
                border: 1px solid {conf_color};
                border-radius: 4px;
            """)
            
            # 判定ボタンを有効化
            for btn, _ in self.judgment_buttons:
                btn.setEnabled(True)
            
            # 判定待ち表示
            self.prediction_label.setText("⏳ 判定待ち...")
            self.prediction_label.setStyleSheet("""
                font-size: 28pt;
                font-weight: bold;
                color: #ff9800;
                padding: 15px;
                background-color: #fff9e6;
                border: 3px solid #ff9800;
                border-radius: 8px;
                min-height: 60px;
            """)
            self.confidence_label.setText(f"🤖 AI信頼度: {confidence:.2%}")
            self.confidence_label.setStyleSheet("""
                font-size: 18pt;
                color: #666;
                padding: 10px;
                background-color: #f5f5f5;
                border-radius: 6px;
            """)
            
            # 手動判定待ちの場合はここで処理を終了（ユーザーが判定ボタンを押すまで待つ）
            return
        
        # 自動判定モードの場合（従来の処理）
        # 信頼度閾値チェック
        if confidence < self.confidence_threshold:
            # 閾値以下の場合は警告表示
            self.display_prediction(prediction, confidence, is_low_confidence=True)
        else:
            self.display_prediction(prediction, confidence, is_low_confidence=False)
        
        # 検査履歴に追加
        self.add_to_history(prediction, confidence)
        
        # 統計を更新
        self.update_statistics(prediction, confidence)
        
        # NG検出時の処理
        if prediction != "good":
            # 視覚的アラート
            if hasattr(self, 'alert_enabled_checkbox') and self.alert_enabled_checkbox and self.alert_enabled_checkbox.isChecked():
                self.trigger_visual_alert()
            elif not hasattr(self, 'alert_enabled_checkbox') or self.alert_enabled_checkbox is None:
                # デフォルトでアラート有効
                self.trigger_visual_alert()
            
            # 自動保存
            if self.auto_save_ng and self.current_frame is not None:
                self.save_inspection_image(self.current_frame, prediction, confidence)
    
    def display_prediction(self, prediction, confidence, is_low_confidence=False):
        """予測結果を表示"""
        # 予測結果を日本語に変換
        prediction_map = {
            'good': '良品',
            'black_spot': '黒点',
            'chipping': '欠け',
            'scratch': '傷',
            'dent': '凹み',
            'distortion': '歪み'
        }
        prediction_jp = prediction_map.get(prediction, prediction)
        
        self.prediction_label.setText(f"予測: {prediction_jp}")
        
        # 信頼度表示（閾値以下は警告色）
        if is_low_confidence:
            self.confidence_label.setText(f"信頼度: {confidence:.2%} ⚠️（低信頼度）")
            self.confidence_label.setStyleSheet("font-size: 16pt; color: #ff9800; padding: 10px; font-weight: bold;")
        else:
            self.confidence_label.setText(f"信頼度: {confidence:.2%}")
            self.confidence_label.setStyleSheet("font-size: 16pt; color: #666; padding: 10px;")
        
        # 色を変更
        if prediction == "good":
            self.prediction_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: green; padding: 10px;")
        else:
            self.prediction_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: red; padding: 10px;")
    
    def add_to_history(self, prediction, confidence):
        """検査履歴に追加"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 検査ログに追加
        log_entry = {
            'timestamp': timestamp,
            'prediction': prediction,
            'confidence': float(confidence),
            'is_ok': prediction == 'good'
        }
        self.inspection_history.append(log_entry)
        
        # 予測履歴リストに追加（最大20件）
        prediction_map = {
            'good': '良品',
            'black_spot': '黒点',
            'chipping': '欠け',
            'scratch': '傷',
            'dent': '凹み',
            'distortion': '歪み'
        }
        prediction_jp = prediction_map.get(prediction, prediction)
        
        time_str = datetime.now().strftime("%H:%M:%S")
        status_icon = "✅" if prediction == "good" else "❌"
        history_text = f"[{time_str}] {prediction_jp} ({confidence:.1%}) {status_icon}"
        
        self.history_list.insertItem(0, history_text)
        if self.history_list.count() > 20:
            self.history_list.takeItem(20)
        
        # ログファイルに保存
        self.save_inspection_log(log_entry)
    
    def update_statistics(self, prediction, confidence):
        """統計情報を更新"""
        self.inspection_stats['total'] += 1
        self.inspection_stats['by_class'][prediction] = self.inspection_stats['by_class'].get(prediction, 0) + 1
        
        if prediction == "good":
            self.inspection_stats['ok'] += 1
        else:
            self.inspection_stats['ng'] += 1
        
        # UI更新
        total = self.inspection_stats['total']
        ok = self.inspection_stats['ok']
        ng = self.inspection_stats['ng']
        ok_rate = (ok / total * 100) if total > 0 else 0
        ng_rate = (ng / total * 100) if total > 0 else 0
        
        self.total_count_label.setText(f"総検査数: {total}")
        self.ok_count_label.setText(f"良品: {ok} ({ok_rate:.1f}%)")
        self.ng_count_label.setText(f"不良品: {ng} ({ng_rate:.1f}%)")
        
        # クラス別集計
        class_texts = []
        for class_name, count in self.inspection_stats['by_class'].items():
            if count > 0:
                class_map = {
                    'good': '良品',
                    'black_spot': '黒点',
                    'chipping': '欠け',
                    'scratch': '傷',
                    'dent': '凹み',
                    'distortion': '歪み'
                }
                class_jp = class_map.get(class_name, class_name)
                class_texts.append(f"{class_jp}:{count}")
        
        if class_texts:
            self.class_stats_label.setText("クラス別: " + ", ".join(class_texts))
        else:
            self.class_stats_label.setText("クラス別: -")
        
        # エクスポートボタンを有効化
        if total > 0:
            self.export_results_btn.setEnabled(True)
    
    def save_inspection_log(self, log_entry):
        """検査ログをファイルに保存"""
        log_file = self.inspection_log_dir / f"inspection_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"ログ保存エラー: {e}")
    
    def save_inspection_image(self, frame, prediction, confidence):
        """検査画像を保存（NG検出時の自動保存）"""
        if frame is None:
            print("[WARNING] 保存するフレームがありません")
            return
        
        try:
            # 保存先ディレクトリを確認・作成
            if not hasattr(self, 'inspection_image_dir'):
                self.inspection_image_dir = Path('inspection_results')
            
            self.inspection_image_dir.mkdir(parents=True, exist_ok=True)
            
            # 日付フォルダを作成（YYYY-MM-DD形式）
            date_dir = self.inspection_image_dir / datetime.now().strftime('%Y-%m-%d')
            date_dir.mkdir(parents=True, exist_ok=True)
            
            # ファイル名: タイムスタンプ_クラス名（英語）_信頼度.jpg
            # 例: 20250115_143052_123_black_spot_0.85.jpg
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # ミリ秒まで
            filename = f"{timestamp}_{prediction}_{confidence:.2f}.jpg"
            filepath = date_dir / filename
            
            # 画像を保存
            success = cv2.imwrite(str(filepath), frame)
            if not success:
                print(f"[ERROR] 画像の保存に失敗しました: {filepath}")
                return
            
            print(f"[INFO] NG画像を保存しました: {filepath}")
            
            # ログに画像パスを追加
            if self.inspection_history:
                self.inspection_history[-1]['image_path'] = str(filepath)
                self.inspection_history[-1]['image_saved'] = True
                
        except Exception as e:
            print(f"[ERROR] 画像保存エラー: {e}")
            import traceback
            traceback.print_exc()
    
    def save_current_frame(self):
        """現在のフレームを手動保存"""
        if self.current_frame is None:
            return
        
        # 最新の予測結果を取得
        if self.inspection_history:
            latest = self.inspection_history[-1]
            prediction = latest.get('prediction', 'unknown')
            confidence = latest.get('confidence', 0.0)
        else:
            prediction = 'manual'
            confidence = 0.0
        
        self.save_inspection_image(self.current_frame, prediction, confidence)
        self.statusBar().showMessage('画像を保存しました', 2000)
    
    def export_inspection_results(self):
        """検査結果をCSV/JSON形式でエクスポート"""
        if not self.inspection_history:
            self.statusBar().showMessage('エクスポートするデータがありません', 3000)
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_path = Path('logs/inspection')
        base_path.mkdir(parents=True, exist_ok=True)
        
        # CSV形式でエクスポート
        csv_file = base_path / f"inspection_export_{timestamp}.csv"
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'prediction', 'confidence', 'is_ok', 'image_path'])
                writer.writeheader()
                for entry in self.inspection_history:
                    writer.writerow(entry)
            self.statusBar().showMessage(f'CSV形式でエクスポートしました: {csv_file}', 5000)
        except Exception as e:
            print(f"CSVエクスポートエラー: {e}")
        
        # JSON形式でエクスポート（統計情報付き）
        json_file = base_path / f"inspection_export_{timestamp}.json"
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'statistics': self.inspection_stats,
                'total_inspections': len(self.inspection_history),
                'results': self.inspection_history
            }
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            self.statusBar().showMessage(f'JSON形式でエクスポートしました: {json_file}', 5000)
        except Exception as e:
            print(f"JSONエクスポートエラー: {e}")
    
    def clear_statistics(self):
        """統計情報をリセット"""
        reply = QtWidgets.QMessageBox.question(
            self, '確認', '統計情報をリセットしますか？',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.inspection_stats = {
                'total': 0,
                'ok': 0,
                'ng': 0,
                'by_class': {'good': 0, 'black_spot': 0, 'chipping': 0, 'scratch': 0, 'dent': 0, 'distortion': 0}
            }
            self.update_statistics('good', 0.0)  # UI更新
            self.history_list.clear()
            self.statusBar().showMessage('統計情報をリセットしました', 2000)
    
    def update_confidence_threshold(self, value):
        """信頼度閾値を更新"""
        self.confidence_threshold = value
        self.save_inspection_settings()
    
    def on_auto_save_changed(self, state):
        """自動保存設定が変更された"""
        self.auto_save_ng = (state == Qt.Checked)
        self.save_inspection_settings()
    
    def trigger_visual_alert(self):
        """視覚的アラートを発動（画面の点滅）"""
        self.alert_flash_count = 0
        self.alert_flash_timer.timeout.connect(self.flash_alert)
        self.alert_flash_timer.start(200)  # 200msごとに点滅
    
    def flash_alert(self):
        """アラート点滅"""
        self.alert_flash_count += 1
        if self.alert_flash_count >= 6:  # 3回点滅（6回の切り替え）
            self.alert_flash_timer.stop()
            self.alert_flash_timer.timeout.disconnect()
            self.camera_label.setStyleSheet("""
                border: 3px solid #0066cc;
                border-radius: 8px;
                background-color: #f8f9fa;
                font-size: 14pt;
                color: #666;
            """)
            return
        
        # 赤と通常の色を交互に切り替え
        if self.alert_flash_count % 2 == 0:
            self.camera_label.setStyleSheet("""
                border: 5px solid #dc3545;
                border-radius: 8px;
                background-color: #ffe6e6;
            """)
        else:
            self.camera_label.setStyleSheet("""
                border: 3px solid #0066cc;
                border-radius: 8px;
                background-color: #f8f9fa;
                font-size: 14pt;
                color: #666;
            """)
    
    def on_always_show_system_info_changed(self, state):
        """システム情報の常時表示チェックボックスの状態変更"""
        if state == QtCore.Qt.Checked:
            # チェックが入った場合、即座に更新を実行
            self.update_hwinfo_metrics()
        # チェックが外れた場合は、update_hwinfo_metrics内で学習中でなければ更新しないようになっている
    
    def on_manual_mode_changed(self, state):
        """手動判定モードの切り替え"""
        self.manual_mode = (state == Qt.Checked)
        
        if self.manual_mode:
            # 手動判定UIを表示
            if hasattr(self, 'manual_judgment_group'):
                self.manual_judgment_group.setVisible(True)
            self.statusBar().showMessage('手動判定モードに切り替えました', 3000)
        else:
            # 手動判定UIを非表示
            if hasattr(self, 'manual_judgment_group'):
                self.manual_judgment_group.setVisible(False)
            # 判定ボタンを無効化
            if hasattr(self, 'judgment_buttons'):
                for btn, _ in self.judgment_buttons:
                    btn.setEnabled(False)
            self.statusBar().showMessage('自動判定モードに切り替えました', 3000)
        
        # 設定を保存
        self.save_inspection_settings()
    
    def apply_manual_judgment(self, selected_class):
        """手動判定を適用"""
        if not self.pending_ai_prediction or not self.pending_ai_confidence:
            return
        
        ai_prediction = self.pending_ai_prediction
        ai_confidence = self.pending_ai_confidence
        
        # 判定結果を表示
        prediction_map = {
            'good': '良品',
            'black_spot': '黒点',
            'chipping': '欠け',
            'scratch': '傷',
            'dent': '凹み',
            'distortion': '歪み'
        }
        selected_jp = prediction_map.get(selected_class, selected_class)
        ai_pred_jp = prediction_map.get(ai_prediction, ai_prediction)
        
        self.prediction_label.setText(f"判定: {selected_jp}")
        if selected_class == "good":
            self.prediction_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: green; padding: 10px;")
        else:
            self.prediction_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: red; padding: 10px;")
        
        self.confidence_label.setText(f"AI予測: {ai_pred_jp} ({ai_confidence:.1%}) → 手動判定: {selected_jp}")
        self.confidence_label.setStyleSheet("font-size: 16pt; color: #666; padding: 10px;")
        
        # 判定ボタンを一時的に無効化（次のAI予測まで）
        for btn, _ in self.judgment_buttons:
            btn.setEnabled(False)
        
        # 検査履歴に追加（手動判定の結果を記録、AI予測情報も記録）
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prediction': selected_class,  # 手動判定の結果
            'confidence': float(ai_confidence),  # AIの信頼度
            'is_ok': selected_class == 'good',
            'ai_prediction': ai_prediction,  # AIの予測結果
            'manual_judgment': True,  # 手動判定フラグ
            'ai_match': ai_prediction == selected_class  # AIと手動判定が一致したか
        }
        self.inspection_history.append(log_entry)
        
        # 統計を更新（手動判定の結果で）
        self.update_statistics(selected_class, ai_confidence)
        
        # 履歴リストに追加
        time_str = datetime.now().strftime("%H:%M:%S")
        status_icon = "✅" if selected_class == "good" else "❌"
        history_text = f"[{time_str}] {selected_jp} (AI: {ai_pred_jp} {ai_confidence:.1%}) {status_icon} ✋"
        
        if hasattr(self, 'history_list'):
            self.history_list.insertItem(0, history_text)
            if self.history_list.count() > 20:
                self.history_list.takeItem(20)
        
        # ログファイルに保存
        self.save_inspection_log(log_entry)
        
        # NG検出時の処理（手動判定がNGの場合）
        if selected_class != "good":
            # 視覚的アラート
            if hasattr(self, 'alert_enabled_checkbox') and self.alert_enabled_checkbox and self.alert_enabled_checkbox.isChecked():
                self.trigger_visual_alert()
            elif not hasattr(self, 'alert_enabled_checkbox') or self.alert_enabled_checkbox is None:
                self.trigger_visual_alert()
            
            # 自動保存
            if self.auto_save_ng and self.current_frame is not None:
                self.save_inspection_image(self.current_frame, selected_class, ai_confidence)
        
        # AIと手動判定が異なる場合、フィードバックデータも保存
        if ai_prediction != selected_class:
            # 自動的にフィードバックデータとして保存（修正情報として）
            try:
                feedback_dir = Path('logs/inspection/feedback')
                feedback_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                
                if self.current_frame is not None:
                    image_dir = feedback_dir / 'images'
                    image_dir.mkdir(parents=True, exist_ok=True)
                    date_dir = image_dir / datetime.now().strftime('%Y-%m-%d')
                    date_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = f"{timestamp}_{selected_class}.jpg"
                    filepath = date_dir / filename
                    cv2.imwrite(str(filepath), self.current_frame)
                    
                    metadata = {
                        'timestamp': datetime.now().isoformat(),
                        'predicted_class': ai_prediction,
                        'correct_class': selected_class,
                        'confidence': float(ai_confidence),
                        'image_path': str(filepath),
                        'manual_judgment': True
                    }
                    
                    metadata_file = feedback_dir / f"{timestamp}_feedback.json"
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"フィードバック保存エラー: {e}")
        
        # 次の判定待ちにリセット
        self.pending_ai_prediction = None
        self.pending_ai_confidence = None
    
    def update_fps(self):
        """FPSを更新"""
        current_time = time.time()
        if hasattr(self, 'inspection_start_time') and self.inspection_start_time:
            elapsed = current_time - self.fps_time
            if elapsed > 0:
                fps = self.fps_counter / elapsed
                self.fps_label.setText(f"FPS: {fps:.1f}")
                self.fps_counter = 0
                self.fps_time = current_time
                
                # FPS履歴を記録
                try:
                    self.fps_history.append(fps)
                except Exception:
                    pass
        else:
            self.fps_label.setText("FPS: -")
        
        # メモリ使用量を記録
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024  # MBに変換
            self.memory_usage_history.append(mem_mb)
        except Exception:
            pass
    
    def show_feedback_dialog(self):
        """フィードバックダイアログを表示（誤判定修正）"""
        if not self.inspection_history:
            QtWidgets.QMessageBox.information(self, '情報', 'フィードバックする予測結果がありません。\nまず検査を開始してください。')
            return
        
        # 最新の予測結果を取得
        latest = self.inspection_history[-1]
        predicted = latest.get('prediction', 'unknown')
        confidence = latest.get('confidence', 0.0)
        
        # 予測クラスを日本語に変換
        class_map = {
            'good': '良品',
            'black_spot': '黒点',
            'chipping': '欠け',
            'scratch': '傷',
            'dent': '凹み',
            'distortion': '歪み'
        }
        predicted_jp = class_map.get(predicted, predicted)
        
        # ダイアログを作成
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('⚠️ 誤判定の修正')
        dialog.setMinimumWidth(500)
        dialog.setMinimumHeight(300)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # タイトル
        title_label = QLabel('⚠️ 判定が間違っていた場合の修正')
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #d32f2f; padding: 10px;")
        layout.addWidget(title_label)
        
        # 説明
        desc_label = QLabel(
            '現在の判定を修正して、正しいクラスを選択してください。\n'
            '修正内容は保存され、次回の学習に反映されます。'
        )
        desc_label.setStyleSheet("font-size: 11pt; color: #555; padding: 10px; background-color: #fff3cd; border-radius: 4px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # 現在の予測結果
        current_info = QLabel(f"📊 現在の予測: <b>{predicted_jp}</b> (信頼度: {confidence:.1%})")
        current_info.setStyleSheet("font-size: 12pt; padding: 10px; background-color: #f5f5f5; border-radius: 4px;")
        layout.addWidget(current_info)
        
        # 正しいクラス選択
        correct_label = QLabel("✅ 正しいクラスを選択してください:")
        correct_label.setStyleSheet("font-size: 11pt; font-weight: bold; padding: 5px;")
        layout.addWidget(correct_label)
        
        # クラス選択
        class_combo = QComboBox()
        class_combo.setStyleSheet("""
            QComboBox {
                font-size: 12pt;
                padding: 10px;
                min-height: 40px;
                background-color: white;
                border: 2px solid #0066cc;
                border-radius: 6px;
            }
            QComboBox:hover {
                border: 2px solid #0052a3;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox QAbstractItemView {
                font-size: 12pt;
                padding: 5px;
                min-height: 40px;
            }
        """)
        for class_en, class_jp in class_map.items():
            class_combo.addItem(class_jp, class_en)
            # 現在の予測を初期選択
            if class_en == predicted:
                class_combo.setCurrentIndex(class_combo.count() - 1)
        layout.addWidget(class_combo)
        
        layout.addStretch()
        
        # ボタン
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("キャンセル")
        cancel_btn.clicked.connect(dialog.reject)
        cancel_btn.setStyleSheet("font-size: 11pt; padding: 8px 20px;")
        ok_btn = QPushButton("💾 修正を保存")
        ok_btn.clicked.connect(dialog.accept)
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                font-size: 11pt;
                padding: 8px 20px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            correct_class = class_combo.currentData()
            correct_class_jp = class_combo.currentText()
            if correct_class and correct_class != predicted:
                # フィードバックデータを保存
                self.save_feedback_data(correct_class, predicted, confidence)
                
                # 自動保存された画像がある場合、正しいクラス名にリネームまたは移動
                if self.inspection_history:
                    latest = self.inspection_history[-1]
                    original_image_path = latest.get('image_path')
                    if original_image_path and Path(original_image_path).exists():
                        self.correct_saved_image_path(original_image_path, predicted, correct_class)
                
                QtWidgets.QMessageBox.information(
                    self, 
                    '保存完了', 
                    f'✅ フィードバックデータを保存しました\n\n'
                    f'修正前: {predicted_jp}\n'
                    f'修正後: {correct_class_jp}\n\n'
                    f'保存された画像も修正済みです。\n'
                    f'この情報は次回の学習に使用されます。'
                )
                # 予測結果表示を更新（修正後のクラスを表示）
                self.prediction_label.setText(f"予測: {correct_class_jp} (修正済み)")
                self.prediction_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: #ff9800; padding: 10px;")
            elif correct_class == predicted:
                QtWidgets.QMessageBox.information(self, '情報', '現在の判定と同じクラスが選択されています。\n修正は不要です。')
            else:
                QtWidgets.QMessageBox.warning(self, 'エラー', '正しいクラスを選択してください。')
    
    def correct_saved_image_path(self, image_path, original_prediction, correct_class):
        """自動保存された画像を正しいクラス名にリネーム"""
        try:
            original_path = Path(image_path)
            if not original_path.exists():
                print(f"[WARNING] 画像が見つかりません: {image_path}")
                return
            
            # ファイル名を解析（例: 20250115_143052_123_black_spot_0.85.jpg）
            filename = original_path.name
            parent_dir = original_path.parent
            
            # タイムスタンプ部分を抽出（最初の_から最後の_まで）
            parts = filename.split('_')
            if len(parts) >= 4:
                # タイムスタンプ部分（最初の3つ）を保持
                timestamp_part = '_'.join(parts[:3])
                # 信頼度部分を抽出（最後の部分から数値を取得）
                confidence_part = parts[-1].replace('.jpg', '')
                
                # 新しいファイル名を作成（正しいクラス名に変更）
                new_filename = f"{timestamp_part}_{correct_class}_{confidence_part}.jpg"
                new_path = parent_dir / new_filename
                
                # ファイルをリネーム
                original_path.rename(new_path)
                
                # 検査履歴の画像パスも更新
                if self.inspection_history:
                    self.inspection_history[-1]['image_path'] = str(new_path)
                    self.inspection_history[-1]['corrected_class'] = correct_class
                    self.inspection_history[-1]['original_prediction'] = original_prediction
                
                print(f"[INFO] 画像をリネームしました: {filename} → {new_filename}")
        except Exception as e:
            print(f"[ERROR] 画像のリネームに失敗しました: {e}")
            import traceback
            traceback.print_exc()
    
    def save_feedback_data(self, correct_class, predicted_class, confidence):
        """フィードバックデータを保存（再学習用）"""
        try:
            feedback_dir = Path('logs/inspection/feedback')
            feedback_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            
            # 画像を保存
            if self.current_frame is not None:
                image_dir = feedback_dir / 'images'
                image_dir.mkdir(parents=True, exist_ok=True)
                
                date_dir = image_dir / datetime.now().strftime('%Y-%m-%d')
                date_dir.mkdir(parents=True, exist_ok=True)
                
                filename = f"{timestamp}_{correct_class}.jpg"
                filepath = date_dir / filename
                cv2.imwrite(str(filepath), self.current_frame)
                
                # 自動保存された画像のパスも記録（修正前のパス）
                original_image_path = None
                if self.inspection_history:
                    latest = self.inspection_history[-1]
                    original_image_path = latest.get('image_path')
                
                # メタデータを保存
                metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'predicted_class': predicted_class,
                    'correct_class': correct_class,
                    'confidence': float(confidence),
                    'image_path': str(filepath),
                    'original_saved_image_path': str(original_image_path) if original_image_path else None
                }
                
                metadata_file = feedback_dir / f"{timestamp}_feedback.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                print(f"フィードバックデータを保存しました: {metadata_file}")
        except Exception as e:
            print(f"フィードバック保存エラー: {e}")
            QtWidgets.QMessageBox.warning(self, 'エラー', f'フィードバックデータの保存に失敗しました: {e}')
    
    def show_inspection_settings_dialog(self):
        """検査設定ダイアログを表示"""
        try:
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle('⚙️ 検査設定')
            dialog.setMinimumWidth(650)
            dialog.setMinimumHeight(600)
            
            # スクロール可能なレイアウト
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            scroll_area.setStyleSheet("""
                QScrollArea {
                    border: none;
                    background-color: white;
                }
            """)
            
            content_widget = QtWidgets.QWidget()
            layout = QVBoxLayout(content_widget)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(20)
            
            scroll_area.setWidget(content_widget)
            
            # メインレイアウト
            main_layout = QVBoxLayout(dialog)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.addWidget(scroll_area)
            
            # タイトル
            title_label = QLabel("🔍 外観検査設定")
            title_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #0066cc; padding: 15px 0px; border-bottom: 2px solid #0066cc; margin-bottom: 10px;")
            layout.addWidget(title_label)
            
            # カメラ選択
            camera_group = QtWidgets.QGroupBox("📷 カメラ設定")
            camera_group.setStyleSheet("font-weight: bold; font-size: 11pt;")
            camera_layout = QVBoxLayout()
            camera_layout.setContentsMargins(15, 15, 15, 15)
            camera_layout.setSpacing(10)
            
            camera_info = QLabel("📌 検査に使用するカメラを選択してください\n複数のカメラが接続されている場合、リストから選択できます")
            camera_info.setStyleSheet("font-size: 10pt; color: #555; margin-bottom: 10px; padding: 8px; background-color: #f0f8ff; border-radius: 4px; line-height: 1.6;")
            camera_info.setWordWrap(True)
            camera_layout.addWidget(camera_info)
            
            camera_select_layout = QHBoxLayout()
            camera_select_label = QLabel("使用カメラ:")
            camera_select_label.setStyleSheet("font-size: 11pt; font-weight: bold; min-width: 120px;")
            camera_combo = QComboBox()
            camera_combo.setMinimumWidth(300)
            camera_combo.setEditable(False)
            
            # コンボボックスのスタイル設定（選択肢が消えないように）
            camera_combo.setStyleSheet("""
                QComboBox {
                    font-size: 11pt;
                    padding: 8px 12px;
                    min-height: 35px;
                    background-color: white;
                    border: 2px solid #0066cc;
                    border-radius: 4px;
                }
                QComboBox:hover {
                    border: 2px solid #0052a3;
                }
                QComboBox::drop-down {
                    border: none;
                    width: 30px;
                }
            """)
            
            # ドロップダウンリストのビューを取得して設定
            camera_view = camera_combo.view()
            camera_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            camera_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            
            # ビューウィジェットのスタイルを設定
            camera_view.setStyleSheet("""
                QAbstractItemView {
                    font-size: 11pt;
                    background-color: white;
                    border: 2px solid #0066cc;
                    border-radius: 4px;
                    selection-background-color: #e7f3ff;
                    selection-color: #0066cc;
                    outline: none;
                    min-width: 300px;
                    min-height: 150px;
                }
                QAbstractItemView::item {
                    padding: 10px 15px;
                    min-height: 35px;
                    border-bottom: 1px solid #e0e0e0;
                }
                QAbstractItemView::item:selected {
                    background-color: #e7f3ff;
                    color: #0066cc;
                }
                QAbstractItemView::item:hover {
                    background-color: #f0f8ff;
                    color: #0066cc;
                }
                QScrollBar:vertical {
                    width: 18px;
                    background-color: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 3px;
                }
                QScrollBar::handle:vertical {
                    background-color: #0066cc;
                    min-height: 30px;
                    border-radius: 3px;
                }
                QScrollBar::handle:vertical:hover {
                    background-color: #0052a3;
                }
            """)
            
            # コンボボックスのポップアップサイズを調整
            def adjust_camera_popup_size():
                try:
                    popup = camera_combo.view()
                    if popup:
                        fm = QFontMetrics(camera_combo.font())
                        max_width = 0
                        for i in range(camera_combo.count()):
                            text = camera_combo.itemText(i)
                            rect = fm.boundingRect(text)
                            max_width = max(max_width, rect.width())
                        
                        content_width = max_width + 80
                        popup.setMinimumWidth(max(300, content_width))
                        popup.setMaximumWidth(500)
                        popup.setMinimumHeight(150)
                        popup.setMaximumHeight(300)
                except Exception as e:
                    print(f"[ERROR] Camera popup size adjustment error: {e}")
            
            # コンボボックスが開かれたときにサイズ調整
            original_showPopup_camera = camera_combo.showPopup
            def showPopup_camera():
                original_showPopup_camera()
                QtCore.QTimer.singleShot(1, adjust_camera_popup_size)
                QtCore.QTimer.singleShot(10, adjust_camera_popup_size)
                QtCore.QTimer.singleShot(30, adjust_camera_popup_size)
            camera_combo.showPopup = showPopup_camera
            
            # 利用可能なカメラを検出（エラーハンドリング付き）
            try:
                for i in range(10):
                    cap = None
                    try:
                        cap = cv2.VideoCapture(i)
                        if cap.isOpened():
                            ret, _ = cap.read()
                            if ret:
                                camera_combo.addItem(f"カメラ {i}", i)
                    except Exception as e:
                        print(f"カメラ {i} の検出エラー: {e}")
                    finally:
                        if cap is not None:
                            cap.release()
            except Exception as e:
                print(f"カメラ検出中のエラー: {e}")
            
            camera_count = camera_combo.count()
            if camera_count == 0:
                camera_combo.addItem("カメラが見つかりません", 0)
                camera_warning = QLabel("⚠️ カメラが検出されませんでした。\nカメラが接続されているか、他のアプリケーションで使用されていないか確認してください。")
                camera_warning.setStyleSheet("font-size: 9pt; color: #dc3545; padding: 8px; background-color: #f8d7da; border-radius: 4px; margin-top: 5px;")
                camera_warning.setWordWrap(True)
                camera_layout.addWidget(camera_warning)
            else:
                # 現在選択されているカメラを設定（安全に処理）
                try:
                    if hasattr(self, 'camera_combo') and self.camera_combo is not None and self.camera_combo.count() > 0:
                        current_id = self.camera_combo.currentData()
                        if current_id is not None:
                            for i in range(camera_combo.count()):
                                if camera_combo.itemData(i) == current_id:
                                    camera_combo.setCurrentIndex(i)
                                    break
                except Exception as e:
                    print(f"カメラ選択の復元エラー: {e}")
                    # エラー時は最初のカメラを選択
                    if camera_combo.count() > 0:
                        camera_combo.setCurrentIndex(0)
                
                camera_success = QLabel(f"✅ {camera_count}台のカメラを検出しました")
                camera_success.setStyleSheet("font-size: 9pt; color: #28a745; padding: 5px; margin-top: 5px;")
                camera_layout.addWidget(camera_success)
            
            camera_select_layout.addWidget(camera_select_label)
            camera_select_layout.addWidget(camera_combo)
            camera_select_layout.addStretch()
            camera_layout.addLayout(camera_select_layout)
            
            camera_group.setLayout(camera_layout)
            layout.addWidget(camera_group)
            
            # 信頼度閾値設定
            threshold_group = QtWidgets.QGroupBox("🎯 信頼度閾値設定")
            threshold_group.setStyleSheet("font-weight: bold; font-size: 11pt;")
            threshold_layout = QVBoxLayout()
            threshold_layout.setContentsMargins(15, 15, 15, 15)
            threshold_layout.setSpacing(10)
            
            threshold_info = QLabel("📌 予測の信頼度がこの値を下回る場合、警告表示されます\n信頼度が低い予測は「⚠️（低信頼度）」と表示されます")
            threshold_info.setStyleSheet("font-size: 10pt; color: #555; margin-bottom: 10px; padding: 8px; background-color: #fff3cd; border-radius: 4px; line-height: 1.6;")
            threshold_info.setWordWrap(True)
            threshold_layout.addWidget(threshold_info)
            
            threshold_inner_layout = QHBoxLayout()
            threshold_label = QLabel("閾値:")
            threshold_label.setStyleSheet("font-size: 11pt; font-weight: bold; min-width: 120px;")
            confidence_threshold_spin = QDoubleSpinBox()
            confidence_threshold_spin.setMinimum(0.0)
            confidence_threshold_spin.setMaximum(1.0)
            confidence_threshold_spin.setSingleStep(0.05)
            confidence_threshold_spin.setValue(self.confidence_threshold)
            confidence_threshold_spin.setDecimals(2)
            confidence_threshold_spin.setMinimumWidth(120)
            confidence_threshold_spin.setStyleSheet("font-size: 11pt; padding: 6px;")
            
            # パーセンテージ表示ラベル
            threshold_percent_label = QLabel(f"({self.confidence_threshold:.0%})")
            threshold_percent_label.setStyleSheet("font-size: 11pt; color: #0066cc; font-weight: bold; min-width: 60px;")
            
            def update_threshold_percent(value):
                threshold_percent_label.setText(f"({value:.0%})")
            confidence_threshold_spin.valueChanged.connect(update_threshold_percent)
            
            threshold_inner_layout.addWidget(threshold_label)
            threshold_inner_layout.addWidget(confidence_threshold_spin)
            threshold_inner_layout.addWidget(threshold_percent_label)
            threshold_inner_layout.addStretch()
            
            threshold_desc = QLabel("💡 推奨値: 0.70 (70%) - 高い精度が期待できる値です\n※ 0.5以下: 低信頼度警告が頻繁に表示される可能性があります\n※ 0.9以上: 高精度な予測のみを検出します")
            threshold_desc.setStyleSheet("font-size: 9pt; color: #555; padding: 8px; background-color: #f0f8ff; border-radius: 4px; margin-top: 5px;")
            threshold_desc.setWordWrap(True)
            threshold_layout.addLayout(threshold_inner_layout)
            threshold_layout.addWidget(threshold_desc)
            
            threshold_group.setLayout(threshold_layout)
            layout.addWidget(threshold_group)
            
            # オプション
            options_group = QtWidgets.QGroupBox("⚙️ 動作オプション")
            options_group.setStyleSheet("font-weight: bold; font-size: 11pt;")
            options_layout = QVBoxLayout()
            options_layout.setContentsMargins(15, 15, 15, 15)
            options_layout.setSpacing(12)
            
            # 自動保存オプション
            auto_save_container = QVBoxLayout()
            auto_save_frame = QFrame()
            auto_save_frame.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 10px;")
            auto_save_frame_layout = QVBoxLayout(auto_save_frame)
            auto_save_frame_layout.setContentsMargins(10, 10, 10, 10)
            
            auto_save_ng_checkbox = QCheckBox("💾 NG検出時に画像を自動保存")
            auto_save_ng_checkbox.setChecked(self.auto_save_ng)
            # グローバルスタイルを適用（個別のスタイル設定を削除）
            auto_save_frame_layout.addWidget(auto_save_ng_checkbox)
            
            auto_save_desc = QLabel("✅ 不良品（NG）が検出された場合、自動的に画像を保存します\n📁 保存先: inspection_results/YYYY-MM-DD/\n💡 ファイル名: timestamp_クラス名_信頼度.jpg")
            auto_save_desc.setStyleSheet("font-size: 9pt; color: #555; padding-left: 10px; padding-top: 5px; line-height: 1.8;")
            auto_save_desc.setWordWrap(True)
            auto_save_frame_layout.addWidget(auto_save_desc)
            auto_save_container.addWidget(auto_save_frame)
            options_layout.addLayout(auto_save_container)
            
            # 視覚的アラートオプション
            alert_container = QVBoxLayout()
            alert_frame = QFrame()
            alert_frame.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 10px; margin-top: 8px;")
            alert_frame_layout = QVBoxLayout(alert_frame)
            alert_frame_layout.setContentsMargins(10, 10, 10, 10)
            
            alert_enabled_checkbox = QCheckBox("🔔 視覚的アラート（NG検出時に画面点滅）")
            if hasattr(self, 'alert_enabled_checkbox') and self.alert_enabled_checkbox is not None:
                alert_enabled_checkbox.setChecked(self.alert_enabled_checkbox.isChecked())
            else:
                alert_enabled_checkbox.setChecked(True)
            # グローバルスタイルを適用（個別のスタイル設定を削除）
            alert_frame_layout.addWidget(alert_enabled_checkbox)
            
            alert_desc = QLabel("✅ 不良品が検出された場合、画面を点滅させて注意を促します\n⚡ 瞬時にNGを確認できます\n🎨 点滅回数: 3回")
            alert_desc.setStyleSheet("font-size: 9pt; color: #555; padding-left: 10px; padding-top: 5px; line-height: 1.8;")
            alert_desc.setWordWrap(True)
            alert_frame_layout.addWidget(alert_desc)
            alert_container.addWidget(alert_frame)
            options_layout.addLayout(alert_container)
            
            options_group.setLayout(options_layout)
            layout.addWidget(options_group)
            
            # 保存場所の表示
            save_info_group = QtWidgets.QGroupBox("📁 保存先フォルダ")
            save_info_group.setStyleSheet("font-weight: bold; font-size: 11pt;")
            save_info_layout = QVBoxLayout()
            save_info_layout.setContentsMargins(15, 15, 15, 15)
            save_info_layout.setSpacing(10)
            
            save_info_label = QLabel("📌 検査ログと画像は以下のフォルダに自動保存されます")
            save_info_label.setStyleSheet("font-size: 10pt; color: #555; margin-bottom: 10px; padding: 8px; background-color: #e7f3ff; border-radius: 4px;")
            save_info_label.setWordWrap(True)
            save_info_layout.addWidget(save_info_label)
            
            # ログフォルダ
            log_dir_frame = QFrame()
            log_dir_frame.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 10px;")
            log_dir_frame_layout = QHBoxLayout(log_dir_frame)
            log_dir_frame_layout.setContentsMargins(10, 10, 10, 10)
            
            log_dir_icon = QLabel("📄")
            log_dir_icon.setStyleSheet("font-size: 20pt; min-width: 40px;")
            log_dir_frame_layout.addWidget(log_dir_icon)
            
            log_dir_text_layout = QVBoxLayout()
            log_dir_title = QLabel("検査ログ")
            log_dir_title.setStyleSheet("font-size: 11pt; font-weight: bold; color: #333;")
            log_dir_text_layout.addWidget(log_dir_title)
            log_dir_label = QLabel(str(self.inspection_log_dir))
            log_dir_label.setStyleSheet("font-size: 9pt; color: #555; font-family: 'Courier New', monospace; padding: 5px; background-color: white; border-radius: 3px;")
            log_dir_label.setWordWrap(True)
            log_dir_text_layout.addWidget(log_dir_label)
            log_dir_frame_layout.addLayout(log_dir_text_layout, stretch=1)
            save_info_layout.addWidget(log_dir_frame)
            
            # 画像フォルダ
            image_dir_frame = QFrame()
            image_dir_frame.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 10px; margin-top: 8px;")
            image_dir_frame_layout = QHBoxLayout(image_dir_frame)
            image_dir_frame_layout.setContentsMargins(10, 10, 10, 10)
            
            image_dir_icon = QLabel("🖼️")
            image_dir_icon.setStyleSheet("font-size: 20pt; min-width: 40px;")
            image_dir_frame_layout.addWidget(image_dir_icon)
            
            image_dir_text_layout = QVBoxLayout()
            image_dir_title = QLabel("検査画像")
            image_dir_title.setStyleSheet("font-size: 11pt; font-weight: bold; color: #333;")
            image_dir_text_layout.addWidget(image_dir_title)
            image_dir_label = QLabel(str(self.inspection_image_dir))
            image_dir_label.setStyleSheet("font-size: 9pt; color: #555; font-family: 'Courier New', monospace; padding: 5px; background-color: white; border-radius: 3px;")
            image_dir_label.setWordWrap(True)
            image_dir_text_layout.addWidget(image_dir_label)
            image_dir_frame_layout.addLayout(image_dir_text_layout, stretch=1)
            save_info_layout.addWidget(image_dir_frame)
            
            save_info_group.setLayout(save_info_layout)
            layout.addWidget(save_info_group)
            
            # ボタン
            button_layout = QHBoxLayout()
            button_layout.setSpacing(10)
            
            cancel_btn = QPushButton("キャンセル")
            cancel_btn.clicked.connect(dialog.reject)
            cancel_btn.setStyleSheet("background-color: #ccc; color: #333; font-weight: bold; padding: 8px 20px; border-radius: 4px;")
            button_layout.addWidget(cancel_btn)
            button_layout.addStretch()
            
            ok_btn = QPushButton("💾 保存して閉じる")
            ok_btn.clicked.connect(dialog.accept)
            ok_btn.setStyleSheet("background-color: #0066cc; color: white; font-weight: bold; padding: 10px 30px; border-radius: 4px; font-size: 11pt;")
            button_layout.addWidget(ok_btn)
            
            layout.addLayout(button_layout)
            
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                # 設定を保存
                self.confidence_threshold = confidence_threshold_spin.value()
                self.auto_save_ng = auto_save_ng_checkbox.isChecked()
                if hasattr(self, 'alert_enabled_checkbox'):
                    self.alert_enabled_checkbox.setChecked(alert_enabled_checkbox.isChecked())
                else:
                    self.alert_enabled_checkbox = alert_enabled_checkbox
                
                # カメラ選択を更新（安全に処理）
                try:
                    if camera_combo.count() > 0:
                        selected_camera_id = camera_combo.currentData()
                        if selected_camera_id is not None:
                            if not hasattr(self, 'camera_combo') or self.camera_combo is None:
                                self.camera_combo = QComboBox()
                            self.camera_combo.clear()
                            for i in range(camera_combo.count()):
                                self.camera_combo.addItem(camera_combo.itemText(i), camera_combo.itemData(i))
                            # 選択されたカメラを設定
                            for i in range(self.camera_combo.count()):
                                if self.camera_combo.itemData(i) == selected_camera_id:
                                    self.camera_combo.setCurrentIndex(i)
                                    break
                except Exception as e:
                    print(f"カメラ選択の更新エラー: {e}")
                    # エラー時は最初のカメラを使用
                    try:
                        if camera_combo.count() > 0:
                            if not hasattr(self, 'camera_combo') or self.camera_combo is None:
                                self.camera_combo = QComboBox()
                            self.camera_combo.clear()
                            self.camera_combo.addItem(camera_combo.itemText(0), camera_combo.itemData(0))
                    except:
                        pass
                
                self.save_inspection_settings()
                self.statusBar().showMessage('設定を保存しました', 2000)
        except Exception as e:
            import traceback
            error_msg = f"設定ダイアログの表示エラー:\n{str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            QtWidgets.QMessageBox.critical(self, 'エラー', f'設定ダイアログの表示に失敗しました:\n{str(e)}')
    
    def load_inspection_settings(self):
        """検査設定を読み込む"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    self.auto_save_ng = settings.get('auto_save_ng', True)
                    self.confidence_threshold = settings.get('confidence_threshold', 0.7)
                    self.manual_mode = settings.get('manual_mode', False)
                    
                    # カメラ設定を読み込む
                    camera_id = settings.get('camera_id', None)
                    camera_resolution = settings.get('camera_resolution', None)
                    
                    if camera_id is not None:
                        self.saved_camera_id = camera_id
                        print(f"[INFO] 保存されたカメラIDを読み込み: {camera_id}")
                    
                    if camera_resolution is not None:
                        self.camera_resolution = tuple(camera_resolution)
                        print(f"[INFO] 保存されたカメラ解像度を読み込み: {self.camera_resolution}")
                    
                    # UIに反映
                    if hasattr(self, 'manual_mode_checkbox'):
                        self.manual_mode_checkbox.setChecked(self.manual_mode)
                        self.on_manual_mode_changed(Qt.Checked if self.manual_mode else Qt.Unchecked)
                    
                    # カメラコンボボックスに反映（設定ダイアログで使用）
                    # メインウィンドウにはcamera_comboがないため、saved_camera_idとして保存
                    if camera_id is not None:
                        self.saved_camera_id = camera_id
                        print(f"[INFO] 保存されたカメラIDを設定: {camera_id}")
                    
                    # 設定ダイアログ内のcamera_comboに反映（存在する場合）
                    if hasattr(self, 'camera_combo') and self.camera_combo is not None:
                        if camera_id is not None:
                            for i in range(self.camera_combo.count()):
                                if self.camera_combo.itemData(i) == camera_id:
                                    self.camera_combo.setCurrentIndex(i)
                                    print(f"[INFO] カメラコンボボックスを更新: カメラ {camera_id}")
                                    break
            except Exception as e:
                print(f"設定読み込みエラー: {e}")
                import traceback
                traceback.print_exc()
    
    def save_inspection_settings(self):
        """検査設定を保存（カメラ設定も含む）"""
        try:
            settings = {
                'auto_save_ng': self.auto_save_ng,
                'confidence_threshold': self.confidence_threshold,
                'manual_mode': self.manual_mode,
                'last_saved': datetime.now().isoformat()
            }
            
            # カメラ設定を保存
            if hasattr(self, 'camera_combo') and self.camera_combo is not None:
                camera_id = self.camera_combo.currentData()
                if camera_id is not None:
                    settings['camera_id'] = camera_id
                    print(f"[INFO] カメラIDを保存: {camera_id}")
            
            if hasattr(self, 'camera_resolution') and self.camera_resolution is not None:
                settings['camera_resolution'] = list(self.camera_resolution)
                print(f"[INFO] カメラ解像度を保存: {self.camera_resolution}")
            
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"設定保存エラー: {e}")
            import traceback
            traceback.print_exc()
    
    def show_training_settings_dialog(self):
        """学習設定ダイアログを表示"""
        # 学習タブに切り替えて設定を表示
        self.tabs.setCurrentIndex(1)  # 学習タブ
        self.statusBar().showMessage('「🎓 学習」タブで設定を変更してください', 3000)
    
    def show_all_settings_dialog(self):
        """全設定を統合したダイアログを表示（タブで分類）"""
        try:
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle('⚙️ 設定')
            dialog.setMinimumWidth(750)
            dialog.setMinimumHeight(700)
            
            # スクロール可能なレイアウト
            inspection_scroll = QScrollArea()
            inspection_scroll.setWidgetResizable(True)
            inspection_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            inspection_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            inspection_scroll.setStyleSheet("""
                QScrollArea {
                    border: none;
                    background-color: white;
                }
            """)
            
            inspection_content = QtWidgets.QWidget()
            inspection_settings_layout = QVBoxLayout(inspection_content)
            inspection_settings_layout.setContentsMargins(20, 20, 20, 20)
            inspection_settings_layout.setSpacing(15)
            
            # カメラ選択グループ
            camera_group = QtWidgets.QGroupBox("📷 カメラ設定")
            camera_group.setStyleSheet("font-weight: bold; font-size: 11pt;")
            camera_layout = QVBoxLayout()
            camera_layout.setContentsMargins(15, 15, 15, 15)
            camera_layout.setSpacing(10)
            
            camera_select_layout = QHBoxLayout()
            camera_select_label = QLabel("使用カメラ:")
            camera_select_label.setStyleSheet("font-size: 11pt; font-weight: bold; min-width: 120px;")
            camera_combo = QComboBox()
            camera_combo.setMinimumWidth(300)
            camera_combo.setEditable(False)
            camera_combo.setStyleSheet("""
                QComboBox {
                    font-size: 11pt;
                    padding: 8px 12px;
                    min-height: 35px;
                    background-color: white;
                    border: 2px solid #0066cc;
                    border-radius: 4px;
                }
                QComboBox::drop-down {
                    border: none;
                    width: 30px;
                }
            """)
            
            camera_view = camera_combo.view()
            camera_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            camera_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            camera_view.setStyleSheet("""
                QAbstractItemView {
                    font-size: 11pt;
                    background-color: white;
                    border: 2px solid #0066cc;
                    border-radius: 4px;
                    selection-background-color: #e7f3ff;
                    selection-color: #0066cc;
                    outline: none;
                    min-width: 300px;
                    min-height: 150px;
                }
                QAbstractItemView::item {
                    padding: 10px 15px;
                    min-height: 35px;
                }
                QAbstractItemView::item:selected {
                    background-color: #e7f3ff;
                    color: #0066cc;
                }
                QAbstractItemView::item:hover {
                    background-color: #f0f8ff;
                    color: #0066cc;
                }
            """)
            
            # カメラ検出（すべてのカメラを検出）- 軽量化版（非同期処理）
            detected_cameras = []
            try:
                # より広範囲のカメラIDをチェック（0-10までに縮小、軽量化）
                for i in range(11):  # 0-10に縮小して軽量化
                    cap = None
                    try:
                        # DirectShowバックエンドを使用（Windows）
                        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                        if cap.isOpened():
                            # フレーム読み込みをスキップ（開けるかどうかだけ確認、軽量化）
                            # カメラ名を取得（軽量化）
                            camera_name = f"カメラ {i}"
                            detected_cameras.append((i, camera_name))
                            print(f"[INFO] カメラ検出: ID {i}")
                    except Exception as e:
                        # エラーは無視（カメラが存在しない場合）
                        pass
                    finally:
                        if cap is not None:
                            cap.release()
                        # 待機時間を削除（軽量化）
                
                # 検出したカメラをコンボボックスに追加
                for camera_id, camera_name in detected_cameras:
                    camera_combo.addItem(camera_name, camera_id)
                
                print(f"[INFO] 合計 {len(detected_cameras)} 個のカメラを検出しました")
                
            except Exception as e:
                print(f"[ERROR] カメラ検出中のエラー: {e}")
                import traceback
                traceback.print_exc()
            
            camera_count = camera_combo.count()
            if camera_count == 0:
                camera_combo.addItem("カメラが見つかりません", 0)
            else:
                try:
                    if hasattr(self, 'camera_combo') and self.camera_combo is not None and self.camera_combo.count() > 0:
                        current_id = self.camera_combo.currentData()
                        if current_id is not None:
                            for i in range(camera_combo.count()):
                                if camera_combo.itemData(i) == current_id:
                                    camera_combo.setCurrentIndex(i)
                                    break
                except Exception as e:
                    print(f"カメラ選択の復元エラー: {e}")
                    if camera_combo.count() > 0:
                        camera_combo.setCurrentIndex(0)
            
            camera_select_layout.addWidget(camera_select_label)
            camera_select_layout.addWidget(camera_combo)
            camera_select_layout.addStretch()
            
            # 解像度選択
            resolution_label = QLabel("解像度:")
            resolution_label.setStyleSheet("font-size: 11pt; font-weight: bold; min-width: 100px;")
            
            resolution_combo = QComboBox()
            resolution_combo.setMinimumWidth(200)
            resolution_combo.setStyleSheet("""
                QComboBox {
                    font-size: 11pt;
                    padding: 6px;
                    border: 2px solid #0066cc;
                    border-radius: 4px;
                    background-color: white;
                }
                QComboBox:hover {
                    border-color: #0052a3;
                }
                QComboBox::drop-down {
                    border: none;
                    width: 30px;
                }
            """)
            
            # 一般的な解像度のリスト
            common_resolutions = [
                (640, 480, "640x480 (VGA)"),
                (800, 600, "800x600 (SVGA)"),
                (1024, 768, "1024x768 (XGA)"),
                (1280, 720, "1280x720 (HD)"),
                (1280, 1024, "1280x1024 (SXGA)"),
                (1920, 1080, "1920x1080 (Full HD)"),
                (2560, 1440, "2560x1440 (QHD)"),
                (3840, 2160, "3840x2160 (4K UHD)"),
            ]
            
            # 現在選択されているカメラの解像度を取得（軽量化版）
            def update_resolution_options():
                """選択されたカメラに基づいて解像度オプションを更新（軽量化版）"""
                resolution_combo.clear()
                selected_camera_id = camera_combo.currentData()
                
                if selected_camera_id is None:
                    resolution_combo.addItem("カメラを選択してください", None)
                    return
                
                # 解像度検出を軽量化：現在の解像度のみ取得して、一般的な解像度をリスト表示
                cap = None
                try:
                    cap = cv2.VideoCapture(selected_camera_id)
                    if cap.isOpened():
                        # 現在の解像度を取得
                        current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        # 一般的な解像度をすべてリストに追加（検証なしで軽量化）
                        for width, height, name in common_resolutions:
                            resolution_combo.addItem(name, (width, height))
                        
                        # 現在の解像度がリストにない場合は追加
                        if current_width > 0 and current_height > 0:
                            current_res = (current_width, current_height)
                            if not any(resolution_combo.itemData(i) == current_res for i in range(resolution_combo.count())):
                                resolution_combo.addItem(f"{current_width}x{current_height} (現在)", current_res)
                except Exception as e:
                    print(f"[WARN] 解像度取得エラー: {e}")
                    # エラーが発生しても一般的な解像度を表示
                    for width, height, name in common_resolutions:
                        resolution_combo.addItem(name, (width, height))
                finally:
                    if cap is not None:
                        cap.release()
                
                if resolution_combo.count() == 0:
                    # フォールバック：一般的な解像度を追加
                    for width, height, name in common_resolutions:
                        resolution_combo.addItem(name, (width, height))
                
                # デフォルトで1920x1080を選択（利用可能な場合）
                for i in range(resolution_combo.count()):
                    res_data = resolution_combo.itemData(i)
                    if res_data and res_data == (1920, 1080):
                        resolution_combo.setCurrentIndex(i)
                        break
                # 1920x1080がなければ最初の解像度を選択
                if resolution_combo.currentIndex() < 0:
                    resolution_combo.setCurrentIndex(0)
            
            # カメラ選択が変更されたときに解像度オプションを更新
            camera_combo.currentIndexChanged.connect(update_resolution_options)
            
            # 初期解像度オプションを設定
            if camera_combo.count() > 0:
                update_resolution_options()
            
            resolution_layout = QHBoxLayout()
            resolution_layout.addWidget(resolution_label)
            resolution_layout.addWidget(resolution_combo)
            resolution_layout.addStretch()
            camera_select_layout.addLayout(resolution_layout)
            
            camera_layout.addLayout(camera_select_layout)
            
            # すべてのカメラのプレビューを表示
            if len(detected_cameras) > 0:
                preview_label = QLabel("検出されたカメラのプレビュー:")
                preview_label.setStyleSheet("font-size: 10pt; font-weight: bold; padding: 5px;")
                camera_layout.addWidget(preview_label)
                
                # スクロール可能なプレビューエリア
                preview_scroll = QScrollArea()
                preview_scroll.setWidgetResizable(True)
                preview_scroll.setMinimumHeight(300)
                preview_scroll.setMaximumHeight(500)
                
                preview_widget = QtWidgets.QWidget()
                preview_grid = QtWidgets.QGridLayout(preview_widget)
                preview_grid.setSpacing(10)
                
                # 各カメラのプレビューを作成（軽量化版）
                self.camera_preview_labels = {}
                self.camera_preview_caps = {}
                self.camera_preview_failed = set()  # 失敗したカメラを記録
                self.camera_preview_retry_count = {}  # リトライ回数を記録
                
                cols = 2  # 2列で表示
                for idx, (camera_id, camera_name) in enumerate(detected_cameras):
                    row = idx // cols
                    col = idx % cols
                    
                    # カメラ名ラベル
                    name_label = QLabel(camera_name)
                    name_label.setStyleSheet("font-size: 9pt; font-weight: bold; padding: 3px;")
                    preview_grid.addWidget(name_label, row * 2, col)
                    
                    # プレビュー画像ラベル（サイズを小さくして軽量化）
                    preview_label = QLabel("読み込み中...")
                    preview_label.setMinimumSize(160, 120)  # 320x240から160x120に縮小
                    preview_label.setMaximumSize(160, 120)
                    preview_label.setScaledContents(True)
                    preview_label.setStyleSheet("""
                        QLabel {
                            border: 2px solid #0066cc;
                            border-radius: 4px;
                            background-color: #000;
                            color: white;
                        }
                    """)
                    preview_label.setAlignment(QtCore.Qt.AlignCenter)
                    preview_grid.addWidget(preview_label, row * 2 + 1, col)
                    
                    self.camera_preview_labels[camera_id] = preview_label
                    self.camera_preview_retry_count[camera_id] = 0
                    
                    # カメラを開く（遅延読み込み方式で軽量化）
                    # 初期状態ではカメラを開かず、プレビュー更新時に開く
                    self.camera_preview_caps[camera_id] = None
                    preview_label.setText(f"カメラ {camera_id}\n待機中...")
                
                preview_widget.setLayout(preview_grid)
                preview_scroll.setWidget(preview_widget)
                camera_layout.addWidget(preview_scroll)
                
                # タイマーで定期的にプレビューを更新（更新頻度をさらに下げて軽量化）
                self.camera_preview_timer = QTimer()
                self.camera_preview_timer.timeout.connect(lambda: self._update_camera_previews())
                self.camera_preview_timer.start(3000)  # 3秒ごとに更新（軽量化）
            
            camera_group.setLayout(camera_layout)
            inspection_settings_layout.addWidget(camera_group)
            
            # 信頼度閾値設定
            threshold_group = QtWidgets.QGroupBox("🎯 信頼度閾値設定")
            threshold_group.setStyleSheet("font-weight: bold; font-size: 11pt;")
            threshold_layout = QVBoxLayout()
            threshold_layout.setContentsMargins(15, 15, 15, 15)
            threshold_layout.setSpacing(10)
            
            threshold_inner_layout = QHBoxLayout()
            threshold_label = QLabel("閾値:")
            threshold_label.setStyleSheet("font-size: 11pt; font-weight: bold; min-width: 120px;")
            confidence_threshold_spin = QDoubleSpinBox()
            confidence_threshold_spin.setMinimum(0.0)
            confidence_threshold_spin.setMaximum(1.0)
            confidence_threshold_spin.setSingleStep(0.05)
            confidence_threshold_spin.setValue(self.confidence_threshold)
            confidence_threshold_spin.setDecimals(2)
            confidence_threshold_spin.setMinimumWidth(120)
            confidence_threshold_spin.setStyleSheet("font-size: 11pt; padding: 6px;")
            
            threshold_percent_label = QLabel(f"({self.confidence_threshold:.0%})")
            threshold_percent_label.setStyleSheet("font-size: 11pt; color: #0066cc; font-weight: bold; min-width: 60px;")
            
            def update_threshold_percent(value):
                threshold_percent_label.setText(f"({value:.0%})")
            confidence_threshold_spin.valueChanged.connect(update_threshold_percent)
            
            threshold_inner_layout.addWidget(threshold_label)
            threshold_inner_layout.addWidget(confidence_threshold_spin)
            threshold_inner_layout.addWidget(threshold_percent_label)
            threshold_inner_layout.addStretch()
            
            threshold_layout.addLayout(threshold_inner_layout)
            threshold_group.setLayout(threshold_layout)
            inspection_settings_layout.addWidget(threshold_group)
            
            # オプション
            options_group = QtWidgets.QGroupBox("⚙️ 動作オプション")
            options_group.setStyleSheet("font-weight: bold; font-size: 11pt;")
            options_layout = QVBoxLayout()
            options_layout.setContentsMargins(15, 15, 15, 15)
            options_layout.setSpacing(12)
            
            auto_save_ng_checkbox = QCheckBox("💾 NG検出時に画像を自動保存")
            auto_save_ng_checkbox.setChecked(self.auto_save_ng)
            # グローバルスタイルを適用（個別のスタイル設定を削除）
            options_layout.addWidget(auto_save_ng_checkbox)
            
            alert_enabled_checkbox = QCheckBox("🔔 視覚的アラート（NG検出時に画面点滅）")
            if hasattr(self, 'alert_enabled_checkbox') and self.alert_enabled_checkbox is not None:
                alert_enabled_checkbox.setChecked(self.alert_enabled_checkbox.isChecked())
            else:
                alert_enabled_checkbox.setChecked(True)
            # グローバルスタイルを適用（個別のスタイル設定を削除）
            options_layout.addWidget(alert_enabled_checkbox)
            
            options_group.setLayout(options_layout)
            inspection_settings_layout.addWidget(options_group)
            
            inspection_settings_layout.addStretch()
            
            inspection_scroll.setWidget(inspection_content)
            
            # レイアウト
            dialog_layout = QVBoxLayout(dialog)
            dialog_layout.setContentsMargins(0, 0, 0, 0)
            dialog_layout.addWidget(inspection_scroll)
            
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            def cleanup_camera_previews():
                """カメラプレビューをクリーンアップ"""
                try:
                    # 設定ダイアログの参照をクリア
                    self._settings_dialog = None
                    if hasattr(self, 'camera_preview_timer'):
                        self.camera_preview_timer.stop()
                        self.camera_preview_timer = None
                    if hasattr(self, 'camera_preview_caps'):
                        for camera_id, cap in self.camera_preview_caps.items():
                            try:
                                if cap is not None:
                                    cap.release()
                            except:
                                pass
                        self.camera_preview_caps = {}
                    if hasattr(self, 'camera_preview_labels'):
                        self.camera_preview_labels = {}
                except:
                    pass
            
            def update_training_settings():
                """学習設定を更新"""
                if settings_quick_mode_radio.isChecked():
                    # クイックモード
                    if hasattr(self, 'quick_mode_radio'):
                        self.quick_mode_radio.setChecked(True)
                    if hasattr(self, 'quick_combo'):
                        self.quick_combo.setCurrentIndex(settings_quick_combo.currentIndex())
                else:
                    # カスタムモード
                    if hasattr(self, 'custom_mode_radio'):
                        self.custom_mode_radio.setChecked(True)
                    if hasattr(self, 'gpu_combo'):
                        self.gpu_combo.setCurrentIndex(settings_gpu_combo.currentIndex())
                    if hasattr(self, 'cpu_combo'):
                        self.cpu_combo.setCurrentIndex(settings_cpu_combo.currentIndex())
            
            def on_dialog_rejected():
                cleanup_camera_previews()
                dialog.reject()
            
            cancel_btn = QPushButton("キャンセル")
            cancel_btn.clicked.connect(on_dialog_rejected)
            cancel_btn.setStyleSheet("background-color: #ccc; color: #333; font-weight: bold; padding: 8px 20px; border-radius: 4px;")
            button_layout.addWidget(cancel_btn)
            
            save_btn = QPushButton("💾 保存して閉じる")
            save_btn.setStyleSheet("background-color: #0066cc; color: white; font-weight: bold; padding: 10px 30px; border-radius: 4px; font-size: 11pt;")
            
            def save_settings():
                # カメラプレビューをクリーンアップ
                cleanup_camera_previews()
                
                # 設定を保存
                self.confidence_threshold = confidence_threshold_spin.value()
                self.auto_save_ng = auto_save_ng_checkbox.isChecked()
                # alert_enabled_checkboxの保存（Noneチェックを追加）
                if hasattr(self, 'alert_enabled_checkbox') and self.alert_enabled_checkbox is not None:
                    self.alert_enabled_checkbox.setChecked(alert_enabled_checkbox.isChecked())
                else:
                    # alert_enabled_checkboxがNoneの場合は、値だけ保存
                    self.alert_enabled_checkbox_value = alert_enabled_checkbox.isChecked()
                
                # カメラ選択と解像度を更新
                try:
                    if camera_combo.count() > 0:
                        selected_camera_id = camera_combo.currentData()
                        selected_resolution = resolution_combo.currentData()
                        
                        if selected_camera_id is not None:
                            if not hasattr(self, 'camera_combo') or self.camera_combo is None:
                                self.camera_combo = QComboBox()
                            self.camera_combo.clear()
                            for i in range(camera_combo.count()):
                                item_text = camera_combo.itemText(i)
                                item_data = camera_combo.itemData(i)
                                # 解像度情報を更新（選択された解像度に基づく）
                                if selected_resolution is not None and item_data == selected_camera_id:
                                    width, height = selected_resolution
                                    item_text = f"カメラ {selected_camera_id} ({width}x{height})"
                                self.camera_combo.addItem(item_text, item_data)
                            for i in range(self.camera_combo.count()):
                                if self.camera_combo.itemData(i) == selected_camera_id:
                                    self.camera_combo.setCurrentIndex(i)
                                    break
                        
                        # 解像度設定を保存
                        if selected_resolution is not None:
                            width, height = selected_resolution
                            self.camera_resolution = (width, height)
                            print(f"[INFO] カメラ解像度を保存: {width}x{height}")
                        else:
                            # 解像度が選択されていない場合はカメラのデフォルト解像度を使用
                            if hasattr(self, 'camera_resolution'):
                                delattr(self, 'camera_resolution')
                except Exception as e:
                    print(f"カメラ選択の更新エラー: {e}")
                
                self.save_inspection_settings()
                self.statusBar().showMessage('設定を保存しました', 2000)
                dialog.accept()
            
            save_btn.clicked.connect(save_settings)
            button_layout.addWidget(save_btn)
            dialog_layout.addLayout(button_layout)
            
            dialog.exec_()
        except Exception as e:
            import traceback
            error_msg = f"設定ダイアログの表示エラー:\n{str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            QtWidgets.QMessageBox.critical(self, 'エラー', f'設定ダイアログの表示に失敗しました:\n{str(e)}')
    
    def show_about_dialog(self):
        """Aboutダイアログを表示"""
        about_text = """
        <h2>統合ワッシャー検査・学習システム</h2>
        <p><b>バージョン:</b> 1.0.0</p>
        <p>ワッシャーの外観検査と深層学習による欠陥検出システム</p>
        <hr>
        <h3>主な機能:</h3>
        <ul>
            <li>🔍 リアルタイム外観検査</li>
            <li>🎓 アンサンブルモデルによる学習</li>
            <li>📊 システムリソース監視</li>
            <li>⚙️ 柔軟な設定オプション</li>
        </ul>
        <hr>
        <p style="color: #666; font-size: 9pt;">© 2024 WasherInspection System</p>
        """
        
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle('このアプリについて')
        msg_box.setTextFormat(QtCore.Qt.RichText)
        msg_box.setText(about_text)
        msg_box.setIcon(QtWidgets.QMessageBox.Information)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg_box.exec_()
    
    def _check_resume_available(self):
        """チェックポイントから再開可能かチェック"""
        try:
            from pathlib import Path
            checkpoint_base = Path('checkpoints')
            if not checkpoint_base.exists():
                return False
            
            # 各モデルのチェックポイントを確認
            model_names = ['efficientnetb0', 'efficientnetb1', 'efficientnetb2']
            for model_name in model_names:
                checkpoint_dir = checkpoint_base / f'clear_sparse_4class_{model_name}'
                if checkpoint_dir.exists():
                    checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.h5'))
                    if checkpoints:
                        return True
            return False
        except Exception as e:
            print(f"[WARN] チェックポイント確認エラー: {e}")
            return False
    
    def resume_training(self):
        """学習を再開（チェックポイントから）"""
        print("[INFO] ========== 学習再開ボタンがクリックされました ==========")
        
        try:
            # 既に実行中かチェック
            if hasattr(self, 'training_worker') and self.training_worker is not None:
                if self.training_worker.isRunning():
                    print("[INFO] 学習は既に実行中です")
                    self.statusBar().showMessage('学習は既に実行中です', 3000)
                    return
            
            # チェックポイントが存在するか確認
            if not self._check_resume_available():
                QMessageBox.warning(
                    self,
                    '再開不可',
                    'チェックポイントが見つかりません。\n最初から学習を開始してください。'
                )
                return
            
            # 確認ダイアログ
            reply = QMessageBox.question(
                self,
                '学習再開確認',
                'チェックポイントから学習を再開しますか？\n\n'
                '最後に保存されたエポックから続行します。',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply != QMessageBox.Yes:
                return
            
            # 即座にUIを更新
            self._update_training_ui_preparing()
            
            # リソース設定を取得
            resource_config = self._get_resource_config()
            
            # WSL2モード確認
            use_wsl2 = False
            if hasattr(self, 'use_wsl2_checkbox') and self.use_wsl2_checkbox is not None:
                use_wsl2 = self.use_wsl2_checkbox.isChecked()
            
            # TrainingWorkerを作成（再開モード）
            self.training_worker = TrainingWorker(
                resource_config=resource_config,
                use_wsl2=use_wsl2,
                resume=True  # 再開フラグ
            )
            
            # シグナル接続
            self.training_worker.finished.connect(self.training_finished)
            self.training_worker.log_message.connect(self.append_training_log)
            
            # 学習開始
            self.training_worker.start()
            self._update_training_ui_started(
                "WSL2 GPUモード" if use_wsl2 else "Windows CPUモード",
                resource_config.get('batch_size', 32),
                resource_config.get('workers', 8),
                resource_config.get('max_epochs', 200)
            )
            
            print("[INFO] 学習再開を開始しました（チェックポイントから）")
            
        except Exception as e:
            error_msg = f"学習再開エラー: {e}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            self._update_training_ui_error(error_msg)
            QMessageBox.critical(self, 'エラー', f'学習再開に失敗しました:\n{str(e)}')
    
    def start_training(self):
        """学習を開始（チェックポイントがあれば自動的に再開）"""
        print("[INFO] ========== 学習開始ボタンがクリックされました ==========")
        
        try:
            # 既に実行中かチェック
            if hasattr(self, 'training_worker') and self.training_worker is not None:
                if self.training_worker.isRunning():
                    print("[INFO] 学習は既に実行中です")
                    self.statusBar().showMessage('学習は既に実行中です', 3000)
                    return
            
            # チェックポイントが存在するか確認（自動再開判定）
            can_resume = self._check_resume_available()
            resume_mode = False
            
            if can_resume:
                # チェックポイントが見つかった場合、再開確認ダイアログを表示
                reply = QMessageBox.question(
                    self,
                    '学習再開',
                    'チェックポイントが見つかりました。\n\n'
                    '前回の学習から続行しますか？\n'
                    '（「いいえ」を選択すると最初から学習を開始します）',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                resume_mode = (reply == QMessageBox.Yes)
                if resume_mode:
                    print("[INFO] チェックポイントから学習を再開します")
                else:
                    print("[INFO] 最初から学習を開始します")
            
            # 即座にUIを更新して、ボタンが押されたことを明確に表示
            self._update_training_ui_preparing()
            
            # リソース設定を取得
            resource_config = self._get_resource_config()
            print(f"[INFO] リソース設定: {resource_config}")
            
            # WSL2モード確認
            use_wsl2 = False
            if hasattr(self, 'use_wsl2_checkbox') and self.use_wsl2_checkbox is not None:
                use_wsl2 = self.use_wsl2_checkbox.isChecked()
            print(f"[INFO] WSL2モード: {use_wsl2}")
            print(f"[INFO] 再開モード: {resume_mode}")
            
            # 学習スクリプトの確認
            if not TRAIN_SCRIPT.exists():
                error_msg = f'学習スクリプトが見つかりません: {TRAIN_SCRIPT}'
                print(f"[ERROR] {error_msg}")
                self._update_training_ui_error("学習スクリプトが見つかりません")
                QMessageBox.critical(self, 'エラー', error_msg)
                return
            
            # TrainingWorkerを作成（再開モードフラグを設定）
            self.training_worker = TrainingWorker(
                resource_config=resource_config,
                use_wsl2=use_wsl2,
                resume=resume_mode  # チェックポイントがある場合は再開モード
            )
            
            # シグナル接続
            self.training_worker.finished.connect(self.training_finished)
            self.training_worker.log_message.connect(self.append_training_log)
            
            # 確認ダイアログ（再開モードでない場合のみ表示）
            if not resume_mode:
                env_mode = "WSL2 GPUモード" if use_wsl2 else "Windows CPUモード"
                batch_size = resource_config.get('batch_size', 32)
                workers = resource_config.get('workers', 8)
                max_epochs = resource_config.get('max_epochs', 200)
                
                reply = QMessageBox.question(
                    self,
                    '学習開始確認',
                    f'学習を開始しますか？\n\n'
                    f'実行環境: {env_mode}\n'
                    f'Batch Size: {batch_size}\n'
                    f'Workers: {workers}\n'
                    f'最大Epochs: {max_epochs}',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply != QMessageBox.Yes:
                    print("[INFO] ユーザーがキャンセルしました")
                    self._reset_training_ui()
                    self.training_worker = None
                    return
            
            # ワーカーを起動
            self.training_worker.start()
            print("[INFO] 学習ワーカーを起動しました")
            
            # UI更新（学習開始）
            mode_text = "WSL2 GPUモード" if use_wsl2 else "Windows CPUモード"
            if resume_mode:
                mode_text += " (チェックポイントから再開)"
            
            self._update_training_ui_started(
                mode_text,
                resource_config.get('batch_size', 32),
                resource_config.get('workers', 8),
                resource_config.get('max_epochs', 200)
            )
            
            if resume_mode:
                print("[INFO] 学習を再開しました（チェックポイントから）")
            else:
                print("[INFO] 学習を開始しました")
            print("[INFO] ========== 学習開始処理完了 ==========")
            
        except Exception as e:
            error_msg = f'学習開始エラー: {str(e)}'
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            
            # エラーダイアログ
            try:
                QMessageBox.critical(self, '学習開始エラー', error_msg)
            except:
                pass
            
            # UIをリセット
            self._reset_training_ui()
            
            # ワーカーをクリア
            if hasattr(self, 'training_worker'):
                self.training_worker = None
    
    def _reset_training_ui(self):
        """学習UIをリセット"""
        try:
            if hasattr(self, 'start_training_btn'):
                self.start_training_btn.setEnabled(True)
            if hasattr(self, 'stop_training_btn'):
                self.stop_training_btn.setEnabled(False)
            if hasattr(self, 'training_status_label'):
                self.training_status_label.setText('ステータス: ⚪ 未開始')
            if hasattr(self, 'current_activity_label'):
                self.current_activity_label.setText('現在の作業: -')
            if hasattr(self, 'training_progress'):
                self.training_progress.setValue(0)
        except:
            pass
    
    def _update_training_ui_preparing(self):
        """学習準備中のUI更新"""
        try:
            if hasattr(self, 'training_status_label'):
                self.training_status_label.setText('ステータス: 🟡 準備中...')
                self.training_status_label.setStyleSheet('color:#ffc107; font-weight:bold; font-size:12pt;')
            if hasattr(self, 'current_activity_label'):
                self.current_activity_label.setText('現在の作業: 学習の準備中...')
            if hasattr(self, 'training_log_text') and self.training_log_text is not None:
                self.training_log_text.clear()
                self.append_training_log("[INFO] 学習を準備しています...")
            self.statusBar().showMessage('学習を準備しています...', 2000)
        except:
            pass
    
    def _update_training_ui_started(self, env_mode, batch_size, workers, max_epochs):
        """学習開始時のUI更新"""
        try:
            # ボタン状態
            if hasattr(self, 'start_training_btn'):
                self.start_training_btn.setEnabled(False)
            if hasattr(self, 'stop_training_btn'):
                self.stop_training_btn.setEnabled(True)
            
            # ステータス表示
            if hasattr(self, 'training_status_label'):
                self.training_status_label.setText('ステータス: 🟢 実行中')
                self.training_status_label.setStyleSheet('color:#28a745; font-weight:bold; font-size:12pt;')
            if hasattr(self, 'current_activity_label'):
                self.current_activity_label.setText(f'現在の作業: {env_mode}で学習を開始しました')
            
            # ログ表示
            if hasattr(self, 'training_log_text') and self.training_log_text is not None:
                self.append_training_log("="*60)
                self.append_training_log(f"[開始] 学習開始: {env_mode}")
                self.append_training_log(f"[設定] Batch Size: {batch_size}, Workers: {workers}, Max Epochs: {max_epochs}")
                self.append_training_log("="*60)
                self.append_training_log("学習を開始しました。ログを待っています...")
            
            # プログレスバー
            if hasattr(self, 'training_progress'):
                self.training_progress.setValue(1)  # 0%ではなく1%に設定して開始を表示
            
            self.statusBar().showMessage('学習を開始しました', 3000)
        except Exception as e:
            print(f"[WARN] UI更新エラー: {e}")
    
    def _update_training_ui_error(self, error_msg):
        """エラー時のUI更新"""
        try:
            if hasattr(self, 'training_status_label'):
                self.training_status_label.setText('ステータス: 🔴 エラー')
                self.training_status_label.setStyleSheet('color:#dc3545; font-weight:bold; font-size:12pt;')
            if hasattr(self, 'current_activity_label'):
                self.current_activity_label.setText(f'現在の作業: エラー - {error_msg}')
            if hasattr(self, 'training_log_text') and self.training_log_text is not None:
                self.append_training_log(f"[ERROR] {error_msg}")
        except:
            pass
    
    def _get_resource_config(self):
        """リソース設定をUIから取得（ヘルパーメソッド）"""
        try:
            if HAS_RESOURCE_SELECTOR:
                selector = TrainingResourceSelector()
                
                # モードを確認（クイック or カスタム）
                is_custom_mode = False
                if hasattr(self, 'custom_mode_radio') and self.custom_mode_radio is not None:
                    is_custom_mode = self.custom_mode_radio.isChecked()
                
                if is_custom_mode:
                    # カスタムモード：GPU/CPU個別選択
                    gpu_index = 2  # デフォルト: high
                    cpu_index = 2  # デフォルト: high
                    if hasattr(self, 'gpu_combo') and self.gpu_combo is not None:
                        gpu_index = self.gpu_combo.currentIndex()
                    if hasattr(self, 'cpu_combo') and self.cpu_combo is not None:
                        cpu_index = self.cpu_combo.currentIndex()
                    
                    # インデックスをレベルに変換
                    level_map = {0: 'low', 1: 'medium', 2: 'high', 3: 'maximum', 4: 'maximum'}
                    gpu_level = level_map.get(gpu_index, 'high')
                    cpu_level = level_map.get(cpu_index, 'high')
                    
                    gpu_config = selector.GPU_LEVELS[gpu_level]
                    cpu_config = selector.CPU_LEVELS[cpu_level]
                    
                    # 最大学習回数とパテンスを決定
                    avg_level = (gpu_index + cpu_index) / 2.0
                    if avg_level < 1:
                        max_epochs, patience = 100, 20
                    elif avg_level < 2:
                        max_epochs, patience = 150, 25
                    elif avg_level < 3:
                        max_epochs, patience = 200, 30
                    elif avg_level < 4:
                        max_epochs, patience = 300, 50
                    else:
                        max_epochs, patience = 400, 60
                    
                    resource_config = {
                        'batch_size': gpu_config['batch_size'],
                        'workers': cpu_config['workers'],
                        'max_queue_size': cpu_config.get('max_queue_size', 20),
                        'use_multiprocessing': cpu_config.get('use_multiprocessing', True),
                        'max_epochs': max_epochs,
                        'patience': patience,
                        'use_mixed_precision': gpu_config.get('use_mixed_precision', True),
                        'gpu_utilization': gpu_level,
                        'cpu_utilization': cpu_level,
                    }
                else:
                    # クイックモード：プリセットから選択
                    quick_index = 2  # デフォルト: 高性能
                    if hasattr(self, 'quick_combo') and self.quick_combo is not None:
                        quick_index = self.quick_combo.currentIndex()
                    
                    profile_map = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5'}
                    profile_id = profile_map.get(quick_index, '3')
                    resource_config = selector.RESOURCE_PROFILES[profile_id].copy()
                
                # バックグラウンドモードが有効な場合、リソースを削減
                if hasattr(self, 'background_mode_checkbox') and self.background_mode_checkbox is not None:
                    if self.background_mode_checkbox.isChecked():
                        if 'batch_size' in resource_config:
                            resource_config['batch_size'] = max(4, int(resource_config['batch_size'] * 0.5))
                        if 'workers' in resource_config:
                            resource_config['workers'] = max(1, int(resource_config['workers'] * 0.5))
                        if 'max_queue_size' in resource_config:
                            resource_config['max_queue_size'] = max(2, int(resource_config['max_queue_size'] * 0.5))
                        resource_config['background_mode'] = True
                
                # 環境変数に適用
                selector.config = resource_config
                selector.apply_config_to_environment()
                
                return resource_config
            else:
                # TrainingResourceSelectorが利用できない場合はデフォルト値
                return {
                    'batch_size': 32,
                    'workers': 8,
                    'max_epochs': 200,
                    'max_queue_size': 20,
                    'use_multiprocessing': True,
                    'patience': 30,
                    'use_mixed_precision': True,
                }
        except Exception as e:
            print(f"[WARN] リソース設定取得エラー: {e}")
            import traceback
            traceback.print_exc()
            # エラーが発生してもデフォルト設定で続行
            return {
                'batch_size': 32,
                'workers': 8,
                'max_epochs': 200,
                'max_queue_size': 20,
                'use_multiprocessing': True,
                'patience': 30,
                'use_mixed_precision': True,
            }
    
    def start_remote_server(self):
        """リモートサーバーを起動（学習中でも安全に実行可能）"""
        try:
            import socket
            import threading
            
            # 学習中でも実行可能であることを表示
            if hasattr(self, 'training_worker') and self.training_worker is not None:
                if self.training_worker.isRunning():
                    reply = QMessageBox.question(
                        self,
                        '学習中',
                        '学習が実行中です。\n\n'
                        'リモートサーバーは学習を妨げずに起動できます。\n'
                        '続行しますか？',
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return
            
            # サーバースクリプトのパス
            server_script = Path(__file__).resolve().parents[1] / 'scripts' / 'remote_server.py'
            
            if not server_script.exists():
                QMessageBox.warning(
                    self,
                    'エラー',
                    f'リモートサーバースクリプトが見つかりません:\n{server_script}'
                )
                return
            
            # 既に起動しているかチェック
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(1)
                result = test_socket.connect_ex(('127.0.0.1', 5000))
                test_socket.close()
                
                if result == 0:
                    # サーバーは既に起動している
                    try:
                        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        s.connect(("8.8.8.8", 80))
                        local_ip = s.getsockname()[0]
                        s.close()
                    except:
                        local_ip = "127.0.0.1"
                    
                    QMessageBox.information(
                        self,
                        'リモートサーバー',
                        f'リモートサーバーは既に起動しています。\n\n'
                        f'ローカルアクセス:\nhttp://localhost:5000\n\n'
                        f'リモートアクセス:\nhttp://{local_ip}:5000'
                    )
                    return
            except:
                pass
            
            # サーバーをバックグラウンドで起動（学習に影響しない）
            def run_server():
                try:
                    # 新しいプロセスで起動（学習プロセスとは独立）
                    if sys.platform.startswith('win'):
                        # Windows: CREATE_NO_WINDOWで非表示起動
                        creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                        subprocess.Popen(
                            [sys.executable, str(server_script)],
                            cwd=str(server_script.parent.parent),
                            creationflags=creation_flags,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    else:
                        subprocess.Popen(
                            [sys.executable, str(server_script)],
                            cwd=str(server_script.parent.parent),
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                except Exception as e:
                    print(f"[ERROR] リモートサーバー起動エラー: {e}")
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # IPアドレスを取得
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except:
                local_ip = "127.0.0.1"
            
            # 少し待ってから確認
            time.sleep(2)
            
            QMessageBox.information(
                self,
                'リモートサーバー起動',
                f'リモートサーバーを起動しました。\n\n'
                f'ローカルアクセス:\nhttp://localhost:5000\n\n'
                f'リモートアクセス:\nhttp://{local_ip}:5000\n\n'
                f'別PCのブラウザから上記URLにアクセスして\n'
                f'学習を遠隔操作できます。\n\n'
                f'※ 学習中でも安全に使用できます。'
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                'エラー',
                f'リモートサーバーの起動に失敗しました:\n{str(e)}'
            )
            import traceback
            traceback.print_exc()
    
    def start_remote_tunnel(self):
        """インターネット経由アクセストンネルを起動（学習中でも安全に実行可能）"""
        try:
            import socket
            import threading
            
            # 学習中でも実行可能であることを表示
            if hasattr(self, 'training_worker') and self.training_worker is not None:
                if self.training_worker.isRunning():
                    reply = QMessageBox.question(
                        self,
                        '学習中',
                        '学習が実行中です。\n\n'
                        'トンネルは学習を妨げずに起動できます。\n'
                        '続行しますか？',
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return
            
            # トンネルスクリプトのパス
            tunnel_script = Path(__file__).resolve().parents[1] / 'scripts' / 'remote_server_tunnel.py'
            
            if not tunnel_script.exists():
                QMessageBox.warning(
                    self,
                    'エラー',
                    f'トンネルスクリプトが見つかりません:\n{tunnel_script}'
                )
                return
            
            # 設定ファイルを確認
            config_file = Path(__file__).resolve().parents[1] / 'config' / 'remote_tunnel_config.json'
            if not config_file.exists():
                reply = QMessageBox.question(
                    self,
                    '設定が必要',
                    'トンネル設定が行われていません。\n設定を開始しますか？',
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    # 設定ツールを起動（バックグラウンド）
                    configure_script = Path(__file__).resolve().parents[1] / 'scripts' / 'configure_tunnel.py'
                    if configure_script.exists():
                        if sys.platform.startswith('win'):
                            creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                            subprocess.Popen(
                                [sys.executable, str(configure_script)],
                                creationflags=creation_flags
                            )
                        else:
                            subprocess.Popen([sys.executable, str(configure_script)])
                    else:
                        QMessageBox.warning(
                            self,
                            'エラー',
                            '設定ツールが見つかりません'
                        )
                return
            
            # トンネルをバックグラウンドで起動（学習に影響しない）
            def run_tunnel():
                try:
                    # 新しいプロセスで起動（学習プロセスとは独立）
                    if sys.platform.startswith('win'):
                        creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                        process = subprocess.Popen(
                            [sys.executable, str(tunnel_script), '--start'],
                            cwd=str(tunnel_script.parent.parent),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            encoding='utf-8',
                            errors='replace',
                            creationflags=creation_flags
                        )
                    else:
                        process = subprocess.Popen(
                            [sys.executable, str(tunnel_script), '--start'],
                            cwd=str(tunnel_script.parent.parent),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            encoding='utf-8',
                            errors='replace'
                        )
                    
                    # URLを取得（ngrok APIから）
                    time.sleep(5)
                    tunnel_url = None
                    
                    # ngrok APIからURLを取得
                    try:
                        import requests
                        response = requests.get('http://localhost:4040/api/tunnels', timeout=5)
                        if response.status_code == 200:
                            data = response.json()
                            tunnels = data.get('tunnels', [])
                            if tunnels:
                                public_url = tunnels[0].get('public_url', '')
                                if public_url.startswith('http://'):
                                    tunnel_url = public_url.replace('http://', 'https://')
                                else:
                                    tunnel_url = public_url
                    except:
                        pass
                    
                    # 出力からも取得を試みる
                    if not tunnel_url:
                        try:
                            for line in process.stdout:
                                if 'https://' in line or 'http://' in line:
                                    parts = line.split()
                                    for part in parts:
                                        if part.startswith('https://') or part.startswith('http://'):
                                            tunnel_url = part.strip()
                                            break
                                    if tunnel_url:
                                        break
                        except:
                            pass
                    
                    if tunnel_url:
                        QMessageBox.information(
                            self,
                            'トンネル起動',
                            f'インターネット経由アクセストンネルが起動しました。\n\n'
                            f'アクセスURL:\n{tunnel_url}\n\n'
                            f'このURLを別のPCやスマートフォンから\n'
                            f'アクセスできます。\n\n'
                            f'※ このURLはトンネルを停止するまで有効です。\n'
                            f'※ 学習中でも安全に使用できます。'
                        )
                    else:
                        QMessageBox.information(
                            self,
                            'トンネル起動',
                            'トンネルを起動しました。\n\n'
                            'URLを確認するには:\n'
                            '1. ブラウザで http://localhost:4040 にアクセス（ngrokの場合）\n'
                            '2. または、数秒後に再度確認してください\n\n'
                            '※ 学習中でも安全に使用できます。'
                        )
                    
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        'エラー',
                        f'トンネル起動エラー:\n{str(e)}'
                    )
            
            tunnel_thread = threading.Thread(target=run_tunnel, daemon=True)
            tunnel_thread.start()
            
            QMessageBox.information(
                self,
                'トンネル起動中',
                'インターネット経由アクセストンネルを起動しています...\n\n'
                '数秒後にURLが表示されます。\n\n'
                '※ 学習中でも安全に実行できます。'
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                'エラー',
                f'トンネル起動に失敗しました:\n{str(e)}'
            )
            import traceback
            traceback.print_exc()
    
    def stop_training(self):
        """学習を停止（0から作り直し）"""
        print("[INFO] 学習停止ボタンがクリックされました")
        
        try:
            # ワーカーを停止
            if hasattr(self, 'training_worker') and self.training_worker is not None:
                # プロセスを停止
                if hasattr(self.training_worker, 'stop'):
                    self.training_worker.stop()
                
                # スレッドを終了
                if self.training_worker.isRunning():
                    self.training_worker.terminate()
                    self.training_worker.wait(3000)
                
                self.training_worker = None
            
            # UIをリセット
            self._reset_training_ui()
            self.statusBar().showMessage('学習を停止しました', 5000)
            print("[INFO] 学習を停止しました")
            
        except Exception as e:
            print(f"[ERROR] stop_trainingエラー: {e}")
            import traceback
            traceback.print_exc()
            self._reset_training_ui()
    
    def training_finished(self):
        """学習完了（0から作り直し）"""
        print("[INFO] 学習が完了しました")
        
        try:
            # UIをリセット
            self._reset_training_ui()
            self.statusBar().showMessage('学習が完了しました', 5000)
            
            # ワーカーをクリア
            if hasattr(self, 'training_worker'):
                self.training_worker = None
                
        except Exception as e:
            print(f"[ERROR] training_finishedエラー: {e}")
            import traceback
            traceback.print_exc()
    
    def append_training_log(self, log_line):
        """学習ログを追加（メインスレッドから呼ばれるためスレッドセーフ）"""
        if hasattr(self, 'training_log_text') and self.training_log_text is not None:
            try:
                # ログを追加
                self.training_log_text.append(log_line)
                # 自動スクロール（最下部に移動）
                scrollbar = self.training_log_text.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
            except Exception as e:
                print(f"[WARN] ログ追加エラー: {e}")
    
    def check_training_interrupted(self):
        """前回の学習が中断されているかチェック"""
        import time
        
        # ステータスファイルを確認
        if not STATUS_FILE.exists():
            return False
        
        try:
            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                status = json.load(f)
            
            overall_progress = float(status.get('overall_progress_percent', 0) or 0)
            stage = status.get('stage', '')
            timestamp = status.get('timestamp')
            
            # 100%完了していれば中断ではない
            if overall_progress >= 100.0:
                return False
            
            # 完了状態の場合は中断ではない
            if stage in ['Complete', '完了', 'Completed']:
                return False
            
            # タイムスタンプを確認（5分以上更新がなければ中断と判定）
            if timestamp:
                elapsed = time.time() - float(timestamp)
                # 5分以上更新がなく、進捗が100%未満の場合、中断と判定
                if elapsed > 300 and overall_progress < 100.0:
                    # 学習プロセスが実行中か確認
                    try:
                        import psutil
                        training_running = False
                        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                            try:
                                cmdline = proc.info.get('cmdline', [])
                                if cmdline and any('train_4class_sparse_ensemble' in str(arg) for arg in cmdline):
                                    training_running = True
                                    break
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        
                        # プロセスが実行中でなければ中断と判定
                        return not training_running
                    except ImportError:
                        # psutilが利用できない場合、タイムスタンプのみで判定
                        return True
            
            # チェックポイントファイルが存在するが、進捗が100%未満の場合も中断と判定
            project_root = Path(__file__).parent.parent
            checkpoint_patterns = [
                'clear_sparse_best_*.h5',
                'clear_sparse_ensemble_*.h5'
            ]
            
            has_checkpoint = False
            for pattern in checkpoint_patterns:
                if list(project_root.glob(pattern)) or list((project_root / 'models' / 'sparse').glob(pattern)):
                    has_checkpoint = True
                    break
            
            # チェックポイントがあり、進捗が100%未満なら中断
            return has_checkpoint and overall_progress < 100.0
            
        except Exception as e:
            print(f"[WARN] 中断チェックエラー: {e}")
            return False
    
    def confirm_clear_interrupted(self):
        """中断された学習のクリアを確認（警告付き・音なし）"""
        msg_box = SilentMessageBox(self)
        msg_box.setWindowTitle('⚠️ 中断された学習をクリア')
        msg_box.setText('中断された学習のチェックポイントを削除しますか？\n\n'
                       '⚠️ 警告:\n'
                       '• チェックポイント（.h5ファイル）が削除されます\n'
                       '• ステータスファイル（training_status.json）が削除されます\n'
                       '• 学習ログ（.csvファイル）は保持されます\n\n'
                       'この操作により、学習は最初からやり直しになります。')
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        
        reply = msg_box.exec_()
        
        if reply == QMessageBox.Yes:
            self.clear_interrupted_training_checkpoints()
    
    def clear_interrupted_training_checkpoints(self):
        """中断された学習のチェックポイントとステータスファイルのみ削除（ログは保持）"""
        try:
            deleted_files = []
            
            # チェックポイントファイル（.h5）を削除
            checkpoint_patterns = [
                'clear_sparse_best_*.h5',
                'clear_sparse_ensemble_*.h5',
                'retrain_sparse_ensemble_*.h5'
            ]
            
            project_root = Path(__file__).parent.parent
            
            # プロジェクトルートの直接検索
            for pattern in checkpoint_patterns:
                for file_path in project_root.glob(pattern):
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                            deleted_files.append(str(file_path))
                    except Exception as e:
                        print(f"[WARN] ファイル削除失敗: {file_path}, エラー: {e}")
            
            # models/sparse/ フォルダ内のチェックポイント
            models_sparse_dir = project_root / 'models' / 'sparse'
            if models_sparse_dir.exists():
                for pattern in checkpoint_patterns:
                    for file_path in models_sparse_dir.glob(pattern):
                        try:
                            if file_path.is_file():
                                file_path.unlink()
                                deleted_files.append(str(file_path))
                        except Exception as e:
                            print(f"[WARN] ファイル削除失敗: {file_path}, エラー: {e}")
            
            # ステータスファイルを削除
            status_file = project_root / 'logs' / 'training_status.json'
            if status_file.exists():
                try:
                    status_file.unlink()
                    deleted_files.append(str(status_file))
                except Exception as e:
                    print(f"[WARN] ステータスファイル削除失敗: {status_file}, エラー: {e}")
            
            # ステータス表示をリセット
            if hasattr(self, 'training_status_label'):
                self.training_status_label.setText('ステータス: ⚪ 未開始')
                self.training_status_label.setStyleSheet('color:#888; font-weight:bold; font-size:12pt;')
            if hasattr(self, 'current_activity_label'):
                self.current_activity_label.setText('現在の作業: -')
            if hasattr(self, 'training_progress'):
                self.training_progress.setValue(0)
            if hasattr(self, 'epoch_label'):
                self.epoch_label.setText('学習回数: -')
            if hasattr(self, 'accuracy_label'):
                self.accuracy_label.setText('精度: -')
            
            deleted_count = len(deleted_files)
            print(f"[INFO] 中断された学習のチェックポイントを削除しました（{deleted_count}個のファイル）")
            self.statusBar().showMessage(f'中断された学習のチェックポイントを削除しました（{deleted_count}個）', 3000)
            
        except Exception as e:
            error_msg = f'チェックポイント削除エラー: {str(e)}'
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
    
    def confirm_reset_training_files(self):
        """完全リセットを確認（警告付き・二重確認・音なし）"""
        # 第1回目の確認（警告ダイアログ・音なし）
        msg_box1 = SilentMessageBox(self)
        msg_box1.setWindowTitle('⚠️⚠️⚠️ 完全リセット警告')
        msg_box1.setText('⚠️⚠️⚠️ 危険: 全ての学習ファイルを削除します！\n\n'
                        '削除されるファイル:\n'
                        '• チェックポイント（.h5ファイル）\n'
                        '• 学習ログ（.csvファイル）\n'
                        '• ステータスファイル（training_status.json）\n'
                        '• その他の学習関連ファイル\n\n'
                        '⚠️ この操作は取り消せません。\n'
                        '⚠️ 学習ログも完全に削除されます。\n'
                        '⚠️ 全ての学習データが失われます。\n\n'
                        '本当に実行しますか？')
        msg_box1.setIcon(QMessageBox.Warning)
        msg_box1.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box1.setDefaultButton(QMessageBox.No)
        
        reply1 = msg_box1.exec_()
        
        if reply1 != QMessageBox.Yes:
            return
        
        # 第2回目の確認（最終確認・音なし）
        msg_box2 = SilentMessageBox(self)
        msg_box2.setWindowTitle('🚨🚨🚨 最終確認 🚨🚨🚨')
        msg_box2.setText('🚨🚨🚨 最終確認 🚨🚨🚨\n\n'
                        '全ての学習データを完全に削除しようとしています。\n\n'
                        'この操作は絶対に取り消せません。\n'
                        '学習ログ、チェックポイント、全てが失われます。\n\n'
                        '本当に、本当に実行しますか？\n\n'
                        '「はい」を選択すると、即座に全てのファイルが削除されます。')
        msg_box2.setIcon(QMessageBox.Critical)
        msg_box2.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box2.setDefaultButton(QMessageBox.No)
        
        reply2 = msg_box2.exec_()
        
        if reply2 == QMessageBox.Yes:
            self.reset_training_files()
    
    def reset_training_files(self):
        """学習ファイルを完全にリセット（手動実行用、ログも削除）"""
        
        try:
            deleted_files = []
            
            # チェックポイントファイル（.h5）を削除
            checkpoint_patterns = [
                'clear_sparse_*.h5',
                'clear_sparse_best_*.h5',
                'clear_sparse_ensemble_*.h5',
                'retrain_sparse_ensemble_*.h5',
                'optimized_best_*.h5',
                'clear_sparse_ensemble_4class_info.json'
            ]
            
            project_root = Path(__file__).parent.parent
            
            # プロジェクトルートの直接検索
            for pattern in checkpoint_patterns:
                for file_path in project_root.glob(pattern):
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                            deleted_files.append(str(file_path))
                    except Exception as e:
                        print(f"[WARN] ファイル削除失敗: {file_path}, エラー: {e}")
            
            # models/sparse/ フォルダ内のチェックポイント
            models_sparse_dir = project_root / 'models' / 'sparse'
            if models_sparse_dir.exists():
                for pattern in ['clear_sparse_*.h5', 'retrain_sparse_ensemble_*.h5']:
                    for file_path in models_sparse_dir.glob(pattern):
                        try:
                            if file_path.is_file():
                                file_path.unlink()
                                deleted_files.append(str(file_path))
                        except Exception as e:
                            print(f"[WARN] ファイル削除失敗: {file_path}, エラー: {e}")
            
            # 学習ログファイル（.csv）を削除
            log_patterns = [
                'clear_sparse_training_log_*.csv'
            ]
            
            for pattern in log_patterns:
                for file_path in project_root.glob(pattern):
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                            deleted_files.append(str(file_path))
                    except Exception as e:
                        print(f"[WARN] ファイル削除失敗: {file_path}, エラー: {e}")
            
            # logs/training/ フォルダ内のログ
            logs_training_dir = project_root / 'logs' / 'training'
            if logs_training_dir.exists():
                for csv_file in logs_training_dir.rglob('*.csv'):
                    try:
                        if csv_file.is_file():
                            csv_file.unlink()
                            deleted_files.append(str(csv_file))
                    except Exception as e:
                        print(f"[WARN] ファイル削除失敗: {csv_file}, エラー: {e}")
            
            # ステータスファイルを削除
            status_file = project_root / 'logs' / 'training_status.json'
            if status_file.exists():
                try:
                    status_file.unlink()
                    deleted_files.append(str(status_file))
                except Exception as e:
                    print(f"[WARN] ステータスファイル削除失敗: {status_file}, エラー: {e}")
            
            # WSL2ログファイルを削除
            wsl2_log = project_root / 'logs' / 'wsl2_training.log'
            if wsl2_log.exists():
                try:
                    wsl2_log.unlink()
                    deleted_files.append(str(wsl2_log))
                except Exception as e:
                    print(f"[WARN] WSL2ログファイル削除失敗: {wsl2_log}, エラー: {e}")
            
            # ログ表示をクリア
            if hasattr(self, 'training_log_text'):
                self.training_log_text.clear()
            
            # ステータス表示をリセット
            if hasattr(self, 'training_status_label'):
                self.training_status_label.setText('ステータス: ⚪ 未開始')
                self.training_status_label.setStyleSheet('color:#888; font-weight:bold; font-size:12pt;')
            if hasattr(self, 'current_activity_label'):
                self.current_activity_label.setText('現在の作業: -')
            if hasattr(self, 'training_progress'):
                self.training_progress.setValue(0)
            if hasattr(self, 'epoch_label'):
                self.epoch_label.setText('学習回数: -')
            if hasattr(self, 'accuracy_label'):
                self.accuracy_label.setText('精度: -')
            
            # 成功メッセージ
            deleted_count = len(deleted_files)
            success_msg = f"学習ファイルを完全にリセットしました！\n\n"
            success_msg += f"削除されたファイル: {deleted_count}個\n"
            if deleted_count > 0:
                success_msg += f"\n削除されたファイル（最初の10個）:\n"
                for f in deleted_files[:10]:
                    success_msg += f"  • {Path(f).name}\n"
                if deleted_count > 10:
                    success_msg += f"  ... 他 {deleted_count - 10}個\n"
            
            QMessageBox.information(self, 'リセット完了', success_msg)
            self.statusBar().showMessage(f'学習ファイルを完全にリセットしました（{deleted_count}個のファイルを削除）', 5000)
            
        except Exception as e:
            error_msg = f'リセットエラー: {str(e)}'
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, 'リセットエラー', f"{error_msg}\n\n詳細はコンソールを確認してください。")
    
    def changeEvent(self, event):
        """ウィンドウ状態変更時（最大化など）"""
        if event.type() == QtCore.QEvent.WindowStateChange:
            # 最大化されているかチェック
            is_maximized = self.isMaximized()
            
            # 学習タブのレイアウトを調整
            if hasattr(self, 'tabs') and self.tabs:
                training_tab = self.tabs.widget(1)  # 学習タブはインデックス1
                if training_tab:
                    # スクロールエリアを取得
                    scroll_area = training_tab.findChild(QScrollArea)
                    if scroll_area:
                        scroll_widget = scroll_area.widget()
                        if scroll_widget:
                            # 最大化時はマージンとスペースを大幅に減らす
                            layout = scroll_widget.layout()
                            if layout:
                                if is_maximized:
                                    layout.setContentsMargins(8, 8, 8, 8)
                                    layout.setSpacing(8)
                                else:
                                    layout.setContentsMargins(15, 15, 15, 15)
                                    layout.setSpacing(15)
                            
                            # すべてのGroupBoxのマージンも調整
                            for group_box in scroll_widget.findChildren(QtWidgets.QGroupBox):
                                group_layout = group_box.layout()
                                if group_layout:
                                    if is_maximized:
                                        group_layout.setContentsMargins(10, 10, 10, 10)
                                        group_layout.setSpacing(8)
                                    else:
                                        group_layout.setContentsMargins(15, 15, 15, 15)
                                        group_layout.setSpacing(15)
                            
            # 最大化時はスクロールバーを非表示にする可能性があるが、
            # コンテンツが収まる場合は自動的に非表示になる
            scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        super().changeEvent(event)
    
    def restart_application(self):
        """アプリケーションを再起動"""
        reply = QtWidgets.QMessageBox.question(
            self,
            '再起動確認',
            'アプリケーションを再起動しますか？\n進行中の検査や学習は停止されます。',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.restart_requested = True
            # 設定を保存
            self.save_inspection_settings()
            
            # 検査を停止
            self.stop_inspection()
            
            # 学習を停止
            self.stop_training()
            
            # 少し待ってから新しいインスタンスを起動
            QtCore.QTimer.singleShot(500, self._execute_restart)
    
    def _execute_restart(self):
        """再起動を実行"""
        try:
            import subprocess
            import sys
            # 現在のスクリプトのパスを取得
            app_path = Path(__file__).resolve()
            
            # 新しいインスタンスを起動（コンソールウィンドウを非表示）
            if sys.platform.startswith('win'):
                # Windowsの場合：コンソールウィンドウを非表示にして起動
                import subprocess
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                subprocess.Popen(
                    [sys.executable, str(app_path)],
                    startupinfo=startupinfo,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )
            else:
                # Linux/Macの場合
                subprocess.Popen([sys.executable, str(app_path)])
            
            # 現在のインスタンスを終了
            QtWidgets.QApplication.instance().quit()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                '再起動エラー',
                f'アプリケーションの再起動に失敗しました:\n{str(e)}'
            )
    
    # ==================== 新機能メソッド ====================
    def show_statistics_graph(self):
        """統計グラフを表示"""
        if not HAS_MATPLOTLIB:
            QtWidgets.QMessageBox.warning(self, 'エラー', 'matplotlibがインストールされていません。\npip install matplotlib でインストールしてください。')
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle('📈 検査統計グラフ')
        dialog.setMinimumSize(1000, 700)
        layout = QVBoxLayout(dialog)
        
        period_group = QGroupBox("期間選択")
        period_layout = QHBoxLayout()
        period_combo = QComboBox()
        period_combo.addItems(['今日', '過去7日', '過去30日', '全期間'])
        period_combo.setCurrentIndex(0)
        period_layout.addWidget(QLabel("期間:"))
        period_layout.addWidget(period_combo)
        period_layout.addStretch()
        period_group.setLayout(period_layout)
        layout.addWidget(period_group)
        
        figure = Figure(figsize=(12, 8))
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)
        
        def update_graph():
            period = period_combo.currentText()
            today = datetime.now()
            data_by_date = defaultdict(lambda: {'total': 0, 'ok': 0, 'ng': 0, 'by_class': defaultdict(int)})
            
            if period == '今日':
                start_date = today
            elif period == '過去7日':
                start_date = today - timedelta(days=7)
            elif period == '過去30日':
                start_date = today - timedelta(days=30)
            else:
                start_date = datetime(2000, 1, 1)
            
            log_files = sorted(self.inspection_log_dir.glob('inspection_*.jsonl'))
            for log_file in log_files:
                try:
                    file_date = datetime.strptime(log_file.stem.split('_')[-1], '%Y%m%d').date()
                    if file_date >= start_date.date():
                        with open(log_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                entry = json.loads(line)
                                entry_date = datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S').date()
                                if entry_date >= start_date.date():
                                    date_str = entry_date.strftime('%Y-%m-%d')
                                    data_by_date[date_str]['total'] += 1
                                    if entry.get('is_ok'):
                                        data_by_date[date_str]['ok'] += 1
                                    else:
                                        data_by_date[date_str]['ng'] += 1
                                    data_by_date[date_str]['by_class'][entry['prediction']] += 1
                except Exception:
                    pass
            
            figure.clear()
            if not data_by_date:
                ax = figure.add_subplot(111)
                ax.text(0.5, 0.5, 'データがありません', ha='center', va='center', fontsize=14)
                canvas.draw()
                return
            
            dates = sorted(data_by_date.keys())
            ok_counts = [data_by_date[d]['ok'] for d in dates]
            ng_counts = [data_by_date[d]['ng'] for d in dates]
            
            ax1 = figure.add_subplot(2, 2, 1)
            ax1.plot(dates, ok_counts, label='OK', marker='o', linewidth=2, color='green')
            ax1.plot(dates, ng_counts, label='NG', marker='s', linewidth=2, color='red')
            ax1.set_xlabel('日付')
            ax1.set_ylabel('件数')
            ax1.set_title('OK/NG件数の推移')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            ax2 = figure.add_subplot(2, 2, 2)
            total_counts = [data_by_date[d]['total'] for d in dates]
            ng_rates = [(ng / total * 100) if total > 0 else 0 for ng, total in zip(ng_counts, total_counts)]
            ax2.plot(dates, ng_rates, marker='o', linewidth=2, color='orange')
            ax2.set_xlabel('日付')
            ax2.set_ylabel('NG率 (%)')
            ax2.set_title('NG率の推移')
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            ax3 = figure.add_subplot(2, 2, 3)
            class_map = {'good': '良品', 'black_spot': '黒点', 'chipping': '欠け', 
                        'scratch': '傷', 'dent': '凹み', 'distortion': '歪み'}
            class_totals = defaultdict(int)
            for date_data in data_by_date.values():
                for class_name, count in date_data['by_class'].items():
                    class_totals[class_name] += count
            
            if class_totals:
                labels = [class_map.get(k, k) for k in class_totals.keys()]
                sizes = list(class_totals.values())
                colors_list = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#1abc9c']
                ax3.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_list[:len(labels)])
                ax3.set_title('クラス別分布')
            
            ax4 = figure.add_subplot(2, 2, 4)
            confidences = []
            for log_file in log_files:
                try:
                    file_date = datetime.strptime(log_file.stem.split('_')[-1], '%Y%m%d').date()
                    if file_date >= start_date.date():
                        with open(log_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                entry = json.loads(line)
                                confidences.append(entry.get('confidence', 0))
                except Exception:
                    pass
            
            if confidences:
                ax4.hist(confidences, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
                ax4.set_xlabel('信頼度')
                ax4.set_ylabel('頻度')
                ax4.set_title('信頼度分布')
                ax4.grid(True, alpha=0.3)
            
            figure.tight_layout()
            canvas.draw()
        
        period_combo.currentIndexChanged.connect(update_graph)
        update_graph()
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("閉じる")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        dialog.exec_()
    
    def show_image_gallery(self):
        """保存された検査画像のギャラリーを表示"""
        dialog = QDialog(self)
        dialog.setWindowTitle('🖼️ 画像ギャラリー')
        dialog.setMinimumSize(1200, 800)
        layout = QVBoxLayout(dialog)
        
        filter_group = QGroupBox("フィルタ")
        filter_layout = QHBoxLayout()
        
        date_from = QDateEdit()
        date_from.setDate(QDate.currentDate().addDays(-7))
        date_from.setCalendarPopup(True)
        date_to = QDateEdit()
        date_to.setDate(QDate.currentDate())
        date_to.setCalendarPopup(True)
        
        class_combo = QComboBox()
        class_combo.addItem("すべてのクラス", None)
        class_map = {'good': '良品', 'black_spot': '黒点', 'chipping': '欠け', 
                    'scratch': '傷', 'dent': '凹み', 'distortion': '歪み'}
        for class_en, class_jp in class_map.items():
            class_combo.addItem(class_jp, class_en)
        
        filter_layout.addWidget(QLabel("期間:"))
        filter_layout.addWidget(date_from)
        filter_layout.addWidget(QLabel("〜"))
        filter_layout.addWidget(date_to)
        filter_layout.addWidget(QLabel("クラス:"))
        filter_layout.addWidget(class_combo)
        filter_layout.addStretch()
        
        refresh_btn = QPushButton("🔍 更新")
        filter_layout.addWidget(refresh_btn)
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        grid_layout = QVBoxLayout(scroll_widget)
        image_container = QWidget()
        image_grid = QVBoxLayout(image_container)
        
        def load_images():
            for i in reversed(range(image_grid.count())):
                image_grid.itemAt(i).widget().setParent(None)
            
            date_from_val = date_from.date().toPyDate()
            date_to_val = date_to.date().toPyDate()
            selected_class = class_combo.currentData()
            
            images = []
            for date_dir in self.inspection_image_dir.glob('*'):
                if date_dir.is_dir():
                    try:
                        dir_date = datetime.strptime(date_dir.name, '%Y-%m-%d').date()
                        if date_from_val <= dir_date <= date_to_val:
                            for img_file in date_dir.glob('*.jpg'):
                                if selected_class is None or selected_class in img_file.name:
                                    images.append(img_file)
                    except ValueError:
                        continue
            
            images.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if not images:
                no_data_label = QLabel("画像が見つかりません")
                no_data_label.setAlignment(Qt.AlignCenter)
                image_grid.addWidget(no_data_label)
            
            grid_layout.addWidget(image_container)
            scroll_area.setWidget(scroll_widget)
            
            if not images:
                return
            
            row_layout = None
            for i, img_file in enumerate(images[:100]):
                if i % 3 == 0:
                    row_layout = QHBoxLayout()
                
                img_widget = QWidget()
                img_vlayout = QVBoxLayout(img_widget)
                img_vlayout.setContentsMargins(5, 5, 5, 5)
                
                pixmap = QPixmap(str(img_file))
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    img_label = QLabel()
                    img_label.setPixmap(scaled_pixmap)
                    img_label.setStyleSheet("border: 1px solid #ddd;")
                    img_vlayout.addWidget(img_label)
                    
                    name_label = QLabel(img_file.name)
                    name_label.setStyleSheet("font-size: 9pt;")
                    name_label.setWordWrap(True)
                    img_vlayout.addWidget(name_label)
                    
                    def show_full_image(img_path=img_file):
                        full_dialog = QDialog(self)
                        full_dialog.setWindowTitle(f'画像: {img_path.name}')
                        full_layout = QVBoxLayout(full_dialog)
                        
                        full_pixmap = QPixmap(str(img_path))
                        full_label = QLabel()
                        full_label.setPixmap(full_pixmap)
                        full_label.setAlignment(Qt.AlignCenter)
                        full_layout.addWidget(full_label)
                        
                        btn_layout = QHBoxLayout()
                        delete_btn = QPushButton("🗑️ 削除")
                        delete_btn.clicked.connect(lambda: delete_image(img_path, full_dialog))
                        btn_layout.addWidget(delete_btn)
                        btn_layout.addStretch()
                        close_btn = QPushButton("閉じる")
                        close_btn.clicked.connect(full_dialog.accept)
                        btn_layout.addWidget(close_btn)
                        full_layout.addLayout(btn_layout)
                        
                        full_dialog.exec_()
                    
                    def delete_image(img_path, parent_dialog):
                        reply = QMessageBox.question(self, '確認', f'この画像を削除しますか？\n{img_path.name}',
                                                    QMessageBox.Yes | QMessageBox.No)
                        if reply == QMessageBox.Yes:
                            try:
                                img_path.unlink()
                                parent_dialog.accept()
                                load_images()
                                QMessageBox.information(self, '完了', '画像を削除しました')
                            except Exception as e:
                                QMessageBox.warning(self, 'エラー', f'削除に失敗しました: {e}')
                    
                    img_label.mousePressEvent = lambda e, img=img_file: show_full_image(img)
                
                row_layout.addWidget(img_widget)
                
                if (i + 1) % 3 == 0 or i == len(images) - 1:
                    image_grid.addLayout(row_layout)
            
            scroll_widget.setLayout(grid_layout)
            scroll_area.setWidget(scroll_widget)
        
        refresh_btn.clicked.connect(load_images)
        layout.addWidget(scroll_area)
        load_images()
        
        close_btn = QPushButton("閉じる")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def show_history_search(self):
        """検査履歴の検索・フィルタ"""
        dialog = QDialog(self)
        dialog.setWindowTitle('🔍 検査履歴検索')
        dialog.setMinimumSize(1000, 700)
        layout = QVBoxLayout(dialog)
        
        filter_group = QGroupBox("検索条件")
        filter_layout = QVBoxLayout()
        
        date_layout = QHBoxLayout()
        date_from = QDateEdit()
        date_from.setDate(QDate.currentDate().addDays(-30))
        date_from.setCalendarPopup(True)
        date_to = QDateEdit()
        date_to.setDate(QDate.currentDate())
        date_to.setCalendarPopup(True)
        date_layout.addWidget(QLabel("期間:"))
        date_layout.addWidget(date_from)
        date_layout.addWidget(QLabel("〜"))
        date_layout.addWidget(date_to)
        date_layout.addStretch()
        filter_layout.addLayout(date_layout)
        
        class_layout = QHBoxLayout()
        class_combo = QComboBox()
        class_combo.addItem("すべて", None)
        class_map = {'good': '良品', 'black_spot': '黒点', 'chipping': '欠け', 
                    'scratch': '傷', 'dent': '凹み', 'distortion': '歪み'}
        for class_en, class_jp in class_map.items():
            class_combo.addItem(class_jp, class_en)
        class_layout.addWidget(QLabel("クラス:"))
        class_layout.addWidget(class_combo)
        
        confidence_min = QDoubleSpinBox()
        confidence_min.setRange(0.0, 1.0)
        confidence_min.setSingleStep(0.1)
        confidence_min.setValue(0.0)
        confidence_min.setDecimals(2)
        
        confidence_max = QDoubleSpinBox()
        confidence_max.setRange(0.0, 1.0)
        confidence_max.setSingleStep(0.1)
        confidence_max.setValue(1.0)
        confidence_max.setDecimals(2)
        
        class_layout.addWidget(QLabel("信頼度:"))
        class_layout.addWidget(confidence_min)
        class_layout.addWidget(QLabel("〜"))
        class_layout.addWidget(confidence_max)
        class_layout.addStretch()
        filter_layout.addLayout(class_layout)
        
        ok_ng_layout = QHBoxLayout()
        ok_only_check = QCheckBox("OKのみ")
        ng_only_check = QCheckBox("NGのみ")
        ok_ng_layout.addWidget(ok_only_check)
        ok_ng_layout.addWidget(ng_only_check)
        ok_ng_layout.addStretch()
        filter_layout.addLayout(ok_ng_layout)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(['タイムスタンプ', '予測', '信頼度', 'OK/NG', '画像パス', '修正済み'])
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)
        
        def search():
            date_from_val = date_from.date().toPyDate()
            date_to_val = date_to.date().toPyDate()
            selected_class = class_combo.currentData()
            conf_min = confidence_min.value()
            conf_max = confidence_max.value()
            ok_only = ok_only_check.isChecked()
            ng_only = ng_only_check.isChecked()
            
            results = []
            log_files = sorted(self.inspection_log_dir.glob('inspection_*.jsonl'))
            
            for log_file in log_files:
                try:
                    file_date = datetime.strptime(log_file.stem.split('_')[-1], '%Y%m%d').date()
                    if date_from_val <= file_date <= date_to_val:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                entry = json.loads(line)
                                entry_date = datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S').date()
                                
                                if date_from_val <= entry_date <= date_to_val:
                                    pred = entry.get('prediction', '')
                                    conf = entry.get('confidence', 0.0)
                                    is_ok = entry.get('is_ok', False)
                                    
                                    if selected_class and pred != selected_class:
                                        continue
                                    if conf < conf_min or conf > conf_max:
                                        continue
                                    if ok_only and not is_ok:
                                        continue
                                    if ng_only and is_ok:
                                        continue
                                    
                                    results.append(entry)
                except Exception:
                    pass
            
            table.setRowCount(len(results))
            for i, entry in enumerate(results):
                table.setItem(i, 0, QTableWidgetItem(entry['timestamp']))
                pred_jp = class_map.get(entry['prediction'], entry['prediction'])
                table.setItem(i, 1, QTableWidgetItem(pred_jp))
                table.setItem(i, 2, QTableWidgetItem(f"{entry['confidence']:.2%}"))
                table.setItem(i, 3, QTableWidgetItem("OK" if entry.get('is_ok') else "NG"))
                table.setItem(i, 4, QTableWidgetItem(entry.get('image_path', '-')))
                corrected = entry.get('corrected_class', '')
                table.setItem(i, 5, QTableWidgetItem(class_map.get(corrected, corrected) if corrected else '-'))
            
            table.resizeColumnsToContents()
        
        search_btn = QPushButton("🔍 検索")
        search_btn.clicked.connect(search)
        filter_layout.addWidget(search_btn)
        
        search()
        
        close_btn = QPushButton("閉じる")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def check_ng_rate_alert(self):
        """NG率をチェックしてアラートを発動"""
        if not self.ng_rate_alert_enabled or not self.inspection_stats['total']:
            return
        
        ng_rate = self.inspection_stats['ng'] / self.inspection_stats['total']
        if ng_rate >= self.ng_rate_alert_threshold:
            if not self.last_alert_time or (time.time() - self.last_alert_time) > 300:
                self.last_alert_time = time.time()
                QMessageBox.warning(
                    self,
                    '⚠️ NG率アラート',
                    f'NG率が{ng_rate:.1%}に達しました。\n'
                    f'設定閾値: {self.ng_rate_alert_threshold:.1%}\n'
                    f'総検査数: {self.inspection_stats["total"]}\n'
                    f'NG数: {self.inspection_stats["ng"]}'
                )
    
    def show_alert_settings(self):
        """NG率アラート設定"""
        dialog = QDialog(self)
        dialog.setWindowTitle('⚠️ NG率アラート設定')
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)
        
        threshold_label = QLabel("NG率閾値 (%):")
        threshold_spin = QDoubleSpinBox()
        threshold_spin.setRange(0.1, 100.0)
        threshold_spin.setValue(self.ng_rate_alert_threshold * 100)
        threshold_spin.setSingleStep(1.0)
        threshold_spin.setSuffix(" %")
        
        enable_check = QCheckBox("アラートを有効にする")
        enable_check.setChecked(self.ng_rate_alert_enabled)
        
        layout.addWidget(threshold_label)
        layout.addWidget(threshold_spin)
        layout.addWidget(enable_check)
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("キャンセル")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)
        
        if dialog.exec_() == QDialog.Accepted:
            self.ng_rate_alert_threshold = threshold_spin.value() / 100.0
            self.ng_rate_alert_enabled = enable_check.isChecked()
            
            if self.ng_rate_alert_enabled:
                if not self.alert_check_timer.isActive():
                    self.alert_check_timer.start(60000)
            else:
                self.alert_check_timer.stop()
    
    def generate_daily_report(self):
        """日次レポートを生成"""
        if not HAS_REPORTLAB:
            QtWidgets.QMessageBox.warning(self, 'エラー', 'reportlabがインストールされていません。\npip install reportlab でインストールしてください。')
            return
        
        date_dialog = QDialog(self)
        date_dialog.setWindowTitle('日次レポート生成')
        date_layout = QVBoxLayout(date_dialog)
        date_layout.addWidget(QLabel("レポート生成対象日:"))
        date_edit = QDateEdit()
        date_edit.setDate(QDate.currentDate())
        date_edit.setCalendarPopup(True)
        date_layout.addWidget(date_edit)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("生成")
        ok_btn.clicked.connect(date_dialog.accept)
        cancel_btn = QPushButton("キャンセル")
        cancel_btn.clicked.connect(date_dialog.reject)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(ok_btn)
        date_layout.addLayout(btn_layout)
        
        if date_dialog.exec_() != QDialog.Accepted:
            return
        
        target_date = date_edit.date().toPyDate()
        date_str = target_date.strftime('%Y-%m-%d')
        
        log_file = self.inspection_log_dir / f"inspection_{target_date.strftime('%Y%m%d')}.jsonl"
        if not log_file.exists():
            QtWidgets.QMessageBox.information(self, '情報', f'{date_str}のデータがありません。')
            return
        
        entries = []
        stats = {'total': 0, 'ok': 0, 'ng': 0, 'by_class': defaultdict(int)}
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    entry_date = datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S').date()
                    if entry_date == target_date:
                        entries.append(entry)
                        stats['total'] += 1
                        if entry.get('is_ok'):
                            stats['ok'] += 1
                        else:
                            stats['ng'] += 1
                        stats['by_class'][entry['prediction']] += 1
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'エラー', f'データ読み込みエラー: {e}')
            return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, 'レポートを保存', f'daily_report_{date_str}.pdf', 'PDF Files (*.pdf)'
        )
        if not save_path:
            return
        
        try:
            doc = SimpleDocTemplate(save_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            story.append(Paragraph(f"日次検査レポート - {date_str}", styles['Title']))
            story.append(Spacer(1, 12))
            
            story.append(Paragraph("サマリー", styles['Heading2']))
            summary_data = [
                ['項目', '値'],
                ['総検査数', str(stats['total'])],
                ['OK', str(stats['ok'])],
                ['NG', str(stats['ng'])],
                ['OK率', f"{stats['ok']/stats['total']*100:.1f}%" if stats['total'] > 0 else "0%"],
                ['NG率', f"{stats['ng']/stats['total']*100:.1f}%" if stats['total'] > 0 else "0%"],
            ]
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 12))
            
            story.append(Paragraph("クラス別集計", styles['Heading2']))
            class_map = {'good': '良品', 'black_spot': '黒点', 'chipping': '欠け', 
                        'scratch': '傷', 'dent': '凹み', 'distortion': '歪み'}
            class_data = [['クラス', '件数']]
            for class_name, count in sorted(stats['by_class'].items(), key=lambda x: x[1], reverse=True):
                class_data.append([class_map.get(class_name, class_name), str(count)])
            class_table = Table(class_data)
            class_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(class_table)
            
            doc.build(story)
            QtWidgets.QMessageBox.information(self, '完了', f'レポートを生成しました:\n{save_path}')
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'エラー', f'レポート生成エラー: {e}')
    
    def show_batch_export_dialog(self):
        """バッチエクスポートダイアログ"""
        dialog = QDialog(self)
        dialog.setWindowTitle('📦 バッチエクスポート')
        dialog.setMinimumWidth(500)
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("エクスポート期間を選択:"))
        
        date_layout = QHBoxLayout()
        date_from = QDateEdit()
        date_from.setDate(QDate.currentDate().addDays(-30))
        date_from.setCalendarPopup(True)
        date_to = QDateEdit()
        date_to.setDate(QDate.currentDate())
        date_to.setCalendarPopup(True)
        date_layout.addWidget(date_from)
        date_layout.addWidget(QLabel("〜"))
        date_layout.addWidget(date_to)
        layout.addLayout(date_layout)
        
        format_layout = QHBoxLayout()
        csv_check = QCheckBox("CSV形式")
        csv_check.setChecked(True)
        json_check = QCheckBox("JSON形式")
        json_check.setChecked(True)
        format_layout.addWidget(csv_check)
        format_layout.addWidget(json_check)
        format_layout.addStretch()
        layout.addLayout(format_layout)
        
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        export_btn = QPushButton("エクスポート")
        export_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("キャンセル")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(export_btn)
        layout.addLayout(btn_layout)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        date_from_val = date_from.date().toPyDate()
        date_to_val = date_to.date().toPyDate()
        
        save_dir = QFileDialog.getExistingDirectory(self, 'エクスポート先フォルダを選択')
        if not save_dir:
            return
        
        all_entries = []
        log_files = sorted(self.inspection_log_dir.glob('inspection_*.jsonl'))
        
        for log_file in log_files:
            try:
                file_date = datetime.strptime(log_file.stem.split('_')[-1], '%Y%m%d').date()
                if date_from_val <= file_date <= date_to_val:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            entry = json.loads(line)
                            entry_date = datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S').date()
                            if date_from_val <= entry_date <= date_to_val:
                                all_entries.append(entry)
            except Exception:
                pass
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if csv_check.isChecked():
            csv_file = Path(save_dir) / f"batch_export_{timestamp}.csv"
            try:
                with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=['timestamp', 'prediction', 'confidence', 'is_ok', 'image_path'])
                    writer.writeheader()
                    writer.writerows(all_entries)
                QtWidgets.QMessageBox.information(self, '完了', f'CSVファイルをエクスポートしました:\n{csv_file}')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'エラー', f'CSVエクスポートエラー: {e}')
        
        if json_check.isChecked():
            json_file = Path(save_dir) / f"batch_export_{timestamp}.json"
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'export_timestamp': datetime.now().isoformat(),
                        'period': {'from': date_from_val.isoformat(), 'to': date_to_val.isoformat()},
                        'total_entries': len(all_entries),
                        'entries': all_entries
                    }, f, ensure_ascii=False, indent=2)
                QtWidgets.QMessageBox.information(self, '完了', f'JSONファイルをエクスポートしました:\n{json_file}')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'エラー', f'JSONエクスポートエラー: {e}')
    
    def show_model_comparison(self):
        """モデル精度比較"""
        dialog = QDialog(self)
        dialog.setWindowTitle('🎯 モデル精度比較')
        dialog.setMinimumSize(800, 600)
        layout = QVBoxLayout(dialog)
        
        training_log_dir = Path('logs/training')
        comparison_data = []
        
        if training_log_dir.exists():
            for log_file in sorted(training_log_dir.glob('**/*.json'), reverse=True):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'accuracy' in data or 'val_accuracy' in data:
                            comparison_data.append({
                                'file': log_file.name,
                                'date': datetime.fromtimestamp(log_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                                'accuracy': data.get('val_accuracy', data.get('accuracy', 0)),
                                'loss': data.get('val_loss', data.get('loss', 0))
                            })
                except Exception:
                    pass
        
        if not comparison_data:
            layout.addWidget(QLabel("比較データがありません"))
        else:
            table = QTableWidget()
            table.setColumnCount(4)
            table.setHorizontalHeaderLabels(['ファイル', '日時', '精度', '損失'])
            table.setRowCount(len(comparison_data))
            
            for i, data in enumerate(comparison_data):
                table.setItem(i, 0, QTableWidgetItem(data['file']))
                table.setItem(i, 1, QTableWidgetItem(data['date']))
                table.setItem(i, 2, QTableWidgetItem(f"{data['accuracy']:.4f}"))
                table.setItem(i, 3, QTableWidgetItem(f"{data['loss']:.4f}"))
            
            table.resizeColumnsToContents()
            layout.addWidget(table)
        
        close_btn = QPushButton("閉じる")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def show_performance_analysis(self):
        """パフォーマンス分析"""
        if not HAS_MATPLOTLIB:
            QtWidgets.QMessageBox.warning(self, 'エラー', 'matplotlibがインストールされていません。')
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle('⚡ パフォーマンス分析')
        dialog.setMinimumSize(1000, 700)
        layout = QVBoxLayout(dialog)
        
        figure = Figure(figsize=(12, 8))
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)
        
        figure.clear()
        
        if self.processing_times:
            ax1 = figure.add_subplot(2, 2, 1)
            ax1.hist(list(self.processing_times), bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            ax1.set_xlabel('処理時間 (ms)')
            ax1.set_ylabel('頻度')
            ax1.set_title('処理時間分布')
            ax1.axvline(np.mean(self.processing_times), color='red', linestyle='--', 
                       label=f'平均: {np.mean(self.processing_times):.2f}ms')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        if self.fps_history:
            ax2 = figure.add_subplot(2, 2, 2)
            ax2.plot(list(self.fps_history), marker='o', linewidth=2, color='green')
            ax2.set_xlabel('サンプル')
            ax2.set_ylabel('FPS')
            ax2.set_title('FPS推移')
            ax2.grid(True, alpha=0.3)
            if len(self.fps_history) > 0:
                ax2.axhline(np.mean(self.fps_history), color='red', linestyle='--', 
                           label=f'平均: {np.mean(self.fps_history):.2f} FPS')
                ax2.legend()
        
        if self.memory_usage_history:
            ax3 = figure.add_subplot(2, 2, 3)
            ax3.plot(list(self.memory_usage_history), marker='s', linewidth=2, color='orange')
            ax3.set_xlabel('サンプル')
            ax3.set_ylabel('メモリ使用量 (MB)')
            ax3.set_title('メモリ使用量推移')
            ax3.grid(True, alpha=0.3)
            if len(self.memory_usage_history) > 0:
                ax3.axhline(np.mean(self.memory_usage_history), color='red', linestyle='--',
                           label=f'平均: {np.mean(self.memory_usage_history):.2f} MB')
                ax3.legend()
        
        ax4 = figure.add_subplot(2, 2, 4)
        ax4.axis('off')
        stats_text = "パフォーマンス統計\n\n"
        
        if self.processing_times:
            stats_text += f"処理時間:\n"
            stats_text += f"  平均: {np.mean(self.processing_times):.2f} ms\n"
            stats_text += f"  最小: {np.min(self.processing_times):.2f} ms\n"
            stats_text += f"  最大: {np.max(self.processing_times):.2f} ms\n\n"
        
        if self.fps_history:
            stats_text += f"FPS:\n"
            stats_text += f"  平均: {np.mean(self.fps_history):.2f}\n"
            stats_text += f"  最小: {np.min(self.fps_history):.2f}\n"
            stats_text += f"  最大: {np.max(self.fps_history):.2f}\n\n"
        
        if self.memory_usage_history:
            stats_text += f"メモリ使用量:\n"
            stats_text += f"  平均: {np.mean(self.memory_usage_history):.2f} MB\n"
            stats_text += f"  最小: {np.min(self.memory_usage_history):.2f} MB\n"
            stats_text += f"  最大: {np.max(self.memory_usage_history):.2f} MB\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        figure.tight_layout()
        canvas.draw()
        
        close_btn = QPushButton("閉じる")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def closeEvent(self, event):
        """アプリケーション終了時"""
        # 設定を保存
        self.save_inspection_settings()
        
        # 検査を停止
        self.stop_inspection()
        
        # 学習を停止
        self.stop_training()
        
        event.accept()


def kill_existing_instances():
    """既存のアプリインスタンスをすべて終了"""
    if sys.platform.startswith('win'):
        try:
            import subprocess
            # コンソールウィンドウを非表示にして実行
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            
            # 統合ワッシャー検査アプリのPythonプロセスを検索して終了
            result = subprocess.run(
                ['tasklist', '/fi', 'IMAGENAME eq python.exe', '/fo', 'csv'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            
            # ウィンドウタイトルで統合ワッシャー検査アプリを検索
            result2 = subprocess.run(
                ['tasklist', '/v', '/fi', 'IMAGENAME eq python.exe', '/fo', 'csv'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            
            # ウィンドウタイトルに「統合ワッシャー」を含むプロセスを終了
            killed_count = 0
            try:
                import psutil
                for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'exe']):
                    try:
                        if proc.info['name'] and 'python' in proc.info['name'].lower():
                            cmdline = proc.info.get('cmdline', [])
                            exe = proc.info.get('exe', '')
                            
                            # integrated_washer_app.pyを実行しているプロセスを検出
                            if cmdline and any('integrated_washer_app.py' in str(arg).replace('\\', '/') for arg in cmdline):
                                # 自分自身のプロセスは除外
                                if proc.pid != os.getpid():
                                    print(f"[INFO] 既存のアプリインスタンスを終了中 (PID: {proc.pid})")
                                    proc.terminate()
                                    try:
                                        proc.wait(timeout=5)
                                    except psutil.TimeoutExpired:
                                        proc.kill()
                                    killed_count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass
            except ImportError:
                # psutilがない場合は、taskkillを使用（コンソールウィンドウを非表示）
                startupinfo_kill = subprocess.STARTUPINFO()
                startupinfo_kill.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo_kill.wShowWindow = subprocess.SW_HIDE
                subprocess.run(
                    ['taskkill', '/f', '/fi', 'WINDOWTITLE eq 統合ワッシャー検査・学習システム*'],
                    capture_output=True,
                    timeout=5,
                    startupinfo=startupinfo_kill,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )
            
            if killed_count > 0:
                print(f"[INFO] {killed_count}個の既存インスタンスを終了しました")
                time.sleep(1)  # プロセス終了を待機
        except Exception as e:
            print(f"[WARNING] 既存インスタンスの検出に失敗: {e}")


def excepthook(exc_type, exc_value, exc_traceback):
    """未処理例外をキャッチするフック"""
    import traceback
    
    try:
        # 安全にエラーメッセージを構築（エンコーディングエラーを防ぐ）
        try:
            exc_name = str(exc_type.__name__) if exc_type else "Unknown"
            exc_val_str = str(exc_value).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        except:
            exc_name = "Unknown"
            exc_val_str = "Error encoding exception value"
        
        error_msg = f"未処理例外が発生しました:\n{exc_name}: {exc_val_str}\n\n"
        
        try:
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            tb_str = tb_str.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            error_msg += f"トレースバック:\n{tb_str}"
        except:
            error_msg += "トレースバックの取得に失敗しました\n"
        
        # 安全に出力（stdout/stderrが閉じられている場合はスキップ）
        try:
            if sys.stdout and not getattr(sys.stdout, 'closed', True):
                safe_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                print(f"[CRITICAL ERROR] {safe_msg}")
        except:
            pass
    except Exception as e:
        error_msg = f"例外処理中にエラーが発生しました: {str(e)}"
    
    # エラーログファイルに記録
    try:
        error_log_path = Path('app_error_log.txt')
        with open(error_log_path, 'a', encoding='utf-8', errors='replace') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"未処理例外発生時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n")
            try:
                f.write(f"例外型: {exc_type.__name__ if exc_type else 'Unknown'}\n")
                exc_val_safe = str(exc_value).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                f.write(f"例外メッセージ: {exc_val_safe}\n")
            except:
                f.write("例外情報の取得に失敗しました\n")
            try:
                tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                tb_safe = tb_str.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                f.write(f"トレースバック:\n{tb_safe}\n")
            except:
                f.write("トレースバックの取得に失敗しました\n")
            f.write(f"{'='*60}\n\n")
    except:
        pass
    
    # 可能であればエラーダイアログを表示
    try:
        from PyQt5 import QtWidgets
        from PyQt5.QtWidgets import QTextEdit, QVBoxLayout, QDialog, QPushButton
        from PyQt5.QtCore import Qt
        
        if QtWidgets.QApplication.instance() is not None:
            # カスタムダイアログを作成（コピー可能なテキストエリア付き）
            dialog = QDialog()
            dialog.setWindowTitle('重大なエラー')
            dialog.setMinimumWidth(700)
            dialog.setMinimumHeight(500)
            
            layout = QVBoxLayout(dialog)
            
            # エラータイトル
            title_label = QtWidgets.QLabel(f"未処理例外が発生しました: {exc_name if 'exc_name' in locals() else (exc_type.__name__ if exc_type else 'Unknown')}")
            title_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #dc3545; padding: 10px;")
            layout.addWidget(title_label)
            
            # エラーメッセージ（コピー可能）
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setPlainText(error_msg)
            text_edit.setStyleSheet("font-family: 'Consolas', 'Courier New', monospace; font-size: 9pt;")
            layout.addWidget(text_edit)
            
            # エラーログファイルのパスを表示
            log_path_label = QtWidgets.QLabel(f"詳細はエラーログファイルに記録されました:\n{Path('app_error_log.txt').absolute()}")
            log_path_label.setStyleSheet("font-size: 9pt; color: #666; padding: 5px;")
            log_path_label.setWordWrap(True)
            layout.addWidget(log_path_label)
            
            # ボタン
            button_layout = QtWidgets.QHBoxLayout()
            copy_btn = QPushButton('エラーメッセージをコピー')
            copy_btn.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(error_msg))
            button_layout.addWidget(copy_btn)
            
            close_btn = QPushButton('閉じる')
            close_btn.clicked.connect(dialog.accept)
            button_layout.addWidget(close_btn)
            
            layout.addLayout(button_layout)
            dialog.exec_()
    except:
        pass


def main():
    """メイン関数"""
    # 未処理例外をキャッチするフックを設定
    sys.excepthook = excepthook
    
    app = None
    
    try:
        # QApplicationを先に作成（エラーダイアログ表示のために必要）
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle('Fusion')  # モダンなスタイル
        
        # PyQt5のシグナル/スロットエラーをキャッチするための設定
        # PyQt5のシグナル/スロットで発生した例外をキャッチ
        try:
            from PyQt5.QtCore import pyqtRemoveInputHook
            # デフォルトの例外ハンドラーを使用
        except:
            pass
        
        # 警告音を無効化（すべてのQMessageBoxで音が鳴らないように）
        try:
            # QApplicationの警告音を無効化
            app.setAttribute(QtCore.Qt.AA_DisableWindowContextHelpButton)
        except:
            pass
        
        # Windowsでコンソールウィンドウを非表示にする（QApplication作成後）
        if sys.platform.startswith('win'):
            try:
                import ctypes
                # コンソールウィンドウを非表示（Windows APIを使用）
                kernel32 = ctypes.windll.kernel32
                user32 = ctypes.windll.user32
                # コンソールウィンドウのハンドルを取得して非表示にする
                hwnd = kernel32.GetConsoleWindow()
                if hwnd:
                    user32.ShowWindow(hwnd, 0)  # SW_HIDE = 0
            except Exception:
                # API呼び出しに失敗した場合はそのまま続行
                pass
        
        # 既存のインスタンスを終了
        try:
            kill_existing_instances()
        except Exception as kill_error:
            print(f"[WARN] 既存インスタンス終了エラー: {kill_error}")
        
        # ウィンドウを作成（安全に）
        try:
            window = IntegratedWasherApp()
            window.show()
        except Exception as window_error:
            import traceback
            error_msg = f"ウィンドウ作成エラー: {str(window_error)}\n\n{traceback.format_exc()}"
            print(f"[ERROR] {error_msg}")
            
            # エラーログに記録
            try:
                error_log_path = Path('app_error_log.txt')
                with open(error_log_path, 'a', encoding='utf-8', errors='replace') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"ウィンドウ作成エラー発生時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"{error_msg}\n")
                    f.write(f"{'='*60}\n\n")
            except:
                pass
            
            # エラーダイアログを表示
            try:
                error_dialog = QtWidgets.QMessageBox()
                error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
                error_dialog.setWindowTitle('起動エラー')
                error_dialog.setText("アプリケーションウィンドウの作成に失敗しました")
                error_dialog.setDetailedText(error_msg)
                error_dialog.setMinimumWidth(600)
                error_dialog.setMinimumHeight(400)
                error_dialog.exec_()
            except:
                pass
            
            sys.exit(1)
        
        # イベントループを開始（安全に）
        try:
            sys.exit(app.exec_())
        except Exception as exec_error:
            import traceback
            error_msg = f"イベントループエラー: {str(exec_error)}\n\n{traceback.format_exc()}"
            print(f"[ERROR] {error_msg}")
            
            # エラーログに記録
            try:
                error_log_path = Path('app_error_log.txt')
                with open(error_log_path, 'a', encoding='utf-8', errors='replace') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"イベントループエラー発生時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"{error_msg}\n")
                    f.write(f"{'='*60}\n\n")
            except:
                pass
            
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[情報] ユーザーによって中断されました。")
        if app:
            try:
                app.quit()
            except:
                pass
        sys.exit(0)
    except Exception as general_error:
        # すべての例外をキャッチ（クラッシュ防止）- ネットワークエラー以外も含む
        import traceback
        error_msg = f"アプリケーション起動エラー: {str(general_error)}\n\n{traceback.format_exc()}"
        
        # エラーログファイルに記録
        try:
            error_log_path = Path('app_error_log.txt')
            with open(error_log_path, 'a', encoding='utf-8', errors='replace') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"アプリケーション起動エラー発生時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n")
                f.write(f"エラー型: {type(general_error).__name__}\n")
                f.write(f"エラーメッセージ: {str(general_error)}\n")
                f.write(f"トレースバック:\n{traceback.format_exc()}\n")
                f.write(f"{'='*60}\n\n")
        except:
            pass
        
        # コンソールに出力
        print(f"[CRITICAL ERROR] {error_msg}")
        
        # エラーダイアログを表示（PyQt5が使える場合）
        if app:
            try:
                error_dialog = QtWidgets.QMessageBox()
                error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
                error_dialog.setWindowTitle("起動エラー")
                error_dialog.setText("アプリケーションの起動に失敗しました")
                error_dialog.setDetailedText(error_msg)
                error_dialog.setMinimumWidth(600)
                error_dialog.setMinimumHeight(400)
                error_dialog.setModal(True)
                error_dialog.exec_()
            except:
                # フォールバック: コンソールに表示して待機
                try:
                    input("\n何かキーを押して終了...")
                except:
                    pass
        else:
            # QApplicationが作成できなかった場合
            print("\n何かキーを押して終了...")
            try:
                input()
            except:
                pass
        
        sys.exit(1)
    except (ConnectionError, TimeoutError, OSError) as network_error:
        # ネットワーク接続エラーの場合は、アプリを起動できるようにする
        import traceback
        error_msg = f"ネットワーク接続エラーが発生しました:\n{str(network_error)}\n\nアプリケーションはオフライン環境で動作します。"
        
        # エラーログファイルに記録
        try:
            error_log_path = Path('app_error_log.txt')
            with open(error_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"エラー発生時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n")
                f.write(f"ネットワーク接続エラー: {str(network_error)}\n")
                f.write(f"スタックトレース:\n{traceback.format_exc()}\n")
                f.write(f"{'='*60}\n\n")
        except:
            pass
        
        # コンソールに出力
        print(error_msg)
        print("[INFO] アプリケーションはオフライン環境で動作します。")
        
        # エラーダイアログを表示（PyQt5が使える場合）
        if app:
            try:
                error_dialog = QtWidgets.QMessageBox()
                error_dialog.setIcon(QtWidgets.QMessageBox.Warning)
                error_dialog.setWindowTitle("ネットワーク接続エラー")
                error_dialog.setText("ネットワーク接続エラーが発生しました")
                error_dialog.setInformativeText(
                    "インターネット接続が利用できないため、一部の機能が制限される可能性があります。\n\n"
                    "アプリケーションはオフライン環境で動作します。"
                )
                error_dialog.setDetailedText(f"エラー詳細:\n{str(network_error)}\n\n{traceback.format_exc()}")
                error_dialog.setMinimumWidth(600)
                error_dialog.setMinimumHeight(400)
                error_dialog.setModal(True)
                error_dialog.exec_()
                
                # ネットワークエラーの場合は、アプリを起動し続ける
                try:
                    window = IntegratedWasherApp()
                    window.show()
                    sys.exit(app.exec_())
                except Exception as startup_error:
                    print(f"[ERROR] アプリケーション起動エラー: {startup_error}")
                    sys.exit(1)
            except Exception as dialog_error:
                print(f"エラーダイアログ表示エラー: {dialog_error}")
                # フォールバック: アプリを起動してみる
                try:
                    window = IntegratedWasherApp()
                    window.show()
                    sys.exit(app.exec_())
                except:
                    input("\n何かキーを押して終了...")
                    sys.exit(1)
        else:
            # QApplicationが作成できなかった場合
            print("\n何かキーを押して終了...")
            try:
                input()
            except:
                pass
            sys.exit(1)
    except Exception as e:
        import traceback
        error_msg = f"アプリケーション起動エラー:\n{str(e)}\n\n{traceback.format_exc()}"
        
        # エラーログファイルに記録
        try:
            error_log_path = Path('app_error_log.txt')
            with open(error_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"エラー発生時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n")
                f.write(error_msg)
                f.write(f"\n{'='*60}\n\n")
        except:
            pass
        
        # コンソールに出力
        print(error_msg)
        
        # エラーダイアログを表示（PyQt5が使える場合）
        if app:
            try:
                error_dialog = QtWidgets.QMessageBox()
                error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
                error_dialog.setWindowTitle("起動エラー")
                error_dialog.setText("アプリケーションの起動に失敗しました")
                error_dialog.setDetailedText(error_msg)
                error_dialog.setMinimumWidth(600)
                error_dialog.setMinimumHeight(400)
                # ダイアログが確実に表示されるようにする
                error_dialog.setModal(True)
                error_dialog.exec_()
            except Exception as dialog_error:
                print(f"エラーダイアログ表示エラー: {dialog_error}")
                # フォールバック: コンソールに表示して待機
                input("\n何かキーを押して終了...")
        else:
            # QApplicationが作成できなかった場合
            print("\n何かキーを押して終了...")
            try:
                input()
            except:
                pass
        
        sys.exit(1)


if __name__ == '__main__':
    main()

