#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
不良品データ追加後のスパースラーニングと再学習
常に全体と項目ごとの進捗％と残り時間を表示
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
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
    import sys
    import os
    scripts_dir = os.path.join(os.path.dirname(__file__), '.')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from system_detector import SystemSpecDetector
    HAS_SYSTEM_DETECTOR = True
except ImportError:
    HAS_SYSTEM_DETECTOR = False

class EnhancedProgressSparseTrainer:
    def __init__(self, data_path="cs_AItraining_data"):
        self.data_path = Path(data_path)
        self.class_names = ['good', 'black_spot', 'chipping', 'scratch']
        self.models = []
        self.histories = []
        
        # 全体の時間管理
        self.total_start_time = None
        self.section_start_time = None
        self.section_name = ""
        self.total_steps = 0
        self.current_step = 0
        
        # システムスペック検出
        if HAS_SYSTEM_DETECTOR:
            try:
                self.system_detector = SystemSpecDetector()
                self.system_config = self.system_detector.config
                self.system_specs = self.system_detector.specs
                self.batch_size = self.system_config['batch_size']
                self.max_epochs = self.system_config['epochs']
                self.patience = self.system_config['patience']
                self.use_mixed_precision = self.system_config['use_mixed_precision']
            except Exception as e:
                print(f"警告: システムスペック検出に失敗: {e}")
                self.batch_size = 16
                self.max_epochs = 200
                self.patience = 30
                self.use_mixed_precision = False
        else:
            self.batch_size = 16
            self.max_epochs = 200
            self.patience = 30
            self.use_mixed_precision = False
        
        # スパースモデリング設定
        self.sparse_augmentation_params = {
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
        self.sparse_regularization = {
            'l1_lambda': 0.001,
            'l2_lambda': 0.0001,
            'dropout_rate': 0.5,
        }
        
        # 混合精度トレーニング
        if self.use_mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("混合精度トレーニングを有効にしました")
            except Exception as e:
                print(f"警告: 混合精度設定に失敗: {e}")
    
    def format_time(self, seconds):
        """時間を読みやムい形式に変換"""
        if seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}分{secs}秒"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}時間{mins}分{secs}秒"
    
    def print_progress(self, current, total, prefix="", suffix="", 
                      overall_current=None, overall_total=None):
        """進捗を表示（全体と項目ごと）"""
        if total == 0:
            return
        
        # 項目の進捗
        item_percent = 100 * (current / float(total))
        filled_length = int(50 * current // total)
        bar = '=' * filled_length + '-' * (50 - filled_length)
        
        # 項目の時間情報
        item_info = ""
        if self.section_start_time and current > 0:
            item_elapsed = time.time() - self.section_start_time
            if current > 0:
                avg_time = item_elapsed / current
                remaining_items = total - current
                item_remaining = avg_time * remaining_items
                item_info = f" | 項目: {item_percent:.1f}% | 項目経過: {self.format_time(item_elapsed)} | 項目残り: {self.format_time(item_remaining)}"
        
        # 全体の進捗
        overall_info = ""
        if overall_current is not None and overall_total is not None and self.total_start_time:
            overall_percent = 100 * (overall_current / float(overall_total))
            overall_elapsed = time.time() - self.total_start_time
            if overall_current > 0:
                avg_time = overall_elapsed / overall_current
                remaining_steps = overall_total - overall_current
                overall_remaining = avg_time * remaining_steps
                overall_info = f" | 【全体: {overall_percent:.1f}% | 全体経過: {self.format_time(overall_elapsed)} | 全体残り: {self.format_time(overall_remaining)}】"
        
        progress_info = f"\r{prefix} |{bar}|{item_info}{overall_info} - {suffix}"
        print(progress_info, end='', flush=True)
        if current == total:
            print()
    
    def load_all_data(self):
        """すべてのデータを読み込み（新しい不良品データを含む）"""
        self.total_start_time = time.time()
        self.section_start_time = time.time()
        self.section_name = "データ読み込み"
        
        print("\n" + "=" * 80)
        print("不良品データ追加後のデータ読み込み")
        print("=" * 80)
        
        images = []
        labels = []
        class_counts = {}
        
        total_classes = len(self.class_names)
        
        # 全体のステップ数を計算（データ読み込み = 10%）
        self.total_steps = 100
        
        for class_idx, class_name in enumerate(self.class_names):
            self.print_progress(class_idx, total_classes, f"クラス読み込み", 
                              f"{class_name}を処理中",
                              overall_current=class_idx * 2, overall_total=self.total_steps)
            
            class_path = self.data_path / 'resin' / class_name
            if not class_path.exists():
                print(f"\n警告: {class_path} が存在しません")
                continue
            
            class_images = []
            print(f"\n[{class_name}] 画像を読み込み中...")
            
            # すべての画像ファイルを再帰的に検索
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(class_path.rglob(ext))
            
            print(f"[{class_name}] {len(image_files)} 個の画像ファイルが見つかりました")
            
            total_files = len(image_files)
            for file_idx, img_file in enumerate(image_files):
                if file_idx % 5 == 0 or file_idx == total_files - 1:
                    self.print_progress(file_idx + 1, total_files, 
                                      f"{class_name}読み込み",
                                      f"{file_idx + 1}/{total_files}枚",
                                      overall_current=class_idx * 2 + 1, 
                                      overall_total=self.total_steps)
                
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        class_images.append(img)
                except Exception as e:
                    print(f"\n警告: {img_file} の読み込みエラー: {e}")
            
            images.extend(class_images)
            labels.extend([class_idx] * len(class_images))
            class_counts[class_name] = len(class_images)
            print(f"[{class_name}] 完了: {len(class_images)} 枚")
        
        self.print_progress(total_classes, total_classes, "データ読み込み", "完了",
                          overall_current=10, overall_total=self.total_steps)
        
        if not images:
            raise ValueError("画像が見つかりませんでした!")
        
        X = np.array(images)
        y = np.array(labels)
        
        print(f"\n合計画像数: {len(X)}")
        print(f"クラス分布: {class_counts}")
        
        return X, y, class_counts
    
    def build_model(self, model_type='EfficientNetB0'):
        """スパースモデリング対応のモデルを構築"""
        base_model_class = {
            'EfficientNetB0': EfficientNetB0,
            'EfficientNetB1': EfficientNetB1,
            'EfficientNetB2': EfficientNetB2,
        }[model_type]
        
        base_model = base_model_class(
            weights=None,
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # スパース正則化を追加
        for layer in base_model.layers:
            if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):
                layer.add_loss(lambda: self.sparse_regularization['l1_lambda'] * 
                              tf.reduce_sum(tf.abs(layer.kernel)))
                layer.add_loss(lambda: self.sparse_regularization['l2_lambda'] * 
                              tf.reduce_sum(tf.square(layer.kernel)))
        
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.sparse_regularization['dropout_rate'])(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.sparse_regularization['dropout_rate'])(x)
        predictions = layers.Dense(4, activation='softmax')(x)
        
        model = models.Model(inputs=base_model.input, outputs=predictions)
        
        optimizer = optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_with_sparse_modeling(self, X, y, class_counts):
        """スパースラーニングを実行"""
        print("\n" + "=" * 80)
        print("スパースラーニング開始")
        print("=" * 80)
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        y_train_cat = tf.keras.utils.to_categorical(y_train, 4)
        y_test_cat = tf.keras.utils.to_categorical(y_test, 4)
        
        # クラス重み計算
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # データ拡張
        datagen = ImageDataGenerator(**self.sparse_augmentation_params)
        train_gen = datagen.flow(X_train, y_train_cat, batch_size=self.batch_size)
        
        # モデル構築と訓練
        model = self.build_model('EfficientNetB0')
        
        # コールバック
        callbacks_list = [
            callbacks.ModelCheckpoint(
                'best_sparse_model_new_defects.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            ),
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                verbose=1
            )
        ]
        
        # カスタム進捗コールバック
        class ProgressCallback(callbacks.Callback):
            def __init__(self, trainer, total_epochs):
                self.trainer = trainer
                self.total_epochs = total_epochs
                self.epoch_start_time = None
            
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
            
            def on_epoch_end(self, epoch, logs=None):
                epoch_percent = ((epoch + 1) / float(self.total_epochs)) * 100
                epoch_elapsed = time.time() - self.epoch_start_time
                val_acc = logs.get('val_accuracy', 0) * 100
                train_acc = logs.get('accuracy', 0) * 100
                
                # 訓練は全体の70%と仮定（データ読み込み10% + 評価20%）
                overall_step = 10 + int(70 * (epoch + 1) / self.total_epochs)
                
                self.trainer.print_progress(
                    epoch + 1, self.total_epochs,
                    f"Epoch {epoch+1}/{self.total_epochs}",
                    f"訓練精度: {train_acc:.2f}% | 検証精度: {val_acc:.2f}%",
                    overall_current=overall_step,
                    overall_total=100
                )
        
        progress_callback = ProgressCallback(self, self.max_epochs)
        callbacks_list.append(progress_callback)
        
        # 訓練実行
        print("\n訓練開始...")
        history = model.fit(
            train_gen,
            epochs=self.max_epochs,
            validation_data=(X_test, y_test_cat),
            class_weight=class_weight_dict,
            callbacks=callbacks_list,
            verbose=0
        )
        
        self.models.append(model)
        self.histories.append(history)
        
        # 評価
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"\n最終精度: {test_acc*100:.2f}%")
        
        return model, test_acc
    
    def retrain_with_all_data(self, X, y, class_counts):
        """すべてのデータ（今までのデータを含む）で再学習"""
        print("\n" + "=" * 80)
        print("全データでの再学習開始")
        print("=" * 80)
        
        # データ分割（全データを使用）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        y_train_cat = tf.keras.utils.to_categorical(y_train, 4)
        y_test_cat = tf.keras.utils.to_categorical(y_test, 4)
        
        # クラス重み
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # データ拡張
        datagen = ImageDataGenerator(**self.sparse_augmentation_params)
        train_gen = datagen.flow(X_train, y_train_cat, batch_size=self.batch_size)
        
        # モデル構築
        model = self.build_model('EfficientNetB0')
        
        # コールバック
        callbacks_list = [
            callbacks.ModelCheckpoint(
                'best_sparse_model_all_data.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            ),
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                verbose=1
            )
        ]
        
        # 進捗コールバック
        class RetrainProgressCallback(callbacks.Callback):
            def __init__(self, trainer, total_epochs):
                self.trainer = trainer
                self.total_epochs = total_epochs
                self.epoch_start_time = None
                self.retrain_start_time = time.time()
            
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
            
            def on_epoch_end(self, epoch, logs=None):
                epoch_percent = ((epoch + 1) / float(self.total_epochs)) * 100
                epoch_elapsed = time.time() - self.epoch_start_time
                val_acc = logs.get('val_accuracy', 0) * 100
                train_acc = logs.get('accuracy', 0) * 100
                
                # 再学習は全体の80-100%の範囲
                overall_step = 80 + int(20 * (epoch + 1) / self.total_epochs)
                
                self.trainer.print_progress(
                    epoch + 1, self.total_epochs,
                    f"再学習 Epoch {epoch+1}/{self.total_epochs}",
                    f"訓練精度: {train_acc:.2f}% | 検証精度: {val_acc:.2f}%",
                    overall_current=overall_step,
                    overall_total=100
                )
        
        progress_callback = RetrainProgressCallback(self, self.max_epochs)
        callbacks_list.append(progress_callback)
        
        # 再訓練実行
        print("\n全データでの再訓練開始...")
        history = model.fit(
            train_gen,
            epochs=self.max_epochs,
            validation_data=(X_test, y_test_cat),
            class_weight=class_weight_dict,
            callbacks=callbacks_list,
            verbose=0
        )
        
        # 最終評価
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"\n再学習後の最終精度: {test_acc*100:.2f}%")
        
        total_time = time.time() - self.total_start_time
        print(f"\n全体の所要時間: {self.format_time(total_time)}")
        
        return model, test_acc

def main():
    """メイン関数"""
    print("=" * 80)
    print("不良品データ追加後のスパースラーニングと再学習")
    print("=" * 80)
    
    trainer = EnhancedProgressSparseTrainer()
    
    try:
        # ステップ1: すべてのデータを読み込み
        X, y, class_counts = trainer.load_all_data()
        
        # ステップ2: 新しい不良品データを含むスパースラーニング
        model1, acc1 = trainer.train_with_sparse_modeling(X, y, class_counts)
        print(f"\nステップ1完了: スパースラーニング精度 = {acc1*100:.2f}%")
        
        # ステップ3: 全データでの再学習
        model2, acc2 = trainer.retrain_with_all_data(X, y, class_counts)
        print(f"\nステップ2完了: 再学習精度 = {acc2*100:.2f}%")
        
        print("\n" + "=" * 80)
        print("すべての学習が完了しました！")
        print("=" * 80)
        print(f"スパースラーニングモデル: best_sparse_model_new_defects.h5")
        print(f"再学習モデル: best_sparse_model_all_data.h5")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


