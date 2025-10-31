#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Optimization Trainer
高度な最適化手法による学習性能向上システム
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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from collections import Counter
import optuna
from optuna.integration import TFKerasPruningCallback

# Set UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

class AdvancedOptimizationTrainer:
    def __init__(self, data_path="cs_AItraining_data"):
        self.data_path = Path(data_path)
        self.class_names = ['good', 'black_spot', 'chipping', 'scratch']
        self.best_models = []
        self.optimization_history = []
        
        # 進捗管理
        self.status_path = Path('logs') / 'training_status.json'
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.total_start_time = None
        
        # 高度なデータ拡張設定
        self.advanced_augmentation_params = {
            # 幾何学的変換（強化版）
            'rotation_range': 45,
            'width_shift_range': 0.25,
            'height_shift_range': 0.25,
            'shear_range': 0.3,
            'zoom_range': 0.3,
            'horizontal_flip': True,
            'vertical_flip': True,
            
            # 色調テ明度変換（強化版）
            'brightness_range': [0.6, 1.4],
            'channel_shift_range': 0.3,
            
            # 高度な設定
            'fill_mode': 'nearest',
            'cval': 0.0,
            'data_format': 'channels_last'
        }
        
        # 混合精度学習の設定
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("混合精度学習を有効にしました")
        except Exception as e:
            print(f"混合精度設定エラー: {e}")
    
    def update_status(self, payload):
        """進捗状況をJSONで出力"""
        try:
            payload['timestamp'] = time.time()
            with open(self.status_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def load_and_prepare_data(self):
        """データの読み込みと前処理"""
        print("=" * 80)
        print("Advanced Optimization データ読み込み開始")
        print("=" * 80)
        
        self.total_start_time = time.time()
        self.update_status({
            'stage': 'データ読み込み',
            'section': 'データ読み込み',
            'overall_progress_percent': 0.0,
            'message': 'データ読み込み中...'
        })
        
        images = []
        labels = []
        class_counts = {}
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = self.data_path / 'resin' / class_name
            if not class_path.exists():
                print(f"警告: {class_path} が存在しません")
                continue
            
            print(f"\n[{class_name}] 画像を読み込み中...")
            class_images = []
            
            # 画像ファイルを再帰的に検索
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(class_path.rglob(ext))
            
            print(f"[{class_name}] {len(image_files)} 個の画像ファイルが見つかりました")
            
            for img_file in image_files:
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        class_images.append(img)
                except Exception as e:
                    print(f"エラー: {img_file} の読み込み中にエラー: {e}")
            
            images.extend(class_images)
            labels.extend([class_idx] * len(class_images))
            class_counts[class_name] = len(class_images)
            print(f"[{class_name}] 完了: {len(class_images)} 枚")
        
        if not images:
            raise ValueError("画像が見つかりませんでした!")
        
        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        
        print(f"\n合計画像数: {len(X)}")
        print(f"クラス分布: {class_counts}")
        
        # データ不均衡の分析
        print("\nデータ不均衡分析:")
        total_samples = len(X)
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {class_name}: {count}枚 ({percentage:.1f}%)")
        
        self.update_status({
            'stage': 'データ読み込み',
            'section': 'データ読み込み',
            'overall_progress_percent': 10.0,
            'message': 'データ読み込み完了'
        })
        
        return X, y, class_counts
    
    def create_optimized_data_generators(self, X, y):
        """最適化されたデータジェネレーターを作成"""
        print("\n" + "=" * 60)
        print("最適化されたデータジェネレーター作成")
        print("=" * 60)
        
        # データ分割
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"学習データ: {len(X_train)} 枚")
        print(f"検証データ: {len(X_val)} 枚")
        
        # 高度なクラス重みの計算
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"クラス重み: {class_weight_dict}")
        
        # 高度なデータ拡張ジェネレーター
        train_datagen = ImageDataGenerator(
            **self.advanced_augmentation_params,
            rescale=1./255,
        )
        
        val_datagen = ImageDataGenerator(
            rescale=1./255,
        )
        
        return X_train, X_val, y_train, y_val, class_weight_dict, train_datagen, val_datagen
    
    def build_optimized_model(self, trial=None):
        """最適化されたモデルを構築"""
        print("\n" + "=" * 60)
        print("最適化されたモデル構築")
        print("=" * 60)
        
        # Optunaによるハイパーパラメータ最適化
        if trial:
            # 学習率の最適化
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            
            # ドロップアウト率の最適化
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.6)
            
            # 正則化パラメータの最適化
            l1_reg = trial.suggest_float('l1_reg', 1e-6, 1e-3, log=True)
            l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
            
            # バッチサイズの最適化
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
            
            # モデル選択
            model_type = trial.suggest_categorical('model_type', ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2'])
        else:
            # デフォルト値
            learning_rate = 0.001
            dropout_rate = 0.3
            l1_reg = 1e-4
            l2_reg = 1e-4
            batch_size = 16
            model_type = 'EfficientNetB0'
        
        # モデル選択
        if model_type == 'EfficientNetB0':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif model_type == 'EfficientNetB1':
            base_model = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(240, 240, 3))
        else:  # EfficientNetB2
            base_model = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(260, 260, 3))
        
        # 段階的凍結戦略
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # 最適化された分類ヘッド
        model = models.Sequential([
            base_model,
            
            # 空間的注意機構
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='swish',
                       kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            # 特徴抽出層
            layers.Dense(512, activation='swish',
                       kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate * 0.8),
            
            # 最終分類層
            layers.Dense(256, activation='swish',
                       kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate * 0.6),
            
            layers.Dense(4, activation='softmax')
        ])
        
        # 最適化設定
        optimizer = optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model, batch_size
    
    def objective(self, trial, X_train, X_val, y_train, y_val, class_weight_dict):
        """Optuna最適化の目的関数"""
        model, batch_size = self.build_optimized_model(trial)
        
        # 学習実行
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,  # 最適化用は短縮
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=[
                TFKerasPruningCallback(trial, 'val_accuracy'),
                callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
            ],
            verbose=0
        )
        
        # 最高検証精度を返ム
        return max(history.history['val_accuracy'])
    
    def optimize_hyperparameters(self, X_train, X_val, y_train, y_val, class_weight_dict):
        """ハイパーパラメータ最適化"""
        print("\n" + "=" * 80)
        print("ハイパーパラメータ最適化開始")
        print("=" * 80)
        
        # Optunaスタディの作成
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # 最適化実行
        study.optimize(
            lambda trial: self.objective(trial, X_train, X_val, y_train, y_val, class_weight_dict),
            n_trials=20  # 試行回数
        )
        
        print(f"\n最適化完了!")
        print(f"最高精度: {study.best_value:.4f}")
        print(f"最適パラメータ: {study.best_params}")
        
        # 最適パラメータでモデルを再構築
        best_model, best_batch_size = self.build_optimized_model()
        
        # 最適パラメータを適用
        best_params = study.best_params
        best_model.compile(
            optimizer=optimizers.AdamW(
                learning_rate=best_params['learning_rate'],
                weight_decay=1e-4,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return best_model, best_params, best_batch_size
    
    def train_optimized_model(self, X_train, X_val, y_train, y_val, class_weight_dict, train_datagen, val_datagen):
        """最適化されたモデルで学習"""
        print("\n" + "=" * 80)
        print("最適化されたモデルで学習開始")
        print("=" * 80)
        
        # ハイパーパラメータ最適化
        best_model, best_params, best_batch_size = self.optimize_hyperparameters(
            X_train, X_val, y_train, y_val, class_weight_dict
        )
        
        # データジェネレーター
        train_gen = train_datagen.flow(X_train, y_train, batch_size=best_batch_size)
        val_gen = val_datagen.flow(X_val, y_val, batch_size=best_batch_size)
        
        # 高度なコールバック設定
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=25,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'advanced_optimization_best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            callbacks.CSVLogger('advanced_optimization_training_log.csv'),
            callbacks.LearningRateScheduler(
                lambda epoch: best_params['learning_rate'] * (0.95 ** (epoch // 5))
            )
        ]
        
        # 学習実行
        print(f"最適化された学習開始")
        print(f"学習サンプル: {len(X_train)}")
        print(f"検証サンプル: {len(X_val)}")
        print(f"最適バッチサイズ: {best_batch_size}")
        print(f"最適学習率: {best_params['learning_rate']:.6f}")
        
        history = best_model.fit(
            train_gen,
            steps_per_epoch=len(X_train) // best_batch_size,
            epochs=100,
            validation_data=val_gen,
            validation_steps=len(X_val) // best_batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.best_models.append(best_model)
        self.optimization_history.append(history)
        
        print(f"\n最適化された学習完了!")
        return history
    
    def evaluate_optimized_model(self, X_test, y_test):
        """最適化されたモデルの評価"""
        print("\n" + "=" * 80)
        print("最適化されたモデル評価")
        print("=" * 80)
        
        if not self.best_models:
            print("評価するモデルがありません")
            return 0.0
        
        model = self.best_models[0]  # 最良のモデル
        
        # 予測実行
        pred = model.predict(X_test, verbose=0)
        pred_classes = np.argmax(pred, axis=1)
        
        # 精度計算
        accuracy = np.mean(pred_classes == y_test)
        
        print(f"\n最適化されたモデル精度: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 詳細評価
        print("\n混同行列:")
        cm = confusion_matrix(y_test, pred_classes)
        print(cm)
        
        print("\n分類レポート:")
        print(classification_report(y_test, pred_classes, target_names=self.class_names))
        
        # 進捗更新
        self.update_status({
            'stage': '評価',
            'section': '評価',
            'overall_progress_percent': 100.0,
            'message': '完了',
            'final_accuracy': accuracy
        })
        
        return accuracy
    
    def save_optimized_model(self):
        """最適化されたモデルを保存"""
        print("\n" + "=" * 60)
        print("最適化されたモデル保存")
        print("=" * 60)
        
        if not self.best_models:
            print("保存するモデルがありません")
            return
        
        model = self.best_models[0]
        model.save('advanced_optimization_final_model.h5')
        print("保存完了: advanced_optimization_final_model.h5")
        
        # 最適化情報を保存
        optimization_info = {
            'model_name': 'Advanced Optimization 4-Class Washer Inspector',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_classes': 4,
            'class_names': self.class_names,
            'features': [
                'Hyperparameter Optimization (Optuna)',
                'Mixed Precision Training',
                'Advanced Data Augmentation',
                'Progressive Unfreezing',
                'AdamW Optimizer',
                'Swish Activation',
                'L1/L2 Regularization',
                'Learning Rate Scheduling'
            ],
            'description': 'Advanced optimization with hyperparameter tuning and state-of-the-art techniques'
        }
        
        with open('advanced_optimization_info.json', 'w', encoding='utf-8') as f:
            json.dump(optimization_info, f, indent=2, ensure_ascii=False)
        
        print("最適化情報を保存しました: advanced_optimization_info.json")

def main():
    """メイン実行関数"""
    print("=" * 80)
    print("Advanced Optimization Deep Learning Trainer")
    print("高度な最適化手法による学習システムを開始します")
    print("=" * 80)
    
    # トレーナー初期化
    trainer = AdvancedOptimizationTrainer()
    
    try:
        # データ読み込み
        print("\n[ステップ1] データ読み込み")
        X, y, class_counts = trainer.load_and_prepare_data()
        
        # データジェネレーター作成
        print("\n[ステップ2] データジェネレーター作成")
        X_train, X_val, y_train, y_val, class_weight_dict, train_datagen, val_datagen = trainer.create_optimized_data_generators(X, y)
        
        # 最適化された学習実行
        print("\n[ステップ3] 最適化された学習実行")
        history = trainer.train_optimized_model(X_train, X_val, y_train, y_val, class_weight_dict, train_datagen, val_datagen)
        
        # 評価実行
        print("\n[ステップ4] 性能評価")
        test_accuracy = trainer.evaluate_optimized_model(X_val, y_val)
        
        # モデル保存
        print("\n[ステップ5] モデル保存")
        trainer.save_optimized_model()
        
        # 結果表示
        total_time = time.time() - trainer.total_start_time
        print("\n" + "=" * 80)
        print("Advanced Optimization 学習完了!")
        print("=" * 80)
        print(f"最終精度: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"合計所要時間: {total_time/60:.1f}分")
        print("最適化されたモデルを保存しました!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
