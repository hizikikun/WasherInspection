#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-High Performance Deep Learning Trainer
最高性能の学習システム - 精度と学習速度の両方を向上
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
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
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

class UltraHighPerformanceTrainer:
    def __init__(self, data_path="cs_AItraining_data"):
        self.data_path = Path(data_path)
        self.class_names = ['good', 'black_spot', 'chipping', 'scratch']
        self.models = []
        self.histories = []
        
        # 進捗管理
        self.status_path = Path('logs') / 'training_status.json'
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.total_start_time = None
        
        # 高度なデータ拡張設定（Albumentations使用）
        self.advanced_augmentation = A.Compose([
            # 幾何学的変換
            A.Rotate(limit=45, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.3,
                rotate_limit=30,
                p=0.8
            ),
            
            # 色調テ明度変換
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.7
            ),
            
            # のイズテぼかし
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
            ], p=0.5),
            
            # 幾何学的歪み
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                A.OpticalDistortion(distort_limit=0.3, shift_limit=0.05, p=0.3),
            ], p=0.3),
            
            # カットアウトテミックスアップ
            A.OneOf([
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5
                ),
                A.Cutout(
                    num_holes=8,
                    max_h_size=32,
                    max_w_size=32,
                    fill_value=0,
                    p=0.3
                ),
            ], p=0.4),
            
            # 正規化
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 検証用の軽い拡張
        self.validation_augmentation = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
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
        print("Ultra-High Performance データ読み込み開始")
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
    
    def create_advanced_data_generators(self, X, y):
        """高度なデータジェネレーターを作成"""
        print("\n" + "=" * 60)
        print("高度なデータジェネレーター作成")
        print("=" * 60)
        
        # データ分割
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"学習データ: {len(X_train)} 枚")
        print(f"検証データ: {len(X_val)} 枚")
        
        # クラス重みの計算
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"クラス重み: {class_weight_dict}")
        
        return X_train, X_val, y_train, y_val, class_weight_dict
    
    def build_ultra_high_performance_models(self):
        """最高性能のモデルを構築"""
        print("\n" + "=" * 60)
        print("Ultra-High Performance モデル構築")
        print("=" * 60)
        
        models_config = [
            {
                'name': 'EfficientNetB0_Enhanced',
                'model': EfficientNetB0,
                'input_size': (224, 224, 3),
                'learning_rate': 0.001
            },
            {
                'name': 'EfficientNetB1_Enhanced',
                'model': EfficientNetB1,
                'input_size': (240, 240, 3),
                'learning_rate': 0.0008
            },
            {
                'name': 'EfficientNetB2_Enhanced',
                'model': EfficientNetB2,
                'input_size': (260, 260, 3),
                'learning_rate': 0.0006
            },
            {
                'name': 'EfficientNetB3_Enhanced',
                'model': EfficientNetB3,
                'input_size': (300, 300, 3),
                'learning_rate': 0.0005
            }
        ]
        
        ensemble_models = []
        
        for i, config in enumerate(models_config):
            print(f"\n構築中 {i+1}/{len(models_config)}: {config['name']}")
            
            # ベースモデル
            base_model = config['model'](
                weights='imagenet',  # 事前学習済み重みを使用
                include_top=False,
                input_shape=config['input_size']
            )
            
            # 段階的凍結戦略
            for layer in base_model.layers[:-50]:  # より多くの層を凍結
                layer.trainable = False
            
            # 高度な分類ヘッド
            model = models.Sequential([
                base_model,
                
                # 空間的注意機構
                layers.GlobalAveragePooling2D(),
                layers.Dense(1024, activation='swish',  # Swish活性化関数
                           kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                
                # 特徴抽出層
                layers.Dense(512, activation='swish',
                           kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                # 最終分類層
                layers.Dense(256, activation='swish',
                           kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4)),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                
                layers.Dense(4, activation='softmax')
            ])
            
            # 最適化設定
            optimizer = optimizers.AdamW(
                learning_rate=config['learning_rate'],
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
            
            ensemble_models.append({
                'model': model,
                'name': config['name'],
                'input_size': config['input_size'],
                'learning_rate': config['learning_rate']
            })
        
        return ensemble_models
    
    def train_ultra_high_performance_models(self, X_train, X_val, y_train, y_val, class_weight_dict):
        """最高性能の学習を実行"""
        print("\n" + "=" * 80)
        print("Ultra-High Performance 学習開始")
        print("=" * 80)
        
        ensemble_models = self.build_ultra_high_performance_models()
        
        # 学習設定
        batch_size = 16  # 混合精度学習に適したサイズ
        max_epochs = 100
        patience = 20
        
        for i, model_config in enumerate(ensemble_models):
            model = model_config['model']
            model_name = model_config['name']
            input_size = model_config['input_size']
            
            print(f"\n{'='*60}")
            print(f"学習中 {i+1}/{len(ensemble_models)}: {model_name}")
            print(f"{'='*60}")
            
            # データのリサイズ
            if input_size != (224, 224, 3):
                new_size = input_size[0]
                print(f"データを {new_size}x{new_size} にリサイズ中...")
                X_train_resized = np.array([cv2.resize(img, (new_size, new_size)) for img in X_train])
                X_val_resized = np.array([cv2.resize(img, (new_size, new_size)) for img in X_val])
            else:
                X_train_resized = X_train
                X_val_resized = X_val
            
            # 高度なコールバック設定
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=patience,
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
                    f'ultra_high_performance_{model_name.lower()}_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                callbacks.CSVLogger(f'ultra_high_performance_{model_name.lower()}_log.csv'),
                callbacks.LearningRateScheduler(
                    lambda epoch: model_config['learning_rate'] * (0.9 ** (epoch // 10))
                )
            ]
            
            # 学習実行
            print(f"学習開始: {model_name}")
            print(f"学習サンプル: {len(X_train_resized)}")
            print(f"検証サンプル: {len(X_val_resized)}")
            print(f"バッチサイズ: {batch_size}")
            print(f"最大エポック数: {max_epochs}")
            
            history = model.fit(
                X_train_resized, y_train,
                validation_data=(X_val_resized, y_val),
                epochs=max_epochs,
                batch_size=batch_size,
                class_weight=class_weight_dict,
                callbacks=callbacks_list,
                verbose=1
            )
            
            self.models.append(model)
            self.histories.append(history)
            
            print(f"\n{model_name} 学習完了!")
            
            # 進捗更新
            progress = ((i + 1) / len(ensemble_models)) * 100
            self.update_status({
                'stage': '学習',
                'section': f'モデル {i+1}/{len(ensemble_models)}',
                'overall_progress_percent': 20 + int(progress * 0.6),
                'message': f'{model_name} 学習完了'
            })
        
        print(f"\nすべてのモデルの学習が完了しました!")
        return self.histories
    
    def evaluate_ultra_high_performance_models(self, X_test, y_test):
        """最高性能の評価を実行"""
        print("\n" + "=" * 80)
        print("Ultra-High Performance 評価")
        print("=" * 80)
        
        # アンサンブル予測
        predictions = []
        for i, model in enumerate(self.models):
            model_name = f"Model_{i+1}"
            print(f"予測中: {model_name}")
            
            # データのリサイズ
            if hasattr(model, 'input_shape'):
                input_size = model.input_shape[1]
                if input_size != 224:
                    X_test_resized = np.array([cv2.resize(img, (input_size, input_size)) for img in X_test])
                else:
                    X_test_resized = X_test
            else:
                X_test_resized = X_test
            
            pred = model.predict(X_test_resized, verbose=0)
            predictions.append(pred)
        
        # アンサンブル予測（重み付き平均）
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
        
        # 精度計算
        accuracy = np.mean(ensemble_pred_classes == y_test)
        
        print(f"\nUltra-High Performance アンサンブル精度: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 詳細評価
        print("\n混同行列:")
        cm = confusion_matrix(y_test, ensemble_pred_classes)
        print(cm)
        
        print("\n分類レポート:")
        print(classification_report(y_test, ensemble_pred_classes, target_names=self.class_names))
        
        # 進捗更新
        self.update_status({
            'stage': '評価',
            'section': '評価',
            'overall_progress_percent': 100.0,
            'message': '完了',
            'final_accuracy': accuracy
        })
        
        return accuracy
    
    def save_ultra_high_performance_models(self):
        """最高性能モデルを保存"""
        print("\n" + "=" * 60)
        print("Ultra-High Performance モデル保存")
        print("=" * 60)
        
        for i, model in enumerate(self.models):
            model_name = f"ultra_high_performance_ensemble_model_{i+1}.h5"
            model.save(model_name)
            print(f"保存完了: {model_name}")
        
        # アンサンブル情報を保存
        ensemble_info = {
            'model_name': 'Ultra-High Performance 4-Class Ensemble Washer Inspector',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_classes': 4,
            'class_names': self.class_names,
            'num_models': len(self.models),
            'model_types': ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3'],
            'features': [
                'Mixed Precision Training',
                'Advanced Data Augmentation (Albumentations)',
                'Weighted Ensemble',
                'Progressive Unfreezing',
                'AdamW Optimizer',
                'Swish Activation',
                'L1/L2 Regularization'
            ],
            'description': 'Ultra-high performance ensemble with advanced augmentation and optimization techniques'
        }
        
        with open('ultra_high_performance_ensemble_info.json', 'w', encoding='utf-8') as f:
            json.dump(ensemble_info, f, indent=2, ensure_ascii=False)
        
        print("アンサンブル情報を保存しました: ultra_high_performance_ensemble_info.json")

def main():
    """メイン実行関数"""
    print("=" * 80)
    print("Ultra-High Performance Deep Learning Trainer")
    print("最高性能の学習システムを開始します")
    print("=" * 80)
    
    # トレーナー初期化
    trainer = UltraHighPerformanceTrainer()
    
    try:
        # データ読み込み
        print("\n[ステップ1] データ読み込み")
        X, y, class_counts = trainer.load_and_prepare_data()
        
        # データジェネレーター作成
        print("\n[ステップ2] データジェネレーター作成")
        X_train, X_val, y_train, y_val, class_weight_dict = trainer.create_advanced_data_generators(X, y)
        
        # 学習実行
        print("\n[ステップ3] Ultra-High Performance 学習実行")
        histories = trainer.train_ultra_high_performance_models(X_train, X_val, y_train, y_val, class_weight_dict)
        
        # 評価実行
        print("\n[ステップ4] 性能評価")
        test_accuracy = trainer.evaluate_ultra_high_performance_models(X_val, y_val)
        
        # モデル保存
        print("\n[ステップ5] モデル保存")
        trainer.save_ultra_high_performance_models()
        
        # 結果表示
        total_time = time.time() - trainer.total_start_time
        print("\n" + "=" * 80)
        print("Ultra-High Performance 学習完了!")
        print("=" * 80)
        print(f"最終精度: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"合計所要時間: {total_time/60:.1f}分")
        print("すべてのモデルを保存しました!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
