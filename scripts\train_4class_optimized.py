#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精度向上版: 4-Class SPARSE MODELING with Optimized Settings
- データ拡張の最適化
- 正則化の調整
- 事前学習済み重みの使用
"""

import os
import sys
import subprocess
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import json
import time

# UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# GPU設定
if os.name == 'nt' and 'TF_USE_DIRECTML' not in os.environ:
    os.environ['TF_USE_DIRECTML'] = '1'

class OptimizedSparseFourClassTrainer:
    def __init__(self, data_path="cs_AItraining_data"):
        self.data_path = Path(data_path)
        self.class_names = ['good', 'black_spot', 'chipping', 'scratch']
        self.models = []
        self.histories = []
        
        # 最適化されたデータ拡張パラメータ
        self.optimized_augmentation = {
            'rotation_range': 15,           # 30→15（弱める）
            'width_shift_range': 0.1,      # 0.2→0.1（弱める）
            'height_shift_range': 0.1,
            'shear_range': 0.1,            # 0.2→0.1（弱める）
            'zoom_range': 0.1,             # 0.2→0.1（弱める）
            'horizontal_flip': True,
            'vertical_flip': False,        # 垂直反転は無効化
            'brightness_range': [0.9, 1.1],  # [0.8, 1.2]→[0.9, 1.1]（弱める）
            'channel_shift_range': 0.05,   # 0.1→0.05（弱める）
            'fill_mode': 'nearest',
            'cval': 0.0,
        }
        
        # 最適化された正則化パラメータ
        self.optimized_regularization = {
            'l1_lambda': 0.0005,    # 0.001→0.0005（緩める）
            'l2_lambda': 0.00005,   # 0.0001→0.00005（緩める）
            'dropout_rate': 0.3,    # 0.5→0.3（緩める）
        }
        
        # 事前学習済み重みを使用
        self.use_pretrained_weights = True  # ImageNetの重みを使用
        
        print("=" * 60)
        print("最適化された設定で学習")
        print("=" * 60)
        print(f"データ拡張: 中程度（弱め）")
        print(f"Dropout: {self.optimized_regularization['dropout_rate']}")
        print(f"事前学習済み重み: {'使用' if self.use_pretrained_weights else '不使用'}")
        print("=" * 60)
    
    def load_data(self):
        """データを読み込む"""
        print("\nデータ読み込み中...")
        images = []
        labels = []
        class_counts = {}
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = self.data_path / 'resin' / class_name
            if not class_path.exists():
                print(f"警告: {class_path} が見つかりません")
                continue
            
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(class_path.rglob(ext))
            
            print(f"[{class_name}] {len(image_files)}枚の画像を発見")
            
            for img_file in image_files:
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        images.append(img)
                        labels.append(class_idx)
                except Exception as e:
                    print(f"エラー: {img_file} の読み込み失敗: {e}")
            
            class_counts[class_name] = labels.count(class_idx)
        
        X = np.array(images)
        y = np.array(labels)
        
        print(f"\n総画像数: {len(X)}")
        print(f"クラス分布: {class_counts}")
        
        return X, y, class_counts
    
    def create_data_generators(self, X, y):
        """データジェネレータを作成"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 訓練データジェネレータ（最適化された拡張）
        train_datagen = ImageDataGenerator(
            **self.optimized_augmentation,
            rescale=1./255,
        )
        
        # バリデーションデータジェネレータ（拡張なし）
        val_datagen = ImageDataGenerator(
            rescale=1./255,
        )
        
        return train_datagen, val_datagen, X_train, X_val, y_train, y_val
    
    def build_model(self, model_type, input_size=(224, 224, 3)):
        """モデルを構築（事前学習済み重みを使用）"""
        if model_type == 'EfficientNetB0':
            base_model = EfficientNetB0(
                weights='imagenet' if self.use_pretrained_weights else None,
                include_top=False,
                input_shape=input_size
            )
        elif model_type == 'EfficientNetB1':
            base_model = EfficientNetB1(
                weights='imagenet' if self.use_pretrained_weights else None,
                include_top=False,
                input_shape=(240, 240, 3)
            )
        elif model_type == 'EfficientNetB2':
            base_model = EfficientNetB2(
                weights='imagenet' if self.use_pretrained_weights else None,
                include_top=False,
                input_shape=(260, 260, 3)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 転移学習: 最後の50層を学習可能に
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        
        # ヘッドを追加
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l1_l2(
                            l1=self.optimized_regularization['l1_lambda'],
                            l2=self.optimized_regularization['l2_lambda']
                        ))(x)
        x = layers.Dropout(self.optimized_regularization['dropout_rate'])(x)
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l1_l2(
                            l1=self.optimized_regularization['l1_lambda'],
                            l2=self.optimized_regularization['l2_lambda']
                        ))(x)
        x = layers.Dropout(self.optimized_regularization['dropout_rate'])(x)
        predictions = layers.Dense(4, activation='softmax')(x)
        
        model = models.Model(inputs=base_model.input, outputs=predictions)
        
        # 最適化された学習率スケジューラ
        lr_schedule = optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=0.001,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=1.0,
            alpha=0.0001
        )
        
        optimizer = optimizers.Adam(learning_rate=lr_schedule)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X, y):
        """学習を実行"""
        print("\n" + "=" * 60)
        print("最適化された学習開始")
        print("=" * 60)
        
        # データ分割
        train_datagen, val_datagen, X_train, X_val, y_train, y_val = self.create_data_generators(X, y)
        
        # クラス重みの計算（クラス不均衡対策）
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"\nクラス重み: {class_weight_dict}")
        
        # ワンホットエンコーディング
        from tensorflow.keras.utils import to_categorical
        y_train_cat = to_categorical(y_train, 4)
        y_val_cat = to_categorical(y_val, 4)
        
        # モデル構築と学習
        model_configs = [
            {'type': 'EfficientNetB0', 'input_size': (224, 224, 3)},
            {'type': 'EfficientNetB1', 'input_size': (240, 240, 3)},
            {'type': 'EfficientNetB2', 'input_size': (260, 260, 3)},
        ]
        
        for config in model_configs:
            print(f"\n{config['type']} を学習中...")
            model = self.build_model(config['type'], config['input_size'])
            
            # データジェネレータ
            if config['type'] == 'EfficientNetB0':
                train_gen = train_datagen.flow(X_train, y_train_cat, batch_size=32)
                val_gen = val_datagen.flow(X_val, y_val_cat, batch_size=32)
            elif config['type'] == 'EfficientNetB1':
                X_train_resized = np.array([cv2.resize(img, (240, 240)) for img in X_train])
                X_val_resized = np.array([cv2.resize(img, (240, 240)) for img in X_val])
                train_gen = train_datagen.flow(X_train_resized, y_train_cat, batch_size=32)
                val_gen = val_datagen.flow(X_val_resized, y_val_cat, batch_size=32)
            else:  # B2
                X_train_resized = np.array([cv2.resize(img, (260, 260)) for img in X_train])
                X_val_resized = np.array([cv2.resize(img, (260, 260)) for img in X_val])
                train_gen = train_datagen.flow(X_train_resized, y_train_cat, batch_size=32)
                val_gen = val_datagen.flow(X_val_resized, y_val_cat, batch_size=32)
            
            # コールバック
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=50,
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ModelCheckpoint(
                    f'optimized_best_{config["type"].lower()}_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
            ]
            
            # 学習
            history = model.fit(
                train_gen,
                steps_per_epoch=len(X_train) // 32,
                epochs=200,
                validation_data=val_gen,
                validation_steps=len(X_val) // 32,
                class_weight=class_weight_dict,
                callbacks=callbacks_list,
                verbose=1
            )
            
            self.models.append(model)
            self.histories.append(history)
            
            print(f"\n{config['type']} 学習完了")
            print(f"  最終訓練精度: {history.history['accuracy'][-1]:.4f}")
            print(f"  最終バリデーション精度: {history.history['val_accuracy'][-1]:.4f}")
        
        return self.models, self.histories
    
    def evaluate_ensemble(self, X_test, y_test):
        """アンサンブル評価"""
        print("\n" + "=" * 60)
        print("アンサンブル評価")
        print("=" * 60)
        
        predictions = []
        for i, model in enumerate(self.models):
            model_type = ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2'][i]
            
            # 入力サイズに合わせてリサイズ
            if model_type == 'EfficientNetB0':
                X_test_resized = X_test
            elif model_type == 'EfficientNetB1':
                X_test_resized = np.array([cv2.resize(img, (240, 240)) for img in X_test])
            else:  # B2
                X_test_resized = np.array([cv2.resize(img, (260, 260)) for img in X_test])
            
            pred = model.predict(X_test_resized, verbose=0)
            predictions.append(pred)
            
            # 個別精度
            pred_classes = np.argmax(pred, axis=1)
            acc = np.mean(pred_classes == y_test)
            print(f"{model_type} テスト精度: {acc:.4f} ({acc*100:.2f}%)")
        
        # アンサンブル予測（平均）
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
        ensemble_acc = np.mean(ensemble_pred_classes == y_test)
        
        print(f"\nアンサンブル精度: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
        
        return ensemble_acc
    
    def save_models(self):
        """モデルを保存"""
        print("\nモデルを保存中...")
        for i, model in enumerate(self.models):
            model_type = ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2'][i]
            filename = f'optimized_ensemble_{model_type.lower()}_model.h5'
            model.save(filename)
            print(f"保存: {filename}")

def main():
    print("=" * 80)
    print("精度向上版: 4-Class SPARSE MODELING (最適化設定)")
    print("=" * 80)
    
    trainer = OptimizedSparseFourClassTrainer()
    
    try:
        # データ読み込み
        X, y, class_counts = trainer.load_data()
        
        # 学習
        models, histories = trainer.train(X, y)
        
        # 評価
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        test_acc = trainer.evaluate_ensemble(X_test, y_test)
        
        # 保存
        trainer.save_models()
        
        print("\n" + "=" * 80)
        print("学習完了")
        print("=" * 80)
        print(f"最終テスト精度: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()



















