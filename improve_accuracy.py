#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精度改善のための包括的なスクリプト
"""

import os
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def create_enhanced_data_augmentation():
    """強化されたデータ拡張の設定"""
    print("=" * 60)
    print("強化されたデータ拡張の設定")
    print("=" * 60)
    
    # 黒点データに特化した拡張
    black_spot_augmentation = ImageDataGenerator(
        rotation_range=30,  # 回転範囲を拡大
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],  # 明度調整を強化
        channel_shift_range=20,        # 色相調整
        fill_mode='nearest'
    )
    
    # 一般的な拡張
    general_augmentation = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )
    
    print("データ拡張設定完了")
    return black_spot_augmentation, general_augmentation

def create_improved_model():
    """改善されたモデル構造"""
    print("\n" + "=" * 60)
    print("改善されたモデル構造の作成")
    print("=" * 60)
    
    model = keras.Sequential([
        # 入力層
        keras.layers.Input(shape=(224, 224, 3)),
        
        # 第1ブロック
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # 第2ブロック
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # 第3ブロック
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # 第4ブロック
        keras.layers.Conv2D(256, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(256, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # 全結合層
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(4, activation='softmax')  # 4クラス
    ])
    
    # カスタム損失関数（クラス重み付き）
    def weighted_categorical_crossentropy(y_true, y_pred):
        # クラス重み: 良品=1, 欠け=2, 黒点=3, 傷=2
        class_weights = tf.constant([1.0, 2.0, 3.0, 2.0])
        y_true_float = tf.cast(y_true, tf.float32)
        weights = tf.reduce_sum(class_weights * y_true_float, axis=1)
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return loss * weights
    
    # コンパイル
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=weighted_categorical_crossentropy,
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    print("改善されたモデル構造:")
    model.summary()
    
    return model

def create_advanced_training():
    """高度な学習戦略"""
    print("\n" + "=" * 60)
    print("高度な学習戦略")
    print("=" * 60)
    
    # コールバック設定
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'improved_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.CSVLogger('training_log.csv')
    ]
    
    print("コールバック設定完了")
    return callbacks

def analyze_current_performance():
    """現在の性能分析"""
    print("\n" + "=" * 60)
    print("現在の性能分析")
    print("=" * 60)
    
    print("問題点:")
    print("1. データ不均衡: 良品100枚 vs 黒点19枚")
    print("2. 特徴学習不足: 黒点と傷の区別が困難")
    print("3. 信頼度が低い: 0.25は不確実")
    
    print("\n改善策:")
    print("1. データ拡張で黒点データを3倍に増加")
    print("2. クラス重みで黒点を重視")
    print("3. より深いネットワーク構造")
    print("4. 転移学習の活用")
    
    print("\n推奨アクション:")
    print("1. 黒点データの追加撮影（30枚以上）")
    print("2. モデルの再学習")
    print("3. ハイパーパラメータの調整")

def create_retraining_script():
    """再学習スクリプトの作成"""
    print("\n" + "=" * 60)
    print("再学習スクリプトの作成")
    print("=" * 60)
    
    script_content = '''
# 精度改善のための再学習スクリプト
python resin_washer_trainer.py --train --improve-accuracy

# 改善されたモデルでテスト
python realtime_inspection.py --model improved_model.h5 --camera 0
'''
    
    with open('retrain_improved.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("再学習スクリプト作成完了")

if __name__ == "__main__":
    create_enhanced_data_augmentation()
    create_improved_model()
    create_advanced_training()
    analyze_current_performance()
    create_retraining_script()
    
    print("\n" + "=" * 60)
    print("精度改善のための次のステップ")
    print("=" * 60)
    print("1. 黒点データの追加撮影")
    print("2. 改善されたモデルで再学習")
    print("3. ハイパーパラメータの最適化")
    print("4. 性能評価と調整")
