#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
フィードバックデータから再学習するスクリプト
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

class FeedbackRetrainer:
    """フィードバックデータから再学習"""
    
    def __init__(self, feedback_dir="feedback_data", model_path="resin_washer_model/resin_washer_model.h5"):
        self.feedback_dir = Path(feedback_dir)
        self.model_path = model_path
        self.image_size = (224, 224)
        
        # クラス名の定義
        self.class_names = {
            0: 'good',
            1: 'chipped', 
            2: 'black_spot',
            3: 'scratched'
        }
        
        # 日本語クラス名
        self.japanese_names = {
            0: '良品',
            1: '欠け',
            2: '黒点',
            3: '傷'
        }
        
    def load_feedback_data(self):
        """フィードバックデータを読み込み"""
        print("=" * 60)
        print("フィードバックデータの読み込み")
        print("=" * 60)
        
        if not self.feedback_dir.exists():
            print("エラー: フィードバックデータディレクトリが見つかりません")
            return [], []
        
        # メタデータファイルを取得
        metadata_files = list(self.feedback_dir.glob("*_metadata.json"))
        
        if not metadata_files:
            print("警告: フィードバックデータが見つかりません")
            return [], []
        
        print(f"フィードバックデータ数: {len(metadata_files)}")
        
        images = []
        labels = []
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                image_path = Path(metadata['image_path'])
                if image_path.exists():
                    # 画像を読み込み
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        # リサイズ
                        image = cv2.resize(image, self.image_size)
                        images.append(image)
                        
                        # ラベルを取得（正解クラス）
                        correct_class = metadata['correct_class']
                        labels.append(correct_class)
                        
                        print(f"  読み込み: {image_path.name} -> {self.japanese_names[correct_class]}")
                
            except Exception as e:
                print(f"エラー: {metadata_file} の読み込みに失敗 - {e}")
        
        print(f"読み込み完了: {len(images)}枚")
        return images, labels
    
    def create_enhanced_dataset(self, images, labels):
        """強化されたデータセットを作成"""
        print("\n" + "=" * 60)
        print("強化されたデータセットの作成")
        print("=" * 60)
        
        if not images:
            print("エラー: 画像データがありません")
            return None, None
        
        # データ拡張の設定
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest'
        )
        
        # 画像とラベルをnumpy配列に変換
        X = np.array(images)
        y = np.array(labels)
        
        # カテゴリカルエンコーディング
        y_categorical = tf.keras.utils.to_categorical(y, num_classes=4)
        
        print(f"データセット作成完了: {X.shape}, {y_categorical.shape}")
        return X, y_categorical
    
    def create_improved_model(self):
        """改善されたモデルを作成"""
        print("\n" + "=" * 60)
        print("改善されたモデルの作成")
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
            keras.layers.Dense(4, activation='softmax')
        ])
        
        # コンパイル
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("改善されたモデル作成完了")
        return model
    
    def retrain_model(self, X, y):
        """モデルを再学習"""
        print("\n" + "=" * 60)
        print("モデルの再学習")
        print("=" * 60)
        
        if X is None or y is None:
            print("エラー: データがありません")
            return None
        
        # モデルを作成
        model = self.create_improved_model()
        
        # コールバック設定
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                'improved_model.h5',
                monitor='accuracy',
                save_best_only=True,
                mode='max'
            ),
            keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # 学習
        print("学習開始...")
        history = model.fit(
            X, y,
            epochs=50,
            batch_size=8,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        print("学習完了")
        return model, history
    
    def evaluate_model(self, model, X, y):
        """モデルを評価"""
        print("\n" + "=" * 60)
        print("モデル評価")
        print("=" * 60)
        
        if model is None:
            print("エラー: モデルがありません")
            return
        
        # 評価
        loss, accuracy = model.evaluate(X, y, verbose=0)
        print(f"損失: {loss:.4f}")
        print(f"精度: {accuracy:.4f}")
        
        # 予測
        predictions = model.predict(X, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        
        # クラス別精度
        for i, class_name in self.japanese_names.items():
            mask = true_classes == i
            if np.any(mask):
                class_accuracy = np.mean(predicted_classes[mask] == true_classes[mask])
                print(f"{class_name}: {class_accuracy:.4f}")
    
    def run_retraining(self):
        """再学習を実行"""
        print("=" * 60)
        print("フィードバックデータからの再学習")
        print("=" * 60)
        
        # フィードバックデータを読み込み
        images, labels = self.load_feedback_data()
        
        if not images:
            print("フィードバックデータがありません。インタラクティブ学習システムでデータを収集してください。")
            return
        
        # データセットを作成
        X, y = self.create_enhanced_dataset(images, labels)
        
        if X is None:
            return
        
        # モデルを再学習
        model, history = self.retrain_model(X, y)
        
        if model is not None:
            # モデルを評価
            self.evaluate_model(model, X, y)
            
            # モデルを保存
            model.save('improved_model.h5')
            print("\n改善されたモデルを保存: improved_model.h5")
            
            # 学習履歴を保存
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('training_history.png')
            plt.close()
            
            print("学習履歴を保存: training_history.png")

def main():
    """メイン関数"""
    retrainer = FeedbackRetrainer()
    retrainer.run_retraining()

if __name__ == "__main__":
    main()







