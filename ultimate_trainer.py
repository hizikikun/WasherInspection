#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
樹脂製ワッシャー 究極版ディープラーニング学習システム
画像読み込み問題を根本的に解決し、高精度を実現
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import json
import glob
from pathlib import Path
import argparse
from collections import Counter
import pandas as pd
from PIL import Image
import math
import warnings
warnings.filterwarnings('ignore')

# 文字化け対策
plt.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib
matplotlib.use('Agg')

class UltimateResinWasherDataset:
    """究極版樹脂製ワッシャーデータセット管理クラス"""
    
    def __init__(self, data_dir="cs AI学習データ/樹脂"):
        self.data_dir = Path(data_dir)
        self.image_size = (224, 224)
        self.classes = ['良品', '欠け', '黒点', '傷']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        self.loaded_count = 0
        self.failed_count = 0
        
    def load_images_robust(self):
        """堅牢な画像読み込み"""
        print("=== 究極版データセット読み込み開始 ===")
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"警告: {class_name} ディレクトリが見つかりません: {class_dir}")
                continue
                
            print(f"\n{class_name} 処理中...")
            class_images = []
            class_labels = []
            
            # 複数の拡張子とパターンを試す
            extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']
            
            for ext in extensions:
                pattern = str(class_dir / "**" / ext)
                files = glob.glob(pattern, recursive=True)
                
                for file_path in files:
                    try:
                        # 複数の読み込み方法を試す
                        img = self._load_image_multiple_methods(file_path)
                        if img is not None:
                            # 前処理
                            img = self._preprocess_image(img)
                            class_images.append(img)
                            class_labels.append(self.class_to_idx[class_name])
                            self.loaded_count += 1
                        else:
                            self.failed_count += 1
                    except Exception as e:
                        print(f"エラー: {file_path} - {str(e)}")
                        self.failed_count += 1
                        continue
            
            print(f"  {class_name}: {len(class_images)}枚読み込み成功")
            self.images.extend(class_images)
            self.labels.extend(class_labels)
        
        print(f"\n=== 読み込み完了 ===")
        print(f"成功: {self.loaded_count}枚")
        print(f"失敗: {self.failed_count}枚")
        print(f"総画像数: {len(self.images)}")
        
        return np.array(self.images), np.array(self.labels)
    
    def _load_image_multiple_methods(self, file_path):
        """複数の方法で画像を読み込む"""
        methods = [
            self._load_with_cv2,
            self._load_with_pil,
            self._load_with_binary,
            self._load_with_unicode_path
        ]
        
        for method in methods:
            try:
                img = method(file_path)
                if img is not None and img.size > 0:
                    return img
            except:
                continue
        return None
    
    def _load_with_cv2(self, file_path):
        """OpenCVで読み込み"""
        img = cv2.imread(file_path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None
    
    def _load_with_pil(self, file_path):
        """PILで読み込み"""
        try:
            img = Image.open(file_path)
            return np.array(img.convert('RGB'))
        except:
            return None
    
    def _load_with_binary(self, file_path):
        """バイナリ読み込み"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            pass
        return None
    
    def _load_with_unicode_path(self, file_path):
        """Unicodeパス対応読み込み"""
        try:
            # 短いパス名に変換
            short_path = self._get_short_path_name(file_path)
            if short_path:
                img = cv2.imread(short_path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            pass
        return None
    
    def _get_short_path_name(self, long_name):
        """長いパス名を短いパス名に変換"""
        try:
            import win32api
            return win32api.GetShortPathName(long_name)
        except:
            return None
    
    def _preprocess_image(self, img):
        """画像前処理"""
        # リサイズ
        img = cv2.resize(img, self.image_size)
        
        # ノイズ除去
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        # 正規化
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def get_class_distribution(self):
        """クラス分布を取得"""
        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = {}
        for i, class_name in enumerate(self.classes):
            if i in unique:
                idx = np.where(unique == i)[0][0]
                distribution[class_name] = counts[idx]
            else:
                distribution[class_name] = 0
        return distribution

class UltimateModelBuilder:
    """究極版モデルビルダー"""
    
    def __init__(self, num_classes=4, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
    
    def build_transfer_learning_model(self):
        """Transfer Learningモデルを構築"""
        # MobileNetV2をベースに使用
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # ベースモデルを凍結
        base_model.trainable = False
        
        # カスタム分類ヘッド
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_custom_cnn_model(self):
        """カスタムCNNモデルを構築"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model

class UltimateDataAugmentation:
    """究極版データ拡張"""
    
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
    
    def create_augmentation_pipeline(self):
        """データ拡張パイプラインを作成"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # 高度なデータ拡張
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        return datagen
    
    def create_class_specific_augmentation(self, class_name):
        """クラス別データ拡張"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        if class_name == '黒点':
            # 黒点用の特別な拡張
            return ImageDataGenerator(
                rotation_range=45,
                width_shift_range=0.3,
                height_shift_range=0.3,
                shear_range=0.3,
                zoom_range=0.3,
                brightness_range=[0.7, 1.3],
                fill_mode='nearest'
            )
        elif class_name == '傷':
            # 傷用の特別な拡張
            return ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.15,
                height_shift_range=0.15,
                shear_range=0.15,
                zoom_range=0.15,
                brightness_range=[0.9, 1.1],
                fill_mode='nearest'
            )
        else:
            # 標準拡張
            return self.create_augmentation_pipeline()

class UltimateTrainer:
    """究極版トレーナー"""
    
    def __init__(self, data_dir="cs AI学習データ/樹脂"):
        self.data_dir = data_dir
        self.dataset = UltimateResinWasherDataset(data_dir)
        self.model_builder = UltimateModelBuilder()
        self.augmentation = UltimateDataAugmentation()
        
    def train(self, use_transfer_learning=True, epochs=100):
        """学習実行"""
        print("=== 究極版学習開始 ===")
        
        # データ読み込み
        X, y = self.dataset.load_images_robust()
        
        if len(X) == 0:
            print("エラー: 読み込めた画像が0枚です")
            return None
        
        print(f"読み込み成功: {len(X)}枚")
        
        # クラス分布表示
        distribution = self.dataset.get_class_distribution()
        print("\nクラス分布:")
        for class_name, count in distribution.items():
            print(f"  {class_name}: {count}枚")
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # クラス重み計算
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        print(f"\nクラス重み: {class_weight_dict}")
        
        # モデル構築
        if use_transfer_learning:
            model = self.model_builder.build_transfer_learning_model()
            print("Transfer Learningモデルを使用")
        else:
            model = self.model_builder.build_custom_cnn_model()
            print("カスタムCNNモデルを使用")
        
        # モデルコンパイル
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # コールバック設定
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'resin_washer_model/resin_washer_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # データ拡張
        datagen = self.augmentation.create_augmentation_pipeline()
        
        # 学習実行
        print("\n学習開始...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=epochs,
            validation_data=(X_test, y_test),
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # 評価
        print("\n=== 評価結果 ===")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"テスト精度: {test_accuracy:.4f}")
        print(f"テスト損失: {test_loss:.4f}")
        
        # 予測
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # 分類レポート
        print("\n分類レポート:")
        print(classification_report(
            y_test, y_pred_classes,
            target_names=self.dataset.classes
        ))
        
        # 混同行列
        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.dataset.classes,
                   yticklabels=self.dataset.classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 学習履歴プロット
        self._plot_training_history(history)
        
        print(f"\nモデル保存完了: resin_washer_model/resin_washer_model.h5")
        print("学習完了")
        
        return model, history
    
    def _plot_training_history(self, history):
        """学習履歴をプロット"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 精度
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # 損失
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='究極版樹脂製ワッシャー学習システム')
    parser.add_argument('--data_dir', default='cs AI学習データ/樹脂', help='データディレクトリ')
    parser.add_argument('--epochs', type=int, default=100, help='学習エポック数')
    parser.add_argument('--transfer', action='store_true', help='Transfer Learningを使用')
    parser.add_argument('--quick', action='store_true', help='クイック学習（10エポック）')
    
    args = parser.parse_args()
    
    # モデルディレクトリ作成
    os.makedirs('resin_washer_model', exist_ok=True)
    
    # 学習実行
    trainer = UltimateTrainer(args.data_dir)
    
    epochs = 10 if args.quick else args.epochs
    use_transfer = args.transfer or not args.quick
    
    model, history = trainer.train(
        use_transfer_learning=use_transfer,
        epochs=epochs
    )
    
    if model is not None:
        print("\n=== 学習成功 ===")
        print("モデルが正常に学習されました")
    else:
        print("\n=== 学習失敗 ===")
        print("学習に失敗しました")

if __name__ == "__main__":
    main()

