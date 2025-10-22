#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
樹脂製ワッシャー 精度改善版ディープラーニング学習システム
既存の学習システムを基に精度を大幅改善
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

class ImprovedAccuracyDataset:
    """精度改善版データセット管理クラス"""
    
    def __init__(self, data_dir="cs AI学習データ/樹脂"):
        self.data_dir = Path(data_dir)
        self.image_size = (224, 224)
        self.classes = ['良品', '欠け', '黒点', '傷']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
    def load_images_improved(self):
        """改善された画像読み込み"""
        print("=== 精度改善版データセット読み込み開始 ===")
        
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
                        # PILで読み込み（より確実）
                        img = Image.open(file_path)
                        img = img.convert('RGB')
                        img = np.array(img)
                        
                        if img is not None and img.size > 0:
                            # 前処理
                            img = self._preprocess_image(img)
                            class_images.append(img)
                            class_labels.append(self.class_to_idx[class_name])
                            print(f"  読み込み成功: {os.path.basename(file_path)}")
                        else:
                            print(f"  読み込み失敗: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"  エラー: {os.path.basename(file_path)} - {str(e)}")
                        continue
            
            print(f"  {class_name}: {len(class_images)}枚読み込み成功")
            self.images.extend(class_images)
            self.labels.extend(class_labels)
        
        print(f"\n=== 読み込み完了 ===")
        print(f"総画像数: {len(self.images)}")
        
        return np.array(self.images), np.array(self.labels)
    
    def _preprocess_image(self, img):
        """画像前処理"""
        # リサイズ
        img = cv2.resize(img, self.image_size)
        
        # ノイズ除去
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        # 正規化
        img = img.astype(np.float32) / 255.0
        
        return img

class ImprovedAccuracyModel:
    """精度改善版モデルビルダー"""
    
    def __init__(self, num_classes=4, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
    
    def build_improved_model(self):
        """改善されたモデルを構築"""
        # MobileNetV2をベースに使用
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # ベースモデルを凍結
        base_model.trainable = False
        
        # 改善された分類ヘッド
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.6),  # 過学習防止を強化
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model

class ImprovedAccuracyTrainer:
    """精度改善版トレーナー"""
    
    def __init__(self, data_dir="cs AI学習データ/樹脂"):
        self.data_dir = data_dir
        self.dataset = ImprovedAccuracyDataset(data_dir)
        self.model_builder = ImprovedAccuracyModel()
        
    def train_improved(self, epochs=100):
        """改善された学習"""
        print("=== 精度改善版学習開始 ===")
        
        # データ読み込み
        X, y = self.dataset.load_images_improved()
        
        if len(X) == 0:
            print("エラー: 読み込めた画像が0枚です")
            return None
        
        print(f"読み込み成功: {len(X)}枚")
        
        # クラス分布表示
        unique, counts = np.unique(y, return_counts=True)
        print("\nクラス分布:")
        for i, class_name in enumerate(self.dataset.classes):
            if i in unique:
                idx = np.where(unique == i)[0][0]
                print(f"  {class_name}: {counts[idx]}枚")
            else:
                print(f"  {class_name}: 0枚")
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 強力なクラス重み計算
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        # 重みを大幅に強化
        class_weight_dict = {i: class_weights[i] * 5.0 for i in range(len(class_weights))}
        
        print(f"\nクラス重み: {class_weight_dict}")
        
        # モデル構築
        model = self.model_builder.build_improved_model()
        
        # モデルコンパイル（学習率を大幅に下げる）
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00005),  # 学習率を大幅に下げる
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # コールバック設定（より厳しい条件）
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,  # より強力な学習率削減
                patience=8,
                min_lr=1e-8
            ),
            keras.callbacks.ModelCheckpoint(
                'resin_washer_model/resin_washer_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # 学習実行
        print("\n学習開始...")
        history = model.fit(
            X_train, y_train,
            batch_size=8,  # バッチサイズを小さく
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
        plt.title('Confusion Matrix - Improved Accuracy')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_improved.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nモデル保存完了: resin_washer_model/resin_washer_model.h5")
        print("学習完了")
        
        return model, history

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='精度改善版樹脂製ワッシャー学習システム')
    parser.add_argument('--data_dir', default='cs AI学習データ/樹脂', help='データディレクトリ')
    parser.add_argument('--epochs', type=int, default=100, help='学習エポック数')
    
    args = parser.parse_args()
    
    # モデルディレクトリ作成
    os.makedirs('resin_washer_model', exist_ok=True)
    
    # 学習実行
    trainer = ImprovedAccuracyTrainer(args.data_dir)
    model, history = trainer.train_improved(epochs=args.epochs)
    
    if model is not None:
        print("\n=== 学習成功 ===")
        print("モデルが正常に学習されました")
    else:
        print("\n=== 学習失敗 ===")
        print("学習に失敗しました")

if __name__ == "__main__":
    main()
