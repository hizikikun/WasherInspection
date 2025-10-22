#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
樹脂製ワッシャー 精度修正版ディープラーニング学習システム
クラス不均衡と過学習を根本的に解決
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

class PrecisionFixDataset:
    """精度修正版データセット管理クラス"""
    
    def __init__(self, data_dir="cs AI学習データ/樹脂"):
        self.data_dir = Path(data_dir)
        self.image_size = (224, 224)
        self.classes = ['良品', '欠け', '黒点', '傷']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
    def load_images_with_balance(self):
        """バランスを考慮した画像読み込み"""
        print("=== 精度修正版データセット読み込み開始 ===")
        
        # 各クラスの画像を読み込み
        class_data = {}
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"警告: {class_name} ディレクトリが見つかりません")
                class_data[class_name] = []
                continue
                
            print(f"\n{class_name} 処理中...")
            class_images = []
            
            # 複数の拡張子を試す
            extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']
            
            for ext in extensions:
                pattern = str(class_dir / "**" / ext)
                files = glob.glob(pattern, recursive=True)
                
                for file_path in files:
                    try:
                        img = self._load_image_robust(file_path)
                        if img is not None:
                            img = self._preprocess_image(img)
                            class_images.append(img)
                    except Exception as e:
                        continue
            
            print(f"  {class_name}: {len(class_images)}枚読み込み成功")
            class_data[class_name] = class_images
        
        # データバランス調整
        self._balance_data(class_data)
        
        print(f"\n=== 読み込み完了 ===")
        print(f"総画像数: {len(self.images)}")
        
        return np.array(self.images), np.array(self.labels)
    
    def _load_image_robust(self, file_path):
        """堅牢な画像読み込み"""
        try:
            # PILで読み込み
            img = Image.open(file_path)
            img = img.convert('RGB')
            return np.array(img)
        except:
            try:
                # OpenCVで読み込み
                img = cv2.imread(file_path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                pass
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
    
    def _balance_data(self, class_data):
        """データバランス調整"""
        # 各クラスの最小枚数を取得
        min_count = min(len(images) for images in class_data.values() if len(images) > 0)
        target_count = max(min_count, 50)  # 最低50枚
        
        print(f"\n=== データバランス調整 ===")
        print(f"目標枚数: {target_count}枚/クラス")
        
        for class_name, images in class_data.items():
            if len(images) == 0:
                print(f"  {class_name}: 0枚 (スキップ)")
                continue
            
            # データ拡張で目標枚数まで増やす
            if len(images) < target_count:
                augmented_images = self._augment_class_data(images, target_count - len(images))
                images.extend(augmented_images)
            
            # ランダムサンプリングで目標枚数に調整
            if len(images) > target_count:
                indices = np.random.choice(len(images), target_count, replace=False)
                images = [images[i] for i in indices]
            
            # ラベルを追加
            class_idx = self.class_to_idx[class_name]
            labels = [class_idx] * len(images)
            
            self.images.extend(images)
            self.labels.extend(labels)
            
            print(f"  {class_name}: {len(images)}枚 (目標達成)")
    
    def _augment_class_data(self, images, target_count):
        """クラス別データ拡張"""
        augmented = []
        
        for _ in range(target_count):
            # ランダムに画像を選択
            img = images[np.random.randint(0, len(images))]
            
            # データ拡張を適用
            augmented_img = self._apply_augmentation(img)
            augmented.append(augmented_img)
        
        return augmented
    
    def _apply_augmentation(self, img):
        """データ拡張を適用"""
        # ランダム回転
        angle = np.random.uniform(-30, 30)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
        
        # ランダムシフト
        shift_x = np.random.uniform(-0.1, 0.1) * w
        shift_y = np.random.uniform(-0.1, 0.1) * h
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (w, h))
        
        # ランダムズーム
        zoom = np.random.uniform(0.9, 1.1)
        h, w = img.shape[:2]
        new_h, new_w = int(h * zoom), int(w * zoom)
        img = cv2.resize(img, (new_w, new_h))
        
        # 中央クロップ
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        img = img[start_y:start_y+h, start_x:start_x+w]
        
        # ランダムフリップ
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        # 明度調整
        brightness = np.random.uniform(0.8, 1.2)
        img = np.clip(img * brightness, 0, 1)
        
        return img

class PrecisionFixModel:
    """精度修正版モデルビルダー"""
    
    def __init__(self, num_classes=4, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
    
    def build_balanced_model(self):
        """バランス調整されたモデルを構築"""
        # MobileNetV2をベースに使用
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # ベースモデルを凍結
        base_model.trainable = False
        
        # バランス調整された分類ヘッド
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),  # 過学習防止を強化
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

class PrecisionFixTrainer:
    """精度修正版トレーナー"""
    
    def __init__(self, data_dir="cs AI学習データ/樹脂"):
        self.data_dir = data_dir
        self.dataset = PrecisionFixDataset(data_dir)
        self.model_builder = PrecisionFixModel()
        
    def train_with_balance(self, epochs=100):
        """バランス調整された学習"""
        print("=== 精度修正版学習開始 ===")
        
        # データ読み込み
        X, y = self.dataset.load_images_with_balance()
        
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
        
        # クラス重み計算（より強力な重み付け）
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] * 2.0 for i in range(len(class_weights))}  # 重みを2倍に
        
        print(f"\nクラス重み: {class_weight_dict}")
        
        # モデル構築
        model = self.model_builder.build_balanced_model()
        
        # モデルコンパイル（学習率を下げる）
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # 学習率を下げる
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # コールバック設定（より厳しい条件）
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,  # より早く停止
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,  # より強力な学習率削減
                patience=5,
                min_lr=1e-7
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
            batch_size=16,  # バッチサイズを小さく
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
        plt.title('Confusion Matrix - Precision Fix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_precision_fix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nモデル保存完了: resin_washer_model/resin_washer_model.h5")
        print("学習完了")
        
        return model, history

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='精度修正版樹脂製ワッシャー学習システム')
    parser.add_argument('--data_dir', default='cs AI学習データ/樹脂', help='データディレクトリ')
    parser.add_argument('--epochs', type=int, default=100, help='学習エポック数')
    
    args = parser.parse_args()
    
    # モデルディレクトリ作成
    os.makedirs('resin_washer_model', exist_ok=True)
    
    # 学習実行
    trainer = PrecisionFixTrainer(args.data_dir)
    model, history = trainer.train_with_balance(epochs=args.epochs)
    
    if model is not None:
        print("\n=== 学習成功 ===")
        print("モデルが正常に学習されました")
    else:
        print("\n=== 学習失敗 ===")
        print("学習に失敗しました")

if __name__ == "__main__":
    main()
