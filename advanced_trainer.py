#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
樹脂製ワッシャー 高度版ディープラーニング学習システム
Transfer Learning + 高度なデータ拡張で精度向上
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
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

# 文字化け対策
plt.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib
matplotlib.use('Agg')

class AdvancedResinWasherDataset:
    """高度版樹脂製ワッシャーデータセット管理クラス"""
    
    def __init__(self, data_dir="cs AI学習データ/樹脂"):
        self.data_dir = Path(data_dir)
        self.images = []
        self.labels = []
        self.defect_types = []
        self.image_size = (224, 224)
        
        # 欠陥タイプの定義
        self.defect_categories = {
            '良品(樹脂)': 0,
            '欠け樹脂': 1,
            '黒点樹脂': 2,
            '傷樹脂': 3
        }
        
        # 良品データのサブフォルダーも検索
        self.good_subfolders = ['樹脂20251015(良品)', '良品', 'good']
        
    def load_images(self):
        """画像を読み込み（高度な前処理）"""
        print("=== 高度版データセット読み込み開始 ===")
        
        for category, label in self.defect_categories.items():
            category_path = self.data_dir / category
            if not category_path.exists():
                print(f"警告: {category_path} が見つかりません")
                continue
                
            # 良品の場合はサブフォルダーも検索
            if category == '良品(樹脂)':
                image_files = []
                for subfolder in self.good_subfolders:
                    subfolder_path = category_path / subfolder
                    if subfolder_path.exists():
                        files = glob.glob(str(subfolder_path / "*.jpg")) + glob.glob(str(subfolder_path / "*.jpeg"))
                        image_files.extend(files)
                        print(f"  {subfolder}: {len(files)}枚")
            else:
                image_files = glob.glob(str(category_path / "*.jpg")) + glob.glob(str(category_path / "*.jpeg"))
                print(f"  {category}: {len(image_files)}枚")
            
            # 重複除去
            unique_files = list(set(image_files))
            print(f"  重複除去後: {len(unique_files)}枚")
            
            # 各画像を読み込み
            loaded_count = 0
            for img_path in unique_files:
                try:
                    # 複数の方法で画像読み込み
                    img = self._load_image_multiple_methods(img_path)
                    if img is not None:
                        # 高度な前処理
                        img = self._advanced_preprocessing(img)
                        self.images.append(img)
                        self.labels.append(label)
                        self.defect_types.append(category)
                        loaded_count += 1
                except Exception as e:
                    print(f"  読み込み失敗: {img_path} - {e}")
            
            print(f"  実際に読み込めた画像: {loaded_count}枚")
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        print(f"\n=== データセット統計 ===")
        print(f"総画像数: {len(self.images)}")
        for category, label in self.defect_categories.items():
            count = np.sum(self.labels == label)
            print(f"{category}: {count}枚")
        
        return self.images, self.labels
    
    def _load_image_multiple_methods(self, img_path):
        """複数の方法で画像を読み込み"""
        # 方法1: OpenCV
        try:
            img = cv2.imread(img_path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            pass
        
        # 方法2: PIL
        try:
            img = Image.open(img_path)
            return np.array(img.convert('RGB'))
        except:
            pass
        
        # 方法3: バイナリ読み込み
        try:
            with open(img_path, 'rb') as f:
                img_data = f.read()
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            pass
        
        return None
    
    def _advanced_preprocessing(self, img):
        """高度な前処理"""
        # リサイズ
        img = cv2.resize(img, self.image_size)
        
        # グレースケール変換（EfficientNetの入力チャンネル問題を回避）
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # ノイズ除去
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        # コントラスト調整
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return img
    
    def create_balanced_dataset(self, target_samples_per_class=200):
        """バランスの取れたデータセットを作成（より多くのサンプル）"""
        print(f"\n=== バランス調整開始 (目標: 各クラス{target_samples_per_class}枚) ===")
        
        balanced_images = []
        balanced_labels = []
        balanced_types = []
        
        for category, label in self.defect_categories.items():
            # 現在のクラスの画像を取得
            class_mask = self.labels == label
            class_images = self.images[class_mask]
            class_types = [self.defect_types[i] for i in range(len(self.defect_types)) if class_mask[i]]
            
            current_count = len(class_images)
            print(f"{category}: {current_count}枚")
            
            if current_count >= target_samples_per_class:
                # 十分なデータがある場合はランダムサンプリング
                indices = np.random.choice(current_count, target_samples_per_class, replace=False)
                balanced_images.extend(class_images[indices])
                balanced_labels.extend([label] * target_samples_per_class)
                balanced_types.extend([class_types[i] for i in indices])
            else:
                # データが不足している場合は拡張
                needed = target_samples_per_class - current_count
                print(f"  不足: {needed}枚 → 高度なデータ拡張で補完")
                
                # 既存データを追加
                balanced_images.extend(class_images)
                balanced_labels.extend([label] * current_count)
                balanced_types.extend(class_types)
                
                # 高度なデータ拡張で不足分を補完
                augmented_images = self._advanced_augmentation(class_images, needed)
                balanced_images.extend(augmented_images)
                balanced_labels.extend([label] * needed)
                balanced_types.extend([category] * needed)
        
        self.images = np.array(balanced_images)
        self.labels = np.array(balanced_labels)
        self.defect_types = balanced_types
        
        print(f"\n=== バランス調整後 ===")
        print(f"総画像数: {len(self.images)}")
        for category, label in self.defect_categories.items():
            count = np.sum(self.labels == label)
            print(f"{category}: {count}枚")
    
    def _advanced_augmentation(self, images, target_count):
        """高度なデータ拡張"""
        augmented = []
        current_count = len(images)
        
        # 必要な回数だけ拡張
        for i in range(target_count):
            # ランダムに元画像を選択
            idx = i % current_count
            img = images[idx].copy()
            
            # 複数の変換を組み合わせ
            if np.random.random() > 0.5:
                img = cv2.flip(img, 1)  # 水平反転
            
            # 回転
            angle = np.random.uniform(-20, 20)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
            
            # 平行移動
            tx = np.random.uniform(-10, 10)
            ty = np.random.uniform(-10, 10)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w, h))
            
            # スケーリング
            scale = np.random.uniform(0.9, 1.1)
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
            img = cv2.resize(img, (w, h))
            
            # 明度調整
            brightness = np.random.uniform(0.7, 1.3)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)
            
            # コントラスト調整
            contrast = np.random.uniform(0.8, 1.2)
            img = np.clip(img * contrast, 0, 255).astype(np.uint8)
            
            # 色相調整
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:,:,0] = (hsv[:,:,0] + np.random.uniform(-10, 10)) % 180
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # ノイズ追加
            noise = np.random.normal(0, 3, img.shape).astype(np.uint8)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # ぼかし
            if np.random.random() > 0.7:
                kernel_size = np.random.choice([3, 5])
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
            augmented.append(img)
        
        return augmented

class AdvancedResinWasherModel:
    """高度版樹脂製ワッシャー分類モデル（Transfer Learning）"""
    
    def __init__(self, num_classes=4, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self):
        """Transfer Learningを使用したモデルを構築"""
        # EfficientNetB0をベースモデルとして使用
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # ベースモデルを凍結（最初は）
        base_model.trainable = False
        
        # モデル構築
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
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, class_weights=None):
        """モデルをコンパイル"""
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def fine_tune(self):
        """ファインチューニング（ベースモデルの一部を解凍）"""
        # ベースモデルの最後の数層を解凍
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # 最後の数層のみを学習可能にする
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
        # 学習率を下げる
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

def train_advanced_model():
    """高度版モデルを学習"""
    print("=== 高度版樹脂製ワッシャー分類モデル学習開始 ===")
    
    # データセット読み込み
    dataset = AdvancedResinWasherDataset()
    images, labels = dataset.load_images()
    
    # バランス調整（より多くのサンプル）
    dataset.create_balanced_dataset(target_samples_per_class=200)
    images, labels = dataset.images, dataset.labels
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # クラス重み計算
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"クラス重み: {class_weight_dict}")
    
    # モデル構築
    model_builder = AdvancedResinWasherModel()
    model = model_builder.build_model()
    model_builder.compile_model(class_weights=class_weight_dict)
    
    # 学習設定
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # 第1段階学習（ベースモデル凍結）
    print("\n=== 第1段階学習開始（ベースモデル凍結） ===")
    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # 第2段階学習（ファインチューニング）
    print("\n=== 第2段階学習開始（ファインチューニング） ===")
    model_builder.fine_tune()
    
    history2 = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # 評価
    print("\n=== 評価結果 ===")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"テスト精度: {test_accuracy:.4f}")
    print(f"テスト損失: {test_loss:.4f}")
    
    # 予測と分類レポート
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    class_names = ['良品', '欠け', '黒点', '傷']
    print("\n分類レポート:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))
    
    # モデル保存
    os.makedirs("resin_washer_model", exist_ok=True)
    model.save("resin_washer_model/resin_washer_model.h5")
    print(f"\nモデルを保存しました: resin_washer_model/resin_washer_model.h5")
    
    print("学習完了！")
    return model, history1, history2

if __name__ == "__main__":
    train_advanced_model()