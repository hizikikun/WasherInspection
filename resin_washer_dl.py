#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
樹脂製ワッシャー外観検査用ディープラーニングシステム
京都先端科学大学 機械電気システム工学科
プロジェクト: 低コスト画像検査装置の開発
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import time
from pathlib import Path
import argparse
from collections import Counter
import pandas as pd

# 文字化け対策
plt.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.use('Agg')

class ResinWasherDataset:
    """樹脂製ワッシャーデータセット管理クラス"""
    
    def __init__(self, data_dir="resin_washer_data"):
        self.data_dir = Path(data_dir)
        self.images = []
        self.labels = []
        self.defect_types = []
        self.image_size = (224, 224)
        
        # 欠陥タイプの定義
        self.defect_categories = {
            'good': 0,
            'black_spot': 1,
            'crack': 2,
            'foreign_matter': 3,
            'deformation': 4,
            'discoloration': 5
        }
        
    def load_dataset(self):
        """データセットを読み込み"""
        print("=== 樹脂製ワッシャーデータセット読み込み ===")
        
        if not self.data_dir.exists():
            print(f"データディレクトリが見つかりません: {self.data_dir}")
            return False
            
        # 画像ファイルを検索
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.data_dir.rglob(f"*{ext}"))
            image_files.extend(self.data_dir.rglob(f"*{ext.upper()}"))
        
        print(f"見つかった画像ファイル数: {len(image_files)}")
        
        if len(image_files) == 0:
            print("画像ファイルが見つかりませんでした")
            return False
        
        # 画像を読み込み、ラベルを決定
        for img_path in image_files:
            try:
                # ファイル名からラベルを決定
                filename = img_path.stem.lower()
                
                if 'good' in filename:
                    label = 'good'
                elif 'defect_black_spot' in filename or 'black_spot' in filename:
                    label = 'black_spot'
                elif 'defect_crack' in filename or 'crack' in filename:
                    label = 'crack'
                elif 'defect_foreign_matter' in filename or 'foreign_matter' in filename:
                    label = 'foreign_matter'
                elif 'defect_deformation' in filename or 'deformation' in filename:
                    label = 'deformation'
                elif 'defect_discoloration' in filename or 'discoloration' in filename:
                    label = 'discoloration'
                else:
                    # ファイル名から判定できない場合はスキップ
                    print(f"ラベルを判定できません: {img_path}")
                    continue
                
                # 画像を読み込み
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"画像を読み込めません: {img_path}")
                    continue
                
                # 画像をリサイズ
                image = cv2.resize(image, self.image_size)
                
                # 欠陥箇所のマークを検出（赤い円）
                marked_image, defect_areas = self._detect_defect_marks(image)
                
                self.images.append(marked_image)
                self.labels.append(label)
                self.defect_types.append(self.defect_categories[label])
                
            except Exception as e:
                print(f"エラー: {img_path} - {e}")
                continue
        
        print(f"読み込み完了: {len(self.images)}枚")
        print(f"ラベル分布: {Counter(self.labels)}")
        
        return len(self.images) > 0
    
    def _detect_defect_marks(self, image):
        """画像内の赤いマーク（欠陥箇所）を検出"""
        # HSV色空間に変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 赤色の範囲を定義
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # 赤色マスクを作成
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # 赤いマークを検出
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defect_areas = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # 小さなノイズを除外
                x, y, w, h = cv2.boundingRect(contour)
                defect_areas.append((x, y, w, h))
        
        return image, defect_areas
    
    def get_data_info(self):
        """データセット情報を取得"""
        if not self.images:
            return None
        
        info = {
            'total_images': len(self.images),
            'image_size': self.image_size,
            'label_distribution': dict(Counter(self.labels)),
            'defect_categories': self.defect_categories
        }
        
        return info

class ResinWasherCNN:
    """樹脂製ワッシャー用CNNモデル"""
    
    def __init__(self, num_classes=6, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self):
        """CNNモデルを構築"""
        model = keras.Sequential([
            # データ拡張層
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # 畳み込み層
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
            
            # 全結合層
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """モデルを学習"""
        if self.model is None:
            self.build_model()
        
        # コールバック設定
        callbacks = [
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
            )
        ]
        
        # 学習実行
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """モデルを評価"""
        if self.model is None:
            return None
        
        # 予測
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # 評価指標
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'y_pred': y_pred,
            'y_pred_classes': y_pred_classes
        }
    
    def save_model(self, filepath):
        """モデルを保存"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"モデルを保存しました: {filepath}")
    
    def load_model(self, filepath):
        """モデルを読み込み"""
        self.model = keras.models.load_model(filepath)
        print(f"モデルを読み込みました: {filepath}")

class ResinWasherInspector:
    """樹脂製ワッシャー検査システム"""
    
    def __init__(self, model_path=None):
        self.cnn = ResinWasherCNN()
        self.defect_categories = {
            0: 'good',
            1: 'black_spot',
            2: 'crack', 
            3: 'foreign_matter',
            4: 'deformation',
            5: 'discoloration'
        }
        
        if model_path and os.path.exists(model_path):
            self.cnn.load_model(model_path)
    
    def predict_single_image(self, image_path):
        """単一画像の予測"""
        if self.cnn.model is None:
            print("モデルが読み込まれていません")
            return None
        
        # 画像を読み込み
        image = cv2.imread(image_path)
        if image is None:
            print(f"画像を読み込めません: {image_path}")
            return None
        
        # 前処理
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = image.astype('float32') / 255.0
        
        # 予測
        prediction = self.cnn.model.predict(image)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return {
            'predicted_class': predicted_class,
            'class_name': self.defect_categories[predicted_class],
            'confidence': confidence,
            'all_probabilities': prediction[0]
        }
    
    def predict_batch(self, image_paths):
        """バッチ予測"""
        if self.cnn.model is None:
            print("モデルが読み込まれていません")
            return None
        
        results = []
        for image_path in image_paths:
            result = self.predict_single_image(image_path)
            if result:
                result['image_path'] = image_path
                results.append(result)
        
        return results

def train_resin_washer_model(data_dir="resin_washer_data", output_dir="resin_washer_model"):
    """樹脂製ワッシャーモデルの学習"""
    print("=== 樹脂製ワッシャーディープラーニング学習開始 ===")
    
    # データセット読み込み
    dataset = ResinWasherDataset(data_dir)
    if not dataset.load_dataset():
        print("データセットの読み込みに失敗しました")
        return False
    
    # データ情報表示
    info = dataset.get_data_info()
    print(f"データセット情報: {info}")
    
    # データ準備
    X = np.array(dataset.images)
    y = np.array(dataset.defect_types)
    
    # データ正規化
    X = X.astype('float32') / 255.0
    
    # 訓練・検証・テスト分割
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"訓練データ: {X_train.shape[0]}枚")
    print(f"検証データ: {X_val.shape[0]}枚")
    print(f"テストデータ: {X_test.shape[0]}枚")
    
    # モデル構築・学習
    cnn = ResinWasherCNN()
    model = cnn.build_model()
    
    print("モデル構造:")
    model.summary()
    
    # 学習実行
    history = cnn.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    # 評価
    eval_results = cnn.evaluate(X_test, y_test)
    
    print(f"テスト精度: {eval_results['test_accuracy']:.4f}")
    print(f"テスト損失: {eval_results['test_loss']:.4f}")
    
    # 分類レポート
    class_names = list(dataset.defect_categories.keys())
    print("\n分類レポート:")
    print(classification_report(y_test, eval_results['y_pred_classes'], 
                              target_names=[dataset.defect_categories[i] for i in class_names]))
    
    # モデル保存
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "resin_washer_model.h5")
    cnn.save_model(model_path)
    
    # 学習履歴保存
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump({
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }, f)
    
    # 可視化
    plot_training_history(history, output_dir)
    plot_confusion_matrix(y_test, eval_results['y_pred_classes'], 
                         [dataset.defect_categories[i] for i in class_names], output_dir)
    
    print(f"学習完了。モデルと結果は {output_dir} に保存されました。")
    return True

def plot_training_history(history, output_dir):
    """学習履歴を可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 損失
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # 精度
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """混同行列を可視化"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="樹脂製ワッシャー外観検査システム")
    parser.add_argument("--data-dir", type=str, default="resin_washer_data", 
                       help="データセットディレクトリ")
    parser.add_argument("--output-dir", type=str, default="resin_washer_model",
                       help="出力ディレクトリ")
    parser.add_argument("--train", action="store_true", help="学習モード")
    parser.add_argument("--predict", type=str, help="予測する画像パス")
    parser.add_argument("--model-path", type=str, help="学習済みモデルパス")
    
    args = parser.parse_args()
    
    if args.train:
        # 学習モード
        success = train_resin_washer_model(args.data_dir, args.output_dir)
        if success:
            print("学習が正常に完了しました")
        else:
            print("学習に失敗しました")
    
    elif args.predict:
        # 予測モード
        inspector = ResinWasherInspector(args.model_path)
        result = inspector.predict_single_image(args.predict)
        if result:
            print(f"予測結果: {result['class_name']} (信頼度: {result['confidence']:.4f})")
    
    else:
        print("使用方法:")
        print("学習: python resin_washer_dl.py --train --data-dir resin_washer_data")
        print("予測: python resin_washer_dl.py --predict image.jpg --model-path resin_washer_model/resin_washer_model.h5")

if __name__ == "__main__":
    main()







