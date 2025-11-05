#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
樹脂製ワッシャー ディープラーニング学習システム
フォルダー構造対応版
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
import glob
from pathlib import Path
import argparse
from collections import Counter
import pandas as pd

# 文字化け対策
plt.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib
matplotlib.use('Agg')

class ResinWasherDataset:
    """樹脂製ワッシャーデータセット管理クラス（フォルダー構造対応）"""
    
    def __init__(self, data_dir="cs AI学習データ/樹脂"):
        self.data_dir = Path(data_dir)
        self.images = []
        self.labels = []
        self.defect_types = []
        self.image_size = (224, 224)
        
        # 欠陥タイプの定義（フォルダー名に対応）
        self.defect_categories = {
            '良品(樹脂)': 0,
            '欠け樹脂': 1,
            '黒点樹脂': 2,
            '傷樹脂': 3
        }
        
        # 良品データのサブフォルダーも検索
        self.good_subfolders = ['樹脂20251015(良品)', '良品', 'good']
        
        # 英語名との対応
        self.category_names = {
            0: 'good',
            1: 'chipped',
            2: 'black_spot', 
            3: 'scratched'
        }
        
    def load_dataset(self):
        """データセットを読み込み"""
        print("=" * 60)
        print("樹脂製ワッシャーデータセット読み込み")
        print("=" * 60)
        
        if not self.data_dir.exists():
            print(f"エラー: データディレクトリが見つかりません: {self.data_dir}")
            return False
        
        # 各フォルダーから画像を読み込み
        total_images = 0
        category_counts = {}
        
        for category_name, category_id in self.defect_categories.items():
            category_dir = self.data_dir / category_name
            if not category_dir.exists():
                print(f"警告: フォルダーが見つかりません: {category_name}")
                continue
            
            # 画像ファイルを検索
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
            category_images = []
            
            # 良品の場合はサブフォルダーも検索
            if category_name == '良品(樹脂)':
                for subfolder in self.good_subfolders:
                    subfolder_path = category_dir / subfolder
                    if subfolder_path.exists():
                        print(f"  サブフォルダー発見: {subfolder}")
                        for ext in image_extensions:
                            category_images.extend(glob.glob(str(subfolder_path / ext)))
                            category_images.extend(glob.glob(str(subfolder_path / ext.upper())))
            else:
                for ext in image_extensions:
                    category_images.extend(glob.glob(str(category_dir / ext)))
                    category_images.extend(glob.glob(str(category_dir / ext.upper())))
            
            print(f"{category_name}: {len(category_images)}枚")
            category_counts[category_name] = len(category_images)
            
            # 画像を読み込み
            for img_path in category_images:
                try:
                    # 画像を読み込み（日本語パス対応）
                    import os
                    import numpy as np
                    img_path_str = str(img_path)
                    if not os.path.exists(img_path_str):
                        print(f"警告: ファイルが見つかりません: {img_path_str}")
                        continue
                    
                    # ファイル名を確認
                    filename = os.path.basename(img_path_str)
                    print(f"  読み込み中: {filename}")
                    
                    # 複数の方法で画像を読み込み
                    image = None
                    
                    # 方法1: cv2.imread
                    try:
                        image = cv2.imread(img_path_str)
                        if image is not None:
                            print(f"    cv2.imread成功: {filename}")
                    except Exception as e:
                        print(f"    cv2.imread失敗: {e}")
                    
                    # 方法2: PIL経由で読み込み
                    if image is None:
                        try:
                            from PIL import Image
                            pil_image = Image.open(img_path_str)
                            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                            print(f"    PIL経由で成功: {filename}")
                        except Exception as e:
                            print(f"    PIL経由でも失敗: {e}")
                    
                    # 方法3: バイナリ読み込み
                    if image is None:
                        try:
                            with open(img_path_str, 'rb') as f:
                                img_data = f.read()
                            nparr = np.frombuffer(img_data, np.uint8)
                            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if image is not None:
                                print(f"    バイナリ読み込み成功: {filename}")
                        except Exception as e:
                            print(f"    バイナリ読み込みも失敗: {e}")
                    
                    if image is None:
                        print(f"警告: すべての方法で画像を読み込めません: {filename}")
                        continue
                    
                    # 画像をリサイズ
                    image = cv2.resize(image, self.image_size)
                    
                    # 欠陥箇所のマークを検出（赤い円）
                    marked_image, defect_areas = self._detect_defect_marks(image)
                    
                    self.images.append(marked_image)
                    self.labels.append(category_name)
                    self.defect_types.append(category_id)
                    total_images += 1
                    
                except Exception as e:
                    print(f"エラー: {img_path} - {e}")
                    continue
        
        print(f"\n読み込み完了: {total_images}枚")
        print(f"カテゴリ別内訳:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}枚")
        
        # データバランス分析
        self._analyze_data_balance()
        
        return total_images > 0
    
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
    
    def _analyze_data_balance(self):
        """データバランスを分析"""
        print("\n【データバランス分析】")
        
        label_counts = Counter(self.labels)
        total = len(self.images)
        
        print(f"総画像数: {total}枚")
        
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            print(f"  {label}: {count}枚 ({percentage:.1f}%)")
        
        # バランス評価
        if total < 100:
            print("\n[警告] データ量が少ないです（100枚未満）")
            print("推奨: 各カテゴリ最低25枚以上")
        elif total < 300:
            print("\n[注意] データ量が少なめです（100-300枚）")
            print("推奨: 各カテゴリ最低50枚以上")
        else:
            print("\n[OK] データ量は十分です（300枚以上）")
        
        # カテゴリ間のバランスチェック
        if len(label_counts) > 1:
            counts = list(label_counts.values())
            max_count = max(counts)
            min_count = min(counts)
            ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if ratio > 3:
                print(f"[警告] カテゴリ間のバランスが悪いです（比率: {ratio:.1f}）")
                print("推奨: 最大/最小の比率を3以下にしてください")
            else:
                print(f"[OK] カテゴリバランスは良好です（比率: {ratio:.1f}）")
    
    def get_data_info(self):
        """データセット情報を取得"""
        if not self.images:
            return None
        
        info = {
            'total_images': len(self.images),
            'image_size': self.image_size,
            'label_distribution': dict(Counter(self.labels)),
            'defect_categories': self.defect_categories,
            'category_names': self.category_names
        }
        
        return info

class ResinWasherCNN:
    """樹脂製ワッシャー用CNNモデル"""
    
    def __init__(self, num_classes=4, input_shape=(224, 224, 3)):
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

def train_resin_washer_model(data_dir="cs AI学習データ/樹脂", output_dir="resin_washer_model"):
    """樹脂製ワッシャーモデルの学習"""
    print("=" * 60)
    print("樹脂製ワッシャー ディープラーニング学習開始")
    print("=" * 60)
    
    # データセット読み込み
    dataset = ResinWasherDataset(data_dir)
    if not dataset.load_dataset():
        print("データセットの読み込みに失敗しました")
        return False
    
    # データ情報表示
    info = dataset.get_data_info()
    print(f"\nデータセット情報: {info}")
    
    # データ準備
    X = np.array(dataset.images)
    y = np.array(dataset.defect_types)
    
    # データ正規化
    X = X.astype('float32') / 255.0
    
    # 訓練・検証・テスト分割
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"\nデータ分割:")
    print(f"  訓練データ: {X_train.shape[0]}枚")
    print(f"  検証データ: {X_val.shape[0]}枚")
    print(f"  テストデータ: {X_test.shape[0]}枚")
    
    # モデル構築・学習
    cnn = ResinWasherCNN()
    model = cnn.build_model()
    
    print("\nモデル構造:")
    model.summary()
    
    # 学習実行
    print("\n学習開始...")
    history = cnn.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    # 評価
    print("\n評価中...")
    eval_results = cnn.evaluate(X_test, y_test)
    
    print(f"\nテスト精度: {eval_results['test_accuracy']:.4f}")
    print(f"テスト損失: {eval_results['test_loss']:.4f}")
    
    # 分類レポート
    class_names = [dataset.category_names[i] for i in sorted(dataset.category_names.keys())]
    print("\n分類レポート:")
    print(classification_report(y_test, eval_results['y_pred_classes'], 
                              target_names=class_names))
    
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
                         class_names, output_dir)
    
    print(f"\n学習完了！モデルと結果は {output_dir} に保存されました。")
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
    parser.add_argument("--data-dir", type=str, default="cs AI学習データ/樹脂", 
                       help="データセットディレクトリ")
    parser.add_argument("--output-dir", type=str, default="resin_washer_model",
                       help="出力ディレクトリ")
    parser.add_argument("--train", action="store_true", help="学習モード")
    
    args = parser.parse_args()
    
    if args.train:
        # 学習モード
        success = train_resin_washer_model(args.data_dir, args.output_dir)
        if success:
            print("学習が正常に完了しました")
        else:
            print("学習に失敗しました")
    else:
        print("使用方法:")
        print("学習: python resin_washer_trainer.py --train")

if __name__ == "__main__":
    main()
