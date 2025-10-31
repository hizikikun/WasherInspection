#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Enhanced Trainer
シンプルで効果的な学習性能向上システム
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import json
import time

# Set UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

class SimpleEnhancedTrainer:
    def __init__(self, data_path="cs_AItraining_data"):
        self.data_path = Path(data_path)
        self.class_names = ['good', 'black_spot', 'chipping', 'scratch']
        self.models = []
        self.histories = []
        
        # 進捗管理
        self.status_path = Path('logs') / 'training_status.json'
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.total_start_time = None
        
        # 高度なデータ拡張設定
        self.enhanced_augmentation_params = {
            'rotation_range': 40,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.2,
            'zoom_range': 0.2,
            'horizontal_flip': True,
            'vertical_flip': True,
            'brightness_range': [0.7, 1.3],
            'channel_shift_range': 0.2,
            'fill_mode': 'nearest',
            'cval': 0.0
        }
    
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
        print("Simple Enhanced データ読み込み開始")
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
    
    def create_enhanced_data_generators(self, X, y):
        """強化されたデータジェネレーターを作成"""
        print("\n" + "=" * 60)
        print("強化されたデータジェネレーター作成")
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
        
        # 強化されたデータ拡張ジェネレーター
        train_datagen = ImageDataGenerator(
            **self.enhanced_augmentation_params,
            rescale=1./255,
        )
        
        val_datagen = ImageDataGenerator(
            rescale=1./255,
        )
        
        return X_train, X_val, y_train, y_val, class_weight_dict, train_datagen, val_datagen
    
    def build_enhanced_models(self):
        """強化されたモデルを構築"""
        print("\n" + "=" * 60)
        print("強化されたモデル構築")
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
            }
        ]
        
        ensemble_models = []
        
        for i, config in enumerate(models_config):
            print(f"\n構築中 {i+1}/{len(models_config)}: {config['name']}")
            
            # ベースモデル
            base_model = config['model'](
                weights='imagenet',
                include_top=False,
                input_shape=config['input_size']
            )
            
            # 段階的凍結戦略
            for layer in base_model.layers[:-40]:
                layer.trainable = False
            
            # 強化された分類ヘッド
            model = models.Sequential([
                base_model,
                
                # 空間的注意機構
                layers.GlobalAveragePooling2D(),
                layers.Dense(1024, activation='relu',
                           kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                
                # 特徴抽出層
                layers.Dense(512, activation='relu',
                           kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                # 最終分類層
                layers.Dense(256, activation='relu',
                           kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4)),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                
                layers.Dense(4, activation='softmax')
            ])
            
            # 最適化設定
            optimizer = optimizers.Adam(
                learning_rate=config['learning_rate'],
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
    
    def train_enhanced_models(self, X_train, X_val, y_train, y_val, class_weight_dict, train_datagen, val_datagen):
        """強化されたモデルで学習"""
        print("\n" + "=" * 80)
        print("強化されたモデルで学習開始")
        print("=" * 80)
        
        ensemble_models = self.build_enhanced_models()
        
        # 学習設定
        batch_size = 16
        max_epochs = 50
        patience = 15
        
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
            
            # データジェネレーター
            train_gen = train_datagen.flow(X_train_resized, y_train, batch_size=batch_size)
            val_gen = val_datagen.flow(X_val_resized, y_val, batch_size=batch_size)
            
            # コールバック設定
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
                    patience=8,
                    min_lr=1e-7,
                    verbose=1
                ),
                callbacks.ModelCheckpoint(
                    f'simple_enhanced_{model_name.lower()}_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                callbacks.CSVLogger(f'simple_enhanced_{model_name.lower()}_log.csv')
            ]
            
            # 学習実行
            print(f"学習開始: {model_name}")
            print(f"学習サンプル: {len(X_train_resized)}")
            print(f"検証サンプル: {len(X_val_resized)}")
            print(f"バッチサイズ: {batch_size}")
            print(f"最大エポック数: {max_epochs}")
            
            history = model.fit(
                train_gen,
                steps_per_epoch=len(X_train_resized) // batch_size,
                epochs=max_epochs,
                validation_data=val_gen,
                validation_steps=len(X_val_resized) // batch_size,
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
    
    def evaluate_enhanced_models(self, X_test, y_test):
        """強化されたモデルの評価"""
        print("\n" + "=" * 80)
        print("強化されたモデル評価")
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
        
        # アンサンブル予測（平均）
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
        
        # 精度計算
        accuracy = np.mean(ensemble_pred_classes == y_test)
        
        print(f"\n強化されたアンサンブル精度: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
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
    
    def save_enhanced_models(self):
        """強化されたモデルを保存"""
        print("\n" + "=" * 60)
        print("強化されたモデル保存")
        print("=" * 60)
        
        for i, model in enumerate(self.models):
            model_name = f"simple_enhanced_ensemble_model_{i+1}.h5"
            model.save(model_name)
            print(f"保存完了: {model_name}")
        
        # アンサンブル情報を保存
        ensemble_info = {
            'model_name': 'Simple Enhanced 4-Class Ensemble Washer Inspector',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_classes': 4,
            'class_names': self.class_names,
            'num_models': len(self.models),
            'model_types': ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2'],
            'features': [
                'Enhanced Data Augmentation',
                'Progressive Unfreezing',
                'Adam Optimizer',
                'L1/L2 Regularization',
                'Batch Normalization',
                'Dropout Regularization'
            ],
            'description': 'Simple enhanced ensemble with improved data augmentation and regularization'
        }
        
        with open('simple_enhanced_ensemble_info.json', 'w', encoding='utf-8') as f:
            json.dump(ensemble_info, f, indent=2, ensure_ascii=False)
        
        print("アンサンブル情報を保存しました: simple_enhanced_ensemble_info.json")

def main():
    """メイン実行関数"""
    print("=" * 80)
    print("Simple Enhanced Deep Learning Trainer")
    print("シンプルで効果的な学習性能向上システムを開始します")
    print("=" * 80)
    
    # トレーナー初期化
    trainer = SimpleEnhancedTrainer()
    
    try:
        # データ読み込み
        print("\n[ステップ1] データ読み込み")
        X, y, class_counts = trainer.load_and_prepare_data()
        
        # データジェネレーター作成
        print("\n[ステップ2] データジェネレーター作成")
        X_train, X_val, y_train, y_val, class_weight_dict, train_datagen, val_datagen = trainer.create_enhanced_data_generators(X, y)
        
        # 強化された学習実行
        print("\n[ステップ3] 強化された学習実行")
        histories = trainer.train_enhanced_models(X_train, X_val, y_train, y_val, class_weight_dict, train_datagen, val_datagen)
        
        # 評価実行
        print("\n[ステップ4] 性能評価")
        test_accuracy = trainer.evaluate_enhanced_models(X_val, y_val)
        
        # モデル保存
        print("\n[ステップ5] モデル保存")
        trainer.save_enhanced_models()
        
        # 結果表示
        total_time = time.time() - trainer.total_start_time
        print("\n" + "=" * 80)
        print("Simple Enhanced 学習完了!")
        print("=" * 80)
        print(f"最終精度: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"合計所要時間: {total_time/60:.1f}分")
        print("強化されたモデルを保存しました!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
