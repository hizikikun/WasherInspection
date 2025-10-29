#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-Class Ultra-High Accuracy Deep Learning with Corrected Data Loading
Complete defect detection system for good, black_spot, chipping, and scratch
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from collections import Counter

class CorrectedFourClassUltraHighAccuracyWasherInspector:
    def __init__(self, data_path="cs_AItraining_data"):
        self.data_path = Path(data_path)
        self.class_names = ['good', 'black_spot', 'chipping', 'scratch']  # 4 classes
        self.models = []
        self.histories = []
        
        # Ultra-advanced spatial modeling parameters
        self.ultra_spatial_augmentation_params = {
            # Geometric transformations
            'rotation_range': 60,  # Increased rotation
            'width_shift_range': 0.3,  # More horizontal shift
            'height_shift_range': 0.3,  # More vertical shift
            'shear_range': 0.4,  # More shear transformation
            'zoom_range': 0.4,  # More zoom variation
            'horizontal_flip': True,
            'vertical_flip': True,
            
            # Color and brightness variations
            'brightness_range': [0.6, 1.4],  # Wider brightness range
            'channel_shift_range': 0.2,  # More color channel shift
            
            # Advanced augmentations
            'fill_mode': 'nearest',
            'cval': 0.0,
        }
        
    def load_and_prepare_data(self):
        """Load data with comprehensive spatial modeling for 4 classes - CORRECTED VERSION"""
        print("Loading data with ultra-advanced spatial modeling for 4 classes (CORRECTED)...")
        
        images = []
        labels = []
        class_counts = {}
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = self.data_path / 'resin' / class_name
            if not class_path.exists():
                print(f"Warning: {class_path} does not exist")
                continue
                
            class_images = []
            print(f"Loading images from: {class_path}")
            
            # Use glob to find all image files recursively
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(class_path.rglob(ext))
            
            print(f"Found {len(image_files)} image files for class '{class_name}'")
            
            for img_file in image_files:
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        class_images.append(img)
                    else:
                        print(f"Warning: Could not load {img_file}")
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
            
            images.extend(class_images)
            labels.extend([class_idx] * len(class_images))
            class_counts[class_name] = len(class_images)
            print(f"Successfully loaded {len(class_images)} images for class '{class_name}'")
        
        if not images:
            raise ValueError("No images found!")
        
        X = np.array(images)
        y = np.array(labels)
        
        print(f"Total images loaded: {len(X)}")
        print(f"Class distribution: {class_counts}")
        
        # Verify expected counts
        expected_counts = {'good': 1144, 'black_spot': 88, 'chipping': 117, 'scratch': 112}
        for class_name, expected_count in expected_counts.items():
            actual_count = class_counts.get(class_name, 0)
            if actual_count != expected_count:
                print(f"WARNING: {class_name} expected {expected_count}, got {actual_count}")
            else:
                print(f"✓ {class_name}: {actual_count} images (correct)")
        
        return X, y, class_counts
    
    def create_ultra_advanced_data_generators(self, X, y):
        """Create ultra-advanced data generators with comprehensive spatial modeling"""
        print("Creating ultra-advanced data generators with comprehensive spatial modeling...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        
        # Ultra-advanced training data generator
        train_datagen = ImageDataGenerator(
            **self.ultra_spatial_augmentation_params,
            rescale=1./255,
        )
        
        # Validation data generator (minimal augmentation)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
        )
        
        return train_datagen, val_datagen, X_train, X_val, y_train, y_val
    
    def build_ensemble_models(self, num_classes=4):
        """Build ensemble of different EfficientNet models for 4 classes"""
        print("Building ensemble of EfficientNet models for 4 classes...")
        
        models_config = [
            {'name': 'EfficientNetB0', 'model': EfficientNetB0, 'input_size': (224, 224, 3)},
            {'name': 'EfficientNetB1', 'model': EfficientNetB1, 'input_size': (240, 240, 3)},
            {'name': 'EfficientNetB2', 'model': EfficientNetB2, 'input_size': (260, 260, 3)},
        ]
        
        ensemble_models = []
        
        for config in models_config:
            print(f"Building {config['name']} for 4-class classification...")
            
            # Base model
            base_model = config['model'](
                weights=None,  # Don't load pretrained weights
                include_top=False,
                input_shape=config['input_size']
            )
            
            # Freeze initial layers
            for layer in base_model.layers[:-30]:
                layer.trainable = False
            
            # Add advanced layers for 4-class classification
            model = models.Sequential([
                base_model,
                
                # Global pooling
                layers.GlobalAveragePooling2D(),
                
                # Advanced feature extraction
                layers.Dense(1024, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                
                # Output layer for 4 classes
                layers.Dense(num_classes, activation='softmax')
            ])
            
            # Advanced optimizer
            optimizer = optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )
            
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            ensemble_models.append({
                'model': model,
                'name': config['name'],
                'input_size': config['input_size']
            })
        
        return ensemble_models
    
    def train_ensemble_with_cross_validation(self, X, y):
        """Train ensemble with cross-validation for 4 classes"""
        print("Starting ensemble training with cross-validation for 4 classes...")
        
        # Create data generators
        train_datagen, val_datagen, X_train, X_val, y_train, y_val = self.create_ultra_advanced_data_generators(X, y)
        
        # Build ensemble models
        ensemble_models = self.build_ensemble_models()
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"Class weights: {class_weight_dict}")
        
        # Train each model in the ensemble
        for i, model_config in enumerate(ensemble_models):
            print(f"\nTraining {model_config['name']} for 4 classes ({i+1}/{len(ensemble_models)})...")
            
            model = model_config['model']
            model_name = model_config['name']
            
            # Advanced callbacks
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=25,  # Increased patience for 4 classes
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=12,  # Increased patience
                    min_lr=1e-7,
                    verbose=1
                ),
                callbacks.ModelCheckpoint(
                    f'corrected_best_4class_{model_name.lower()}_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                callbacks.CSVLogger(f'corrected_training_log_4class_{model_name.lower()}.csv')
            ]
            
            # Resize data for different input sizes
            if model_config['input_size'] != (224, 224, 3):
                new_size = model_config['input_size'][0]
                X_train_resized = np.array([cv2.resize(img, (new_size, new_size)) for img in X_train])
                X_val_resized = np.array([cv2.resize(img, (new_size, new_size)) for img in X_val])
            else:
                X_train_resized = X_train
                X_val_resized = X_val
            
            # Train model
            history = model.fit(
                train_datagen.flow(X_train_resized, y_train, batch_size=16),
                steps_per_epoch=len(X_train_resized) // 16,
                epochs=200,  # Increased epochs for 4 classes
                validation_data=val_datagen.flow(X_val_resized, y_val, batch_size=16),
                validation_steps=len(X_val_resized) // 16,
                class_weight=class_weight_dict,
                callbacks=callbacks_list,
                verbose=1
            )
            
            self.models.append(model)
            self.histories.append(history)
            
            print(f"{model_name} training completed!")
        
        return self.histories
    
    def create_ensemble_predictions(self, X_test):
        """Create ensemble predictions for 4 classes"""
        print("Creating ensemble predictions for 4 classes...")
        
        predictions = []
        
        for i, model_config in enumerate(self.models):
            model_name = f"Model_{i+1}"
            print(f"Predicting with {model_name}...")
            
            # Resize if necessary
            if hasattr(model_config, 'input_shape'):
                input_size = model_config.input_shape[1]
                if input_size != 224:
                    X_test_resized = np.array([cv2.resize(img, (input_size, input_size)) for img in X_test])
                else:
                    X_test_resized = X_test
            else:
                X_test_resized = X_test
            
            # Predict
            pred = model_config.predict(X_test_resized)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
        
        return ensemble_pred, ensemble_pred_classes
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble performance for 4 classes"""
        print("Evaluating ensemble for 4 classes...")
        
        # Create ensemble predictions
        ensemble_pred, ensemble_pred_classes = self.create_ensemble_predictions(X_test)
        
        # Calculate accuracy
        accuracy = np.mean(ensemble_pred_classes == y_test)
        
        print(f"Ensemble Test Accuracy: {accuracy:.4f}")
        
        # Individual model accuracies
        for i, model in enumerate(self.models):
            if hasattr(model, 'input_shape'):
                input_size = model.input_shape[1]
                if input_size != 224:
                    X_test_resized = np.array([cv2.resize(img, (input_size, input_size)) for img in X_test])
                else:
                    X_test_resized = X_test
            else:
                X_test_resized = X_test
            
            pred = model.predict(X_test_resized)
            pred_classes = np.argmax(pred, axis=1)
            model_acc = np.mean(pred_classes == y_test)
            print(f"Model {i+1} Accuracy: {model_acc:.4f}")
        
        # Detailed evaluation
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(y_test, ensemble_pred_classes)
        
        print("\nEnsemble Confusion Matrix:")
        print(cm)
        
        print("\nEnsemble Classification Report:")
        print(classification_report(y_test, ensemble_pred_classes, target_names=self.class_names))
        
        return accuracy
    
    def save_ensemble_models(self):
        """Save all ensemble models for 4 classes"""
        print("Saving ensemble models for 4 classes...")
        
        for i, model in enumerate(self.models):
            model_name = f"corrected_ensemble_4class_model_{i+1}.h5"
            model.save(model_name)
            print(f"Saved {model_name}")
        
        # Save ensemble info
        ensemble_info = {
            'model_name': 'Corrected 4-Class Ultra-High Accuracy Ensemble Washer Inspector',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_classes': 4,
            'class_names': self.class_names,
            'expected_data_counts': {'good': 1144, 'black_spot': 88, 'chipping': 117, 'scratch': 112},
            'num_models': len(self.models),
            'model_types': ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2'],
            'spatial_augmentation_params': self.ultra_spatial_augmentation_params,
            'description': 'Corrected ensemble of EfficientNet models for 4-class defect detection with comprehensive spatial modeling'
        }
        
        with open('corrected_ensemble_4class_info.json', 'w') as f:
            json.dump(ensemble_info, f, indent=2)
        
        print("Ensemble information saved to corrected_ensemble_4class_info.json")

def main():
    """Main training function for 4 classes - CORRECTED VERSION"""
    print("=== Corrected 4-Class Ultra-High Accuracy Deep Learning with Advanced Spatial Modeling ===")
    
    # Initialize inspector
    inspector = CorrectedFourClassUltraHighAccuracyWasherInspector()
    
    try:
        # Load data
        X, y, class_counts = inspector.load_and_prepare_data()
        
        # Train ensemble
        histories = inspector.train_ensemble_with_cross_validation(X, y)
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Evaluate ensemble
        test_acc = inspector.evaluate_ensemble(X_test, y_test)
        
        # Save models
        inspector.save_ensemble_models()
        
        print(f"\n=== Training Complete ===")
        print(f"Final Ensemble Test Accuracy: {test_acc:.4f}")
        print("All corrected 4-class ensemble models saved!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
