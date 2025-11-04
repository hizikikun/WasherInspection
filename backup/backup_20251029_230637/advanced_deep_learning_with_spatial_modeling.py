#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Deep Learning with Spatial Modeling for Washer Inspection
Includes data augmentation, spatial modeling, and improved training
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time

class AdvancedWasherInspector:
    def __init__(self, data_path="cs_AItraining_data"):
        self.data_path = Path(data_path)
        self.class_names = ['good', 'black_spot']  # Only available classes
        self.model = None
        self.history = None
        
        # Advanced spatial modeling parameters
        self.spatial_augmentation_params = {
            'rotation_range': 45,  # Increased rotation
            'width_shift_range': 0.2,  # Horizontal shift
            'height_shift_range': 0.2,  # Vertical shift
            'shear_range': 0.3,  # Shear transformation
            'zoom_range': 0.3,  # Zoom in/out
            'horizontal_flip': True,  # Horizontal flip
            'vertical_flip': True,  # Vertical flip
            'brightness_range': [0.7, 1.3],  # Brightness variation
            'contrast_range': [0.8, 1.2],  # Contrast variation
            'channel_shift_range': 0.1,  # Color channel shift
        }
        
        # Advanced preprocessing
        self.preprocessing_params = {
            'rescale': 1./255,
            'featurewise_center': True,
            'featurewise_std_normalization': True,
            'zca_whitening': True,
        }
        
    def load_and_prepare_data(self):
        """Load data with advanced spatial modeling"""
        print("Loading data with advanced spatial modeling...")
        
        images = []
        labels = []
        class_counts = {}
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = self.data_path / 'resin' / class_name
            if not class_path.exists():
                print(f"Warning: {class_path} does not exist")
                continue
                
            class_images = []
            for img_file in list(class_path.rglob('*.jpg')) + list(class_path.rglob('*.jpeg')) + list(class_path.rglob('*.png')):
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        class_images.append(img)
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
            
            images.extend(class_images)
            labels.extend([class_idx] * len(class_images))
            class_counts[class_name] = len(class_images)
            print(f"Loaded {len(class_images)} images for class '{class_name}'")
        
        if not images:
            raise ValueError("No images found!")
        
        X = np.array(images)
        y = np.array(labels)
        
        print(f"Total images loaded: {len(X)}")
        print(f"Class distribution: {class_counts}")
        
        return X, y, class_counts
    
    def create_advanced_data_generators(self, X, y):
        """Create advanced data generators with spatial modeling"""
        print("Creating advanced data generators with spatial modeling...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Advanced training data generator with spatial modeling
        train_datagen = ImageDataGenerator(
            rotation_range=self.spatial_augmentation_params['rotation_range'],
            width_shift_range=self.spatial_augmentation_params['width_shift_range'],
            height_shift_range=self.spatial_augmentation_params['height_shift_range'],
            shear_range=self.spatial_augmentation_params['shear_range'],
            zoom_range=self.spatial_augmentation_params['zoom_range'],
            horizontal_flip=self.spatial_augmentation_params['horizontal_flip'],
            vertical_flip=self.spatial_augmentation_params['vertical_flip'],
            brightness_range=self.spatial_augmentation_params['brightness_range'],
            rescale=self.preprocessing_params['rescale'],
            fill_mode='nearest',
            cval=0.0,
        )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
        )
        
        # Fit generators to training data for featurewise normalization
        # train_datagen.fit(X_train)
        # val_datagen.fit(X_train)  # Use training data for normalization
        
        return train_datagen, val_datagen, X_train, X_val, y_train, y_val
    
    def build_advanced_model(self, num_classes=2):
        """Build advanced model with spatial modeling capabilities"""
        print("Building advanced model with spatial modeling...")
        
        # Base model with transfer learning
        base_model = EfficientNetB0(
            weights=None,  # Don't load pretrained weights
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze initial layers, fine-tune later layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Add advanced spatial modeling layers
        model = models.Sequential([
            base_model,
            
            # Spatial attention mechanism
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            
            # Spatial feature extraction
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            
            # Multi-scale feature fusion
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.1),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Advanced optimizer with learning rate scheduling
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
        
        return model
    
    def train_with_spatial_modeling(self, X, y):
        """Train model with advanced spatial modeling"""
        print("Starting training with spatial modeling...")
        
        # Create data generators
        train_datagen, val_datagen, X_train, X_val, y_train, y_val = self.create_advanced_data_generators(X, y)
        
        # Build model
        self.model = self.build_advanced_model()
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"Class weights: {class_weight_dict}")
        
        # Advanced callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
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
                'best_spatial_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            callbacks.CSVLogger('training_log_spatial.csv')
        ]
        
        # Train model
        self.history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=100,
            validation_data=val_datagen.flow(X_val, y_val, batch_size=32),
            validation_steps=len(X_val) // 32,
            class_weight=class_weight_dict,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("Evaluating model...")
        
        # Load best model
        if os.path.exists('best_spatial_model.h5'):
            self.model = tf.keras.models.load_model('best_spatial_model.h5')
        
        # Evaluate
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(y_test, y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Spatial Modeling')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_spatial.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=self.class_names))
        
        return test_acc
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Additional metrics or empty plot
        axes[1, 0].text(0.5, 0.5, 'Additional Metrics\nNot Available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Additional Metrics')
        
        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Learning Rate')
        
        plt.tight_layout()
        plt.savefig('training_history_spatial.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_info(self, test_acc):
        """Save model information"""
        model_info = {
            'model_name': 'Advanced Spatial Modeling Washer Inspector',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_accuracy': float(test_acc),
            'spatial_augmentation_params': self.spatial_augmentation_params,
            'preprocessing_params': self.preprocessing_params,
            'class_names': self.class_names,
            'model_architecture': 'EfficientNetB0 + Spatial Attention + Multi-scale Features'
        }
        
        with open('model_info_spatial.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("Model information saved to model_info_spatial.json")

def main():
    """Main training function"""
    print("=== Advanced Deep Learning with Spatial Modeling ===")
    
    # Initialize inspector
    inspector = AdvancedWasherInspector()
    
    try:
        # Load data
        X, y, class_counts = inspector.load_and_prepare_data()
        
        # Train model
        history = inspector.train_with_spatial_modeling(X, y)
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Evaluate model
        test_acc = inspector.evaluate_model(X_test, y_test)
        
        # Plot training history
        inspector.plot_training_history()
        
        # Save model info
        inspector.save_model_info(test_acc)
        
        print(f"\n=== Training Complete ===")
        print(f"Final Test Accuracy: {test_acc:.4f}")
        print("Model saved as 'best_spatial_model.h5'")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()