#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
"""
Advanced Deep Learning Trainer for Resin Washer Defect Detection
- Transfer Learning with EfficientNetB0
- Advanced Data Augmentation
- Class Balancing
- Feature Analysis
- High Accuracy Training
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

class AdvancedDeepLearningTrainer:
    def __init__(self, data_path="C:/Users/tomoh/WasherInspection/cs_AItraining_data"):
        self.data_path = data_path
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 100
        self.class_names = ['good', 'chipping', 'black_spot', 'scratch']
        self.model = None
        self.history = None
        
    def analyze_data_distribution(self):
        """Analyze data distribution and quality"""
        print("=== Data Distribution Analysis ===")
        
        class_counts = {}
        total_images = 0
        
        for class_name in self.class_names:
            class_path = os.path.join(self.data_path, 'resin', class_name)
            if os.path.exists(class_path):
                # Count images in all subdirectories
                count = 0
                for root, dirs, files in os.walk(class_path):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            count += 1
                class_counts[class_name] = count
                total_images += count
                print(f"{class_name}: {count} images")
            else:
                class_counts[class_name] = 0
                print(f"{class_name}: 0 images (path not found)")
        
        print(f"Total images: {total_images}")
        
        # Calculate class weights for balancing
        if total_images > 0:
            class_weights = {}
            max_count = max(class_counts.values())
            for i, class_name in enumerate(self.class_names):
                if class_counts[class_name] > 0:
                    class_weights[i] = max_count / class_counts[class_name]
                else:
                    class_weights[i] = 1.0
            print(f"Class weights: {class_weights}")
            return class_weights
        else:
            print("No data found!")
            return None
    
    def load_and_preprocess_data(self):
        """Load and preprocess data with advanced augmentation"""
        print("=== Loading and Preprocessing Data ===")
        
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.data_path, 'resin', class_name)
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} not found")
                continue
                
            print(f"Loading {class_name} images...")
            count = 0
            
            # Walk through all subdirectories
            for root, dirs, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(root, file)
                        try:
                            # Load image
                            img = cv2.imread(img_path)
                            if img is not None:
                                # Convert BGR to RGB
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                # Resize
                                img = cv2.resize(img, self.img_size)
                                # Normalize
                                img = img.astype(np.float32) / 255.0
                                
                                images.append(img)
                                labels.append(class_idx)
                                count += 1
                                
                                if count % 100 == 0:
                                    print(f"  Loaded {count} {class_name} images...")
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
        
        if len(images) == 0:
            print("No images loaded!")
            return None, None
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"Total loaded: {len(images)} images")
        print(f"Image shape: {images.shape}")
        print(f"Label distribution: {Counter(labels)}")
        
        return images, labels
    
    def create_advanced_augmentation(self):
        """Create advanced data augmentation"""
        return ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2
        )
    
    def build_advanced_model(self):
        """Build advanced model with EfficientNetB0"""
        print("=== Building Advanced Model ===")
        
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        predictions = Dense(len(self.class_names), activation='softmax')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model created with {model.count_params()} parameters")
        return model
    
    def train_model(self, images, labels, class_weights):
        """Train the model with advanced techniques"""
        print("=== Training Advanced Model ===")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Create data generators
        train_datagen = self.create_advanced_augmentation()
        val_datagen = ImageDataGenerator(validation_split=0.0)
        
        # Build model
        self.model = self.build_advanced_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'advanced_model_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train with frozen base
        print("Phase 1: Training with frozen base model...")
        history1 = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=self.batch_size),
            steps_per_epoch=len(X_train) // self.batch_size,
            epochs=30,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Unfreeze some layers for fine-tuning
        print("Phase 2: Fine-tuning with unfrozen layers...")
        self.model.layers[1].trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history2 = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=self.batch_size),
            steps_per_epoch=len(X_train) // self.batch_size,
            epochs=20,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        self.history = {
            'loss': history1.history['loss'] + history2.history['loss'],
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
        }
        
        return self.history
    
    def evaluate_model(self, images, labels):
        """Evaluate model performance"""
        print("=== Model Evaluation ===")
        
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Predictions
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Confusion Matrix
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\nClassification Report:")
        print(classification_report(y_test, predicted_classes, target_names=self.class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, predicted_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return test_accuracy
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        plt.figure(figsize=(15, 5))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self):
        """Save the trained model"""
        if self.model is None:
            print("No model to save!")
            return
        
        # Save model
        self.model.save('advanced_resin_washer_model.h5')
        print("Model saved as 'advanced_resin_washer_model.h5'")
        
        # Save class names
        with open('class_names.json', 'w') as f:
            json.dump(self.class_names, f)
        print("Class names saved as 'class_names.json'")
    
    def run_training(self):
        """Run complete training pipeline"""
        print("=== Advanced Deep Learning Training Pipeline ===")
        
        # Analyze data
        class_weights = self.analyze_data_distribution()
        if class_weights is None:
            return
        
        # Load data
        images, labels = self.load_and_preprocess_data()
        if images is None:
            return
        
        # Train model
        history = self.train_model(images, labels, class_weights)
        
        # Evaluate model
        accuracy = self.evaluate_model(images, labels)
        
        # Plot history
        self.plot_training_history()
        
        # Save model
        self.save_model()
        
        print(f"\n=== Training Complete ===")
        print(f"Final Accuracy: {accuracy:.4f}")
        print("Model saved and ready for inference!")

def main():
    """Main function"""
    trainer = AdvancedDeepLearningTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main()