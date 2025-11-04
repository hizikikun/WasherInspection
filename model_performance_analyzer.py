#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Analysis
- Analyze current model performance
- Identify accuracy issues
- Debug prediction problems
"""

import os
import numpy as np
import cv2
import tensorflow as tf
import json
from collections import Counter
import matplotlib.pyplot as plt

class ModelPerformanceAnalyzer:
    def __init__(self, data_path="C:/Users/tomoh/WasherInspection/cs_AItraining_data"):
        self.data_path = data_path
        self.model = None
        self.class_names = ['good', 'chipping', 'black_spot', 'scratch']
        self.img_size = (224, 224)
        
    def load_model(self):
        """Load the current model"""
        model_paths = [
            "ultra_high_accuracy_resin_washer_model.h5",
            "high_accuracy_resin_washer_model.h5",
            "best_real_defect_model.h5"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    self.model = tf.keras.models.load_model(model_path)
                    print(f"Model loaded: {model_path}")
                    print(f"Model input shape: {self.model.input_shape}")
                    print(f"Model output shape: {self.model.output_shape}")
                    return True
                except Exception as e:
                    print(f"Error loading {model_path}: {e}")
                    continue
        
        print("No model found!")
        return False
    
    def analyze_data_distribution(self):
        """Analyze data distribution"""
        print("=== Data Distribution Analysis ===")
        
        class_counts = {}
        total_images = 0
        
        for class_name in self.class_names:
            class_path = os.path.join(self.data_path, 'resin', class_name)
            if os.path.exists(class_path):
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
                print(f"{class_name}: 0 images")
        
        print(f"Total images: {total_images}")
        
        # Check for class imbalance
        if total_images > 0:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
            
            if imbalance_ratio > 5:
                print("WARNING: Severe class imbalance detected!")
                print("This is likely causing poor accuracy for minority classes.")
        
        return class_counts
    
    def test_model_on_sample_images(self):
        """Test model on sample images from each class"""
        print("=== Testing Model on Sample Images ===")
        
        if self.model is None:
            print("No model loaded!")
            return
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.data_path, 'resin', class_name)
            if not os.path.exists(class_path):
                print(f"Class path not found: {class_path}")
                continue
            
            print(f"\nTesting {class_name} class:")
            
            # Find first image in this class
            sample_image = None
            for root, dirs, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(root, file)
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                sample_image = img
                                break
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
                if sample_image is not None:
                    break
            
            if sample_image is not None:
                # Preprocess image
                img_rgb = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, self.img_size)
                img_normalized = img_resized.astype(np.float32) / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                # Get prediction
                predictions = self.model.predict(img_batch, verbose=0)
                predicted_class_id = np.argmax(predictions)
                predicted_class = self.class_names[predicted_class_id]
                confidence = np.max(predictions)
                
                print(f"  True class: {class_name}")
                print(f"  Predicted class: {predicted_class}")
                print(f"  Confidence: {confidence:.4f}")
                print(f"  All predictions: {dict(zip(self.class_names, predictions[0]))}")
                
                # Check if prediction is correct
                if predicted_class == class_name:
                    print(f"  [CORRECT]")
                else:
                    print(f"  [WRONG]")
            else:
                print(f"  No sample image found for {class_name}")
    
    def analyze_model_predictions(self):
        """Analyze model predictions in detail"""
        print("=== Detailed Model Analysis ===")
        
        if self.model is None:
            print("No model loaded!")
            return
        
        # Test on multiple images from each class
        correct_predictions = 0
        total_predictions = 0
        class_accuracy = {name: {'correct': 0, 'total': 0} for name in self.class_names}
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.data_path, 'resin', class_name)
            if not os.path.exists(class_path):
                continue
            
            print(f"\nAnalyzing {class_name} class:")
            
            # Test on first 10 images
            test_count = 0
            for root, dirs, files in os.walk(class_path):
                for file in files:
                    if test_count >= 10:  # Limit to 10 images per class
                        break
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(root, file)
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                # Preprocess
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img_resized = cv2.resize(img_rgb, self.img_size)
                                img_normalized = img_resized.astype(np.float32) / 255.0
                                img_batch = np.expand_dims(img_normalized, axis=0)
                                
                                # Predict
                                predictions = self.model.predict(img_batch, verbose=0)
                                predicted_class_id = np.argmax(predictions)
                                predicted_class = self.class_names[predicted_class_id]
                                confidence = np.max(predictions)
                                
                                # Update counters
                                total_predictions += 1
                                class_accuracy[class_name]['total'] += 1
                                
                                if predicted_class == class_name:
                                    correct_predictions += 1
                                    class_accuracy[class_name]['correct'] += 1
                                    print(f"  [OK] {file}: {predicted_class} ({confidence:.3f})")
                                else:
                                    print(f"  [NG] {file}: {predicted_class} ({confidence:.3f}) - Expected: {class_name}")
                                
                                test_count += 1
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")
        
        # Calculate overall accuracy
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\n=== Overall Results ===")
        print(f"Total predictions: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Overall accuracy: {overall_accuracy:.4f}")
        
        # Calculate per-class accuracy
        print(f"\n=== Per-Class Accuracy ===")
        for class_name in self.class_names:
            if class_accuracy[class_name]['total'] > 0:
                accuracy = class_accuracy[class_name]['correct'] / class_accuracy[class_name]['total']
                print(f"{class_name}: {accuracy:.4f} ({class_accuracy[class_name]['correct']}/{class_accuracy[class_name]['total']})")
            else:
                print(f"{class_name}: No data")
    
    def identify_problems(self):
        """Identify potential problems with the model"""
        print("=== Problem Identification ===")
        
        # Check data distribution
        class_counts = self.analyze_data_distribution()
        
        # Check for common issues
        issues = []
        
        # 1. Class imbalance
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            if min_count > 0 and max_count / min_count > 5:
                issues.append("Severe class imbalance - minority classes have poor accuracy")
        
        # 2. Insufficient data
        total_images = sum(class_counts.values())
        if total_images < 1000:
            issues.append("Insufficient training data - need more images")
        
        # 3. Model complexity
        if self.model:
            param_count = self.model.count_params()
            if param_count > 1000000 and total_images < 5000:
                issues.append("Model too complex for available data - overfitting likely")
        
        # 4. Data quality
        if total_images > 0:
            avg_images_per_class = total_images / len(self.class_names)
            if avg_images_per_class < 100:
                issues.append("Too few images per class - need at least 100 per class")
        
        print("Identified issues:")
        if issues:
            for i, issue in enumerate(issues, 1):
                print(f"{i}. {issue}")
        else:
            print("No obvious issues identified")
        
        return issues
    
    def suggest_improvements(self, issues):
        """Suggest improvements based on identified issues"""
        print("\n=== Improvement Suggestions ===")
        
        suggestions = []
        
        for issue in issues:
            if "class imbalance" in issue:
                suggestions.append("Use class weights during training")
                suggestions.append("Apply data augmentation to minority classes")
                suggestions.append("Collect more data for minority classes")
            
            if "insufficient data" in issue:
                suggestions.append("Collect more training images")
                suggestions.append("Use data augmentation techniques")
                suggestions.append("Consider transfer learning")
            
            if "too complex" in issue:
                suggestions.append("Reduce model complexity")
                suggestions.append("Add more regularization (dropout, L2)")
                suggestions.append("Use early stopping")
            
            if "too few images" in issue:
                suggestions.append("Collect more images per class")
                suggestions.append("Use advanced data augmentation")
                suggestions.append("Consider synthetic data generation")
        
        if not suggestions:
            suggestions.append("Retrain with more epochs")
            suggestions.append("Tune hyperparameters")
            suggestions.append("Try different model architectures")
        
        print("Suggested improvements:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
    
    def run_analysis(self):
        """Run complete analysis"""
        print("=== Model Performance Analysis ===")
        
        # Load model
        if not self.load_model():
            return
        
        # Analyze data
        self.analyze_data_distribution()
        
        # Test model
        self.test_model_on_sample_images()
        
        # Detailed analysis
        self.analyze_model_predictions()
        
        # Identify problems
        issues = self.identify_problems()
        
        # Suggest improvements
        self.suggest_improvements(issues)

def main():
    """Main function"""
    analyzer = ModelPerformanceAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()