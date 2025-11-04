#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a dummy model for testing purposes
"""

import tensorflow as tf
import json
import os

def create_dummy_model():
    """Create a dummy model for testing"""
    
    # Create a simple CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 classes
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Create dummy data to train the model
    import numpy as np
    dummy_x = np.random.random((100, 224, 224, 3))
    dummy_y = np.random.randint(0, 4, (100,))
    dummy_y = tf.keras.utils.to_categorical(dummy_y, 4)
    
    # Train for a few epochs
    model.fit(dummy_x, dummy_y, epochs=1, verbose=0)
    
    # Save the model
    model.save('balanced_resin_washer_model.h5')
    print("Dummy model saved as 'balanced_resin_washer_model.h5'")
    
    # Create class names file
    class_names = ['good', 'chipping', 'black_spot', 'scratch']
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    print("Class names saved as 'class_names.json'")
    
    return model

if __name__ == "__main__":
    create_dummy_model()
