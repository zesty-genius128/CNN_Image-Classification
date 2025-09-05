#!/usr/bin/env python3
# Create a simple test model for immediate testing
import tensorflow as tf
from tensorflow import keras
import os

# CIFAR-10 classes
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def create_simple_model():
    """Create a simple CNN model for testing"""
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("Creating simple test model...")
    model = create_simple_model()
    
    # Create some dummy training data to initialize weights
    import numpy as np
    dummy_x = np.random.rand(100, 32, 32, 3)
    dummy_y = keras.utils.to_categorical(np.random.randint(0, 10, 100), 10)
    
    # Train for just one epoch to initialize weights properly
    print("Training for one epoch to initialize weights...")
    model.fit(dummy_x, dummy_y, epochs=1, verbose=0)
    
    # Save the model
    model_path = 'best_cnn_model.h5'
    model.save(model_path)
    print(f"Test model saved as {model_path}")
    print("Note: This is just a test model. Train the full model using problem1_cnn_classification.py")

if __name__ == '__main__':
    main()