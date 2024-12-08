# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:49:33 2024

@author: prash
"""
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Input,
    GlobalAveragePooling2D,
)

def build_model(img_size=224, num_classes=18):
    """
    Builds and compiles a CNN model for multi-label classification.

    Args:
        img_size (int): Input image size (default: 224).
        num_classes (int): Number of classes for classification.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    # Example: Simple CNN model
    model = Sequential([
        Input(shape=(img_size, img_size, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')  # Sigmoid for multi-label classification
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # Multi-label loss
        metrics=['accuracy']
    )
    
    return model

