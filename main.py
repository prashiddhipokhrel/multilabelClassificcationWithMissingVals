# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:53:05 2024

@author: prash
"""
import os 
os.chdir(r"/kaggle/working/multilabelClassificcationWithMissingVals")

import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from config import TRAIN_EXCEL_PATH, TEST_EXCEL_PATH, IMAGE_FOLDER
from preprocessing import load_data
from model import build_model


# check the GPU availability 
gpus = tf.config.list_physical_devices('GPU')
if gpus: 
    try: # Set memory growth to True
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU is set up with memory growth.")
    except RuntimeError as e:
        print(e, "Failed to set memory growth.")
else:
    print("No GPU found. Proceeding with CPU.")



# load the preprocessed data from the script preprocessing.py
train_images, train_labels, train_captions = load_data(TRAIN_EXCEL_PATH, IMAGE_FOLDER)
# Load and preprocess test data
test_images, _, test_captions = load_data(TEST_EXCEL_PATH, IMAGE_FOLDER)

print(f"Test Images Shape: {test_images.shape}")
print(f"Captions Example: {test_captions[:5]}")


# Check shapes of the returned data
print(f"Images shape: {train_images.shape}")
print(f"Labels shape: {train_labels.shape}")
print(f"Captions sample: {train_captions[:5]}")

# Visualize preprocessed data
def visualize_data(images, labels, captions, num_samples=5):
    """
    Visualizes a few samples from the dataset.
    
    Args:
        images (np.array): Preprocessed images.
        labels (np.array): Multi-hot encoded labels.
        captions (list): Corresponding captions.
        num_samples (int): Number of samples to visualize.
    """
    for i in range(num_samples):
        plt.figure(figsize=(4, 4))
        plt.imshow(images[i])
        plt.title(f"Labels: {np.where(labels[i] == 1)[0] + 1}\nCaption: {captions[i]}")
        plt.axis("off")
        plt.show()

# Visualize the first 5 samples
visualize_data(train_images, train_labels, train_captions)

with tf.device('/GPU:0' if gpus else '/CPU:0'):
    # Build the model
    model = build_model(img_size=224, num_classes=18)
    model.summary()
    
    # Train the model
    history = model.fit(train_images, train_labels, validation_split=0.2, epochs=2, batch_size=32)
    
    # Evaluate or predict with the test data
    predictions = model.predict(test_images)

#plot the training metrics dynamically
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Binary Crossentropy Loss Over Epochs')
plt.show()



