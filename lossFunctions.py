# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:53:27 2024

@author: prash
"""

from preprocessing import load_data
from loss_function import custom_loss_function
from tensorflow.keras.models import Sequential  # Import TensorFlow here for the main workflow
from config import TRAIN_EXCEL_PATH, TEST_EXCEL_PATH, IMAGE_FOLDER

# Load data
train_images, train_labels, _ = load_data(TRAIN_EXCEL_PATH, IMAGE_FOLDER)

# Build model, compile with custom loss, and train
model = Sequential([...])  # Define model
model.compile(optimizer="adam", loss=custom_loss_function)
model.fit(train_images, train_labels, epochs=10)


