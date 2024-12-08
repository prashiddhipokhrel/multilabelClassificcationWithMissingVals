# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:14:19 2024

@author: prash
"""

# Absolute base directory
BASE_DIR = r"/kaggle/input/multi-label-classification-competition-2024/COMP5329S1A2Dataset"

# Absolute paths for each variable
TRAIN_EXCEL_PATH = f"{BASE_DIR}/train.csv"
TEST_EXCEL_PATH = f"{BASE_DIR}/test.csv"
IMAGE_FOLDER = f"{BASE_DIR}/data"

# Model configuration
IMG_SIZE = 224  # Image size for resizing
NUM_CLASSES = 19  # Number of classes for multi-label classification



