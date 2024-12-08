# -*- coding: utf-8 -*-
"""
@author: prash
"""

import os 
import pandas as pd 
import numpy as np 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from config import TRAIN_EXCEL_PATH, TEST_EXCEL_PATH, IMAGE_FOLDER
from config import IMG_SIZE, NUM_CLASSES

# --- LOAD DATA ---
def load_data(excel_path, image_folder):
    """
    Loads and preprocesses the dataset, handling missing labels or captions.

    Parameters:
        excel_path (str): Path to the Excel file containing metadata (image names, labels, captions).
        image_folder (str): Folder containing the image files.

    Returns:
        Tuple: Preprocessed images, multi-hot encoded labels (if available), and captions (if available).
    """
    # Load Excel file
    df = pd.read_csv(excel_path)  #  columns: 'ImageID', 'Labels', 'Caption'
    print(f"Loaded {len(df)} rows from Excel file.")

    images = []
    labels = []
    captions = []

    # Check if 'Labels' column exists
    has_labels = "Labels" in df.columns

    # Multi-label binarizer for label encoding
    if has_labels:
        mlb = MultiLabelBinarizer(classes=list(range(1, 19)))  # 1-19, exclude 12
        mlb.fit([[1]])  # Dummy fit to initialize classes

    for _, row in df.iterrows():
        # Check if the image file exists
        img_path = os.path.join(image_folder, row["ImageID"])
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}, skipping...")
            continue

        # Load and preprocess image using TensorFlow
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        images.append(img_array)

        # Process labels (only if they exist)
        if has_labels:
            if pd.isnull(row["Labels"]):  # Skip if no labels
                print(f"No labels for image {row['ImageID']}, skipping...")
                continue
            label_list = list(map(int, str(row["Labels"]).split()))  # Split space-separated labels
            labels.append(label_list)

        # Process caption (optional)
        if "Caption" in row and not pd.isnull(row["Caption"]):
            captions.append(row["Caption"])
        else:
            captions.append("")

    # Convert labels to multi-hot encoding if they exist
    if has_labels:
        labels_encoded = mlb.transform(labels)
    else:
        labels_encoded = None

    return np.array(images), labels_encoded, np.array(captions)

# Create a TensorFlow Dataset
def create_tf_dataset(images, labels, captions, batch_size=32):
    def gen():
        for i in range(len(images)):
            yield images[i], labels[i], captions[i]

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.string),
        )
    )

    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Usage
os.chdir(r"/kaggle/working/multilabelClassificcationWithMissingVals")

# Load data
train_images, train_labels, train_captions = load_data(TRAIN_EXCEL_PATH, IMAGE_FOLDER)
test_images, _, test_captions = load_data(TEST_EXCEL_PATH, IMAGE_FOLDER)

# Create TensorFlow datasets
train_dataset = create_tf_dataset(train_images, train_labels, train_captions)
test_dataset = create_tf_dataset(test_images, None, test_captions)

# Print shapes for verification
print(f"Train dataset: {train_dataset}")
print(f"Test dataset: {test_dataset}")
