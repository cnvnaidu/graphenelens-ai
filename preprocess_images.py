import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Set path
DATASET_PATH = r"E:\DWDM PROJECT\archive\seg_dataset"
CATEGORIES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
IMG_SIZE = 224

def load_data():
    data = []
    labels = []

    for i, category in enumerate(CATEGORIES):
        folder_path = os.path.join(DATASET_PATH, category)
        for file in tqdm(os.listdir(folder_path), desc=f"Loading {category}"):
            img_path = os.path.join(folder_path, file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(i)
            except Exception as e:
                print(f"Error loading image: {img_path}, {e}")

    data = np.array(data) / 255.0  # Normalize
    labels = to_categorical(np.array(labels), num_classes=len(CATEGORIES))

    return data, labels

X, y = load_data()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save as .npy for reusability
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
