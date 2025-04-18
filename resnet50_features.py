import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf

# Load data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# Preprocess input as per ResNet50
X_train = preprocess_input(X_train * 255.0)

# Load ResNet50 (no top layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
model = Model(inputs=base_model.input, outputs=x)

# Extract features
features = model.predict(X_train, batch_size=32, verbose=1)
np.save("resnet_features.npy", features)
