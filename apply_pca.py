import numpy as np
from sklearn.decomposition import PCA

# Load the dataset
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# Flatten images for PCA
X_flat = X_train.reshape(len(X_train), -1)

# Apply PCA to reduce features to 500
pca = PCA(n_components=500)
X_pca = pca.fit_transform(X_flat)

# Save PCA output
np.save("X_pca.npy", X_pca)
np.save("y_train_labels.npy", y_train.argmax(axis=1))  # for SMOTE or MLP
