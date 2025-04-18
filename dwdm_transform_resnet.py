import numpy as np
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical

features = np.load("resnet_features.npy")
y_train = np.load("y_train.npy")
y_labels = np.argmax(y_train, axis=1)

# PCA to reduce dimensions
pca = PCA(n_components=200)
X_pca = pca.fit_transform(features)

# SMOTE to balance classes
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_pca, y_labels)
y_resampled_oh = to_categorical(y_resampled, num_classes=4)

# Save new balanced dataset
np.save("X_train_dwdm.npy", X_resampled)
np.save("y_train_dwdm.npy", y_resampled_oh)
