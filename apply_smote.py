import numpy as np
from imblearn.over_sampling import SMOTE

# Load PCA reduced features
X_pca = np.load("X_pca.npy")
y_train = np.load("y_train_labels.npy")  # already in class indices

# Apply SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_pca, y_train)

# Save balanced data
np.save("X_resampled.npy", X_resampled)
np.save("y_resampled.npy", y_resampled)
