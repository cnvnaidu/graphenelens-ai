import numpy as np
from sklearn.decomposition import PCA

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_flat = X_train.reshape(X_train.shape[0], -1)
pca = PCA(n_components=500)
X_pca = pca.fit_transform(X_flat)

np.save("X_pca.npy", X_pca)
