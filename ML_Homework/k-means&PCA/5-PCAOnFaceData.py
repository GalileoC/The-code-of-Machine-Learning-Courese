import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white")

import numpy as np
import pandas as pd
import scipy.io as sio

data = sio.loadmat('./data/ex7faces.mat')
X = data['X']
# print(X.shape)  # (5000, 1024)

# support functions ---------------------------------------
def plot_n_image(X, n):
    """ plot first n images
    n has to be a square number
    """
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    first_n_images = X[:n, :]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,
                                    sharey=True, sharex=True, figsize=(8, 8))

    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)).T)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

# PCA functions ---------------------------------------
def covariance_matrix(X):
    """
    Args:
        X (ndarray) (m, n)
    Return:
        cov_mat (ndarray) (n, n):
            covariance matrix of X
    """
    m = X.shape[0]

    return (X.T @ X) / m

def normalize(X):  # 对每一个特征进行归一化
    """
        for each column, X-mean / std
    """
    X_copy = X.copy()
    m, n = X_copy.shape

    for col in range(n):
        X_copy[:, col] = (X_copy[:, col] - X_copy[:, col].mean()) / X_copy[:, col].std()

    return X_copy

def pca(X):
    """
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
    Args:
        X ndarray(m, n)
    Returns:
        U ndarray(n, n): principle components
    """
    # 1. normalize data
    X_norm = normalize(X)

    # 2. calculate covariance matrix
    Sigma = covariance_matrix(X_norm)  # (n, n)

    # 3. do singular value decomposition
    # remeber, we feed cov matrix in SVD, since the cov matrix is symmetry
    # left sigular vector and right singular vector is the same, which means
    # U is V, so we could use either one to do dim reduction
    U, S, V = np.linalg.svd(Sigma)  # U: principle components (n, n)

    return U, S, V

def project_data(X, U, k):
    """
    Args:
        U (ndarray) (n, n)
    Return:
        projected X (n dim) at k dim
    """
    m, n = X.shape

    if k > n:
        raise ValueError('k should be lower dimension of n')

    return X @ U[:, :k]

def recover_data(Z, U):
    m, n = Z.shape

    if n >= U.shape[0]:
        raise ValueError('Z dimension is >= U, you should recover from lower dimension to higher')

    return Z @ U[:, :n].T

# plot_n_image(X, n=64)
# plt.show()

# run PCA, find principle components
U, _, _ = pca(X)
# print(U.shape)  # (1024, 1024)
# didn't see face in principle components
# plot_n_image(U, n=36)
# plt.show()

# reduce dimension to k=100
# no face in projected dimension too
# Z = project_data(X, U, k=100)
# plot_n_image(Z, n=64)
# plt.show()

# recover from k=100
# X_recover = recover_data(Z, U)
# plot_n_image(X_recover, n=64)
# plt.show()


# sklearn实现
from sklearn.decomposition import PCA
sk_pca = PCA(n_components=100)
Z = sk_pca.fit_transform(X)
# print(Z.shape)  # (5000, 100)
# plot_n_image(Z, 64)
# plt.show()
X_recover = sk_pca.inverse_transform(Z)
# print(X_recover.shape)  # (5000, 1024)
plot_n_image(X_recover, n=64)
plt.show()


