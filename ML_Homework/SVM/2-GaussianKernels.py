# 高斯函数实现

import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio

def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.square(x1-x2).sum() / (2 * np.square(sigma)))

data = sio.loadmat('./data/ex6data2.mat')
X = data['X']
y = data['y']
# print(X.shape)  # (863, 2)
# print(y.shape)  # (863, 1)

# plt.figure()
# plt.scatter(X[:,0], X[:,1], c=y.ravel(), s=10, cmap=plt.cm.Spectral)
# plt.show()

svc = svm.SVC(C=100, kernel='rbf', gamma=10, probability=True)  # rbf：高斯核函数（Gaussian kernel），也称径向基 (RBF) 函数
svc.fit(X, y.ravel())
score = svc.score(X, y)
# print(score)  # 0.9698725376593279

predict_prob = svc.predict_proba(X)[:, 0]

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(X[:,0], X[:,1], s=10, c=predict_prob, cmap='Reds')
plt.show()
