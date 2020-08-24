# 二分类问题

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn import svm

raw_data = loadmat('./data/ex6data1.mat')
X = raw_data['X']
y = raw_data['y']
# print(X.shape)  # (51, 2)
# print(y.shape)  # (51, 1)

# plt.figure()
# plt.scatter(X[:,0], X[:,1], c=y.ravel(), s=20, cmap=plt.cm.Spectral)
# plt.show()

svc1 = svm.LinearSVC(C=1, loss='hinge')
svc1.fit(X, y.ravel())
score1 = svc1.score(X, y)
# print(score1)  # 0.9803921568627451

df1 = svc1.decision_function(X)
# fig, ax = plt.subplots(figsize=(8,6))
# ax.scatter(X[:,0], X[:,1], s=50, c=df1, cmap='RdBu')
# ax.set_title('SVM (C=1) Decision Confidence')
# plt.show()

# C=100
svc100 = svm.LinearSVC(C=100, loss='hinge')
svc100.fit(X, y.ravel())
score100 = svc100.score(X, y)
# print(score100)  # 0.9803921568627451
df100 = svc100.decision_function(X)

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(X[:,0], X[:,1], s=50, c=df100, cmap='RdBu')
ax.set_title('SVM (C=100) Decision Confidence')
plt.show()



