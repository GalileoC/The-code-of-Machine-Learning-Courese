# 数据可视化

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.io as sio

mat = sio.loadmat('./data/ex7data1.mat')
# print(mat.keys())  # X
X = mat['X']
# print(X.shape)  # (50, 2)

# plt.figure()
# plt.scatter(X[:,0], X[:,1], s=20)
# plt.show()

mat2 = sio.loadmat('./data/ex7data2.mat')
# print(mat2.keys())  # X
X2 = mat2['X']
# print(X2.shape)  # (300, 2)

plt.figure()
plt.scatter(X2[:,0], X2[:,1], s=20)
plt.show()

















