# 二维Kmeans

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.io as sio

mat = sio.loadmat('./data/ex7data2.mat')
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
# print(data.info())
X = mat['X']

# plt.figure()
# plt.scatter(X[:,0], X[:,1], s=20)
# plt.show()

# random initialization
def combine_data_C(data, C):  # 为每个样本增加一个样本类型标签
    data_with_c = data.copy()
    data_with_c['C'] = C
    return data_with_c

# k-means fn --------------------------------
def random_init(data, k):  # 质心随机初始化
    """choose k sample from data set as init centroids
    Args:
        data: DataFrame
        k: int
    Returns:
        k samples: ndarray
    """
    return data.sample(k).values  # DataFrame.sample(k): 随机抽取k个样本

def _find_your_cluster(x, centroids):  # 计算每个样本到质心的距离并返回距离最小的索引，即所属簇的索引
    """find the right cluster for x with respect to shortest distance
    Args:
        x: ndarray (n, ) -> n features
        centroids: ndarray (k, n)
    Returns:
        k: int
    """
    distances = np.apply_along_axis(func1d=np.linalg.norm,  # this give you l2 norm
                                    axis=1,
                                    arr=centroids - x)  # use ndarray's broadcast
    # np.apply_along_axis: 将arr数组的每一个元素经过func函数变换形成的一个新数组
    return np.argmin(distances)  # 返回最小值的索引

def assign_cluster(data, centroids):  # 将每个样本分配到距离最近的簇，返回簇心索引数组
    """assign cluster for each node in data
    return C ndarray
    """
    return np.apply_along_axis(lambda x: _find_your_cluster(x, centroids),
                               axis=1,
                               arr=data.values)

def new_centroids(data, C):
    data_with_c = combine_data_C(data, C)

    return data_with_c.groupby('C', as_index=False).\
                       mean().\
                       sort_values(by='C').\
                       drop('C', axis=1).\
                       values

def cost(data, centroids, C):
    m = data.shape[0]

    expand_C_with_centroids = centroids[C]

    distances = np.apply_along_axis(func1d=np.linalg.norm,
                                    axis=1,
                                    arr=data.values - expand_C_with_centroids)
    return distances.sum() / m

def _k_means_iter(data, k, epoch=100, tol=0.0001):
    """one shot k-means
    with early break
    """
    centroids = random_init(data, k)
    cost_progress = []

    for i in range(epoch):
        print('running epoch {}'.format(i))

        C = assign_cluster(data, centroids)
        centroids = new_centroids(data, C)
        cost_progress.append(cost(data, centroids, C))

        if len(cost_progress) > 1:  # early break
            if (np.abs(cost_progress[-1] - cost_progress[-2])) / cost_progress[-1] < tol:
                break

    return C, centroids, cost_progress[-1]

def k_means(data, k, epoch=100, n_init=10):
    """do multiple random init and pick the best one to return
    Args:
        data (pd.DataFrame)
    Returns:
        (C, centroids, least_cost)
    """

    tries = np.array([_k_means_iter(data, k, epoch) for _ in range(n_init)])

    least_cost_idx = np.argmin(tries[:, -1])

    return tries[least_cost_idx]

# cluster assignment
# best_C, best_centroids, least_cost = k_means(data, 3)
# print(least_cost)
# data_with_c = combine_data_C(data, best_C)
# sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)
# plt.show()

# sklearn的Kmeans模块
from sklearn.cluster import KMeans
sk_means = KMeans(n_clusters=3)
sk_means.fit(data)
sk_C = sk_means.predict(data)
data_with_c = combine_data_C(data, sk_C)
sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)
plt.show()


