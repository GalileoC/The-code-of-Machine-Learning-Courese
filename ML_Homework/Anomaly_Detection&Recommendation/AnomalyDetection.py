import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white", palette=sns.color_palette("RdBu"), color_codes=False)

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats  # 统计模块
from sklearn.model_selection import train_test_split

data = sio.loadmat('./data/ex8data1.mat')
X = data['X']
X_val, X_test, y_val, y_test = train_test_split(data.get('Xval'),
                                            data.get('yval').ravel(),
                                            test_size=0.5)
# print(X.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)  # (307, 2) (153, 2) (153,) (154, 2) (154,)

# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], s=10, cmap=plt.cm.Spectral)
# plt.show()

# estimate multivariate Gaussian parameters  μ  and  σ2
mu = X.mean(axis=0)  # 均值
# print(mu, '\n')

cov = np.cov(X.T)  # 协方差矩阵，计算协方差矩阵，输入矩阵的shape格式为(特征数n, 样本数m)
# print(cov)

# example of creating 2d grid to calculate probability density
# np.dstack(np.mgrid[0:3, 0:3])

# create multi-var Gaussian model
multi_normal = stats.multivariate_normal(mu, cov)  # 多变量正态分布

# create a grid
# x, y = np.mgrid[0:30:0.01, 0:30:0.01]
# pos = np.dstack((x, y))
#
# fig, ax = plt.subplots()
#
# # plot probability density
# ax.contourf(x, y, multi_normal.pdf(pos), cmap='Blues')
#
# # plot original data points
# sns.regplot('Latency', 'Throughput',
#            data=pd.DataFrame(X, columns=['Latency', 'Throughput']),
#            fit_reg=False,
#            ax=ax,
#            scatter_kws={"s":10,
#                         "alpha":0.4})
# plt.show()

# select threshold  ϵ
from sklearn.metrics import f1_score, classification_report

# def select_threshold(X, X_val, y_val):
#     """use CV data to find the best epsilon
#     Returns:
#         e: best epsilon with the highest f-score
#         f-score: such best f-score
#     """
#     # create multivariate model using training data
#     mu = X.mean(axis=0)
#     cov = np.cov(X.T)
#     multi_normal = stats.multivariate_normal(mu, cov)
#
#     # this is key, use CV data for fine tuning hyper parameters
#     pval = multi_normal.pdf(X_val)
#
#     # set up epsilon candidates
#     epsilon = np.linspace(np.min(pval), np.max(pval), num=10000)
#
#     # calculate f-score
#     fs = []
#     for e in epsilon:
#         y_pred = (pval <= e).astype('int')
#         fs.append(f1_score(y_val, y_pred))
#
#     # find the best f-score
#     argmax_fs = np.argmax(fs)
#
#     return epsilon[argmax_fs], fs[argmax_fs]

# e, fs = select_threshold(X, X_val, y_val)
# print('Best epsilon: {}\nBest F-score on validation data: {}'.format(e, fs))  # 3.6148577562381784e-05

# visualize prediction of Xval using learned  ϵ
def select_threshold(X, X_val, y_val):
    """use CV data to find the best epsilon
    Returns:
        e: best epsilon with the highest f-score
        f-score: such best f-score
    """
    # create multivariate model using training data
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    multi_normal = stats.multivariate_normal(mu, cov)

    # this is key, use CV data for fine tuning hyper parameters
    pval = multi_normal.pdf(X_val)

    # set up epsilon candidates
    epsilon = np.linspace(np.min(pval), np.max(pval), num=10000)

    # calculate f-score
    fs = []
    for e in epsilon:
        y_pred = (pval <= e).astype('int')
        fs.append(f1_score(y_val, y_pred))

    # find the best f-score
    argmax_fs = np.argmax(fs)

    return epsilon[argmax_fs], fs[argmax_fs]

def predict(X, X_val, e, X_test, y_test):
    """with optimal epsilon, combine X, Xval and predict Xtest
    Returns:
        multi_normal: multivariate normal model
        y_pred: prediction of test data
    """
    Xdata = np.concatenate((X, X_val), axis=0)

    mu = Xdata.mean(axis=0)
    cov = np.cov(Xdata.T)
    multi_normal = stats.multivariate_normal(mu, cov)

    # calculate probability of test data
    pval = multi_normal.pdf(X_test)
    y_pred = (pval <= e).astype('int')

    print(classification_report(y_test, y_pred))

    return multi_normal, y_pred

# e, fs = select_threshold(X, X_val, y_val)
#
# multi_normal, y_pred = predict(X, X_val, e, X_test, y_test)

# construct test DataFrame
# data = pd.DataFrame(X_test, columns=['Latency', 'Throughput'])
# data['y_pred'] = y_pred
#
# # create a grid for graphing
# x, y = np.mgrid[0:30:0.01, 0:30:0.01]
# pos = np.dstack((x, y))
#
# fig, ax = plt.subplots()
#
# # plot probability density
# ax.contourf(x, y, multi_normal.pdf(pos), cmap='Blues')
#
# # plot original Xval points
# sns.regplot('Latency', 'Throughput',
#             data=data,
#             fit_reg=False,
#             ax=ax,
#             scatter_kws={"s":10,
#                          "alpha":0.4})
#
# # mark the predicted anamoly of CV data. We should have a test set for this...
# anamoly_data = data[data['y_pred']==1]
# ax.scatter(anamoly_data['Latency'], anamoly_data['Throughput'], marker='x', s=50)
# plt.show()


# high dimension data
data2 = sio.loadmat('./data/ex8data2.mat')
X2 = data2['X']
X_val2, X_test2, y_val2, y_test2 = train_test_split(data2.get('Xval'),
                                            data2.get('yval').ravel(),
                                            test_size=0.5)
# print(X2.shape, X_val2.shape, y_val2.shape, X_test2.shape, y_test2.shape)  # (1000, 11) (50, 11) (50,) (50, 11) (50,)
e, fs = select_threshold(X2, X_val2, y_val2)
# print('Best epsilon: {}\nBest F-score on validation data: {}'.format(e, fs))  # Best epsilon: 5.045923173851731e-19
multi_normal, y_pred2 = predict(X2, X_val2, e, X_test2, y_test2)
print('find {} anamolies'.format(y_pred2.sum()))














